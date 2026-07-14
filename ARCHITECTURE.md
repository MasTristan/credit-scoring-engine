# Architecture

Technical design notes for the ML Credit Scoring Engine.

---

## Module map

```
src/data_prep.py   ──▶  data/processed/{train,test}.parquet, feature_names.json
src/train.py       ──▶  models/xgboost_model.json, feature_importance.csv,
                        training_metrics.json, shap_background.parquet,
                        feature_names.json
src/predict.py     ──▶  used by streamlit_app.py at inference time
src/explain.py     ──▶  used by streamlit_app.py for SHAP explanations
streamlit_app.py        the public application (5 tabs)
tests/                  pytest suite (15 tests)
```

The four modules are deliberately small (≤ 200 LoC each). The Streamlit
app composes them; it does not duplicate any logic.

---

## Data flow

```
data/raw/uci_credit_card.csv     (30,000 × 24)
              │
              │  load_raw → clean_categoricals → engineer_features
              │  → encode_categoricals → stratified 80/20 split
              ▼
data/processed/{train,test}.parquet                 (30,000 × 35)
data/processed/feature_names.json                   (34 names)
data/sample/sample_1000.csv                         (1,000 × 24, public)
              │
              │  XGBClassifier.fit(..., early_stopping_rounds=50)
              ▼
models/xgboost_model.json                           (~900 KB)
models/feature_importance.csv                       (gain, weight, cover, rank)
models/training_metrics.json                        (ROC-AUC, Gini, KS, ...)
models/shap_background.parquet                      (500 stratified rows)
              │
              │  load_model + get_explainer + predict_proba + compute_shap
              ▼
streamlit_app.py
```

---

## Why these choices

### XGBoost (`tree_method="hist"`)

- Handles mixed numeric / boolean inputs with no pre-scaling.
- `tree_method="hist"` is 5–10× faster than the exact tree builder and is
  the default for XGBoost ≥ 2.x.
- Single-file JSON serialisation, easy to ship in a public repo and load
  on Streamlit Community Cloud without any extra storage layer.
- Native SHAP support through `TreeExplainer`.

### Class imbalance left untouched (calibrated PDs)

The default rate is 22.12%. The model trains on the natural class
distribution: no `scale_pos_weight` and no resampling. This keeps the
predicted PDs calibrated as long-run default frequencies (test ECE ≈ 0.01),
which is the point of a PD model. Re-weighting the positive class
(`scale_pos_weight = n_neg / n_pos ≈ 3.52`) was benchmarked and rejected: it
inflated every PD to roughly twice the base rate without improving ranking,
and a post-hoc isotonic pass did not beat the native model on validation.

### Stratified 60/20/20 split (not temporal)

The UCI dataset only contains six consecutive months of repayment history
and a single forward-looking target ("default next month"). It has no
contract-issue date, so a meaningful split must be random. The split is
stratified on `IS_DEFAULT` (seed = 42), preserving the 22.12% default rate
across train, validation, and test. Early stopping and the operating
threshold are fixed on validation; the test set is scored once. The original
project brief targeted Lending Club 2015–2018 with a temporal
train = 2015–2017 / test = 2018 split; that intent is documented in
`CLAUDE.md`.

### SHAP `tree_path_dependent`

The `interventional` mode requires a background dataset and yields
approximate Shapley values; the `tree_path_dependent` mode walks the trees
using node-cover frequencies as conditional weights and gives the exact
Shapley values that satisfy the local-accuracy property:

> `sum(shap_values) + base_value == logit(predict_proba)`

This is the property that lets us claim, in a governance setting, that
the explanation **is** the prediction.

### Streamlit + cached resources

- `@st.cache_resource` for the model + explainer (heavy, immutable across
  sessions).
- `@st.cache_data` for the sample CSV and the scored sample (cheap to
  recompute but used in three tabs).
- A single 5-tab layout instead of multi-page navigation, so the model
  artefacts are loaded exactly once per session.

---

## Inference path (Tab 1, "Individual Scorer")

```
form inputs (12 widgets)
   │
   │  dict → 1-row DataFrame matching the raw schema
   ▼
src.data_prep.build_feature_matrix
   │     (same code path used for training and for the 1,000-row sample)
   ▼
src.predict.align_features        (adds zero-filled missing columns,
                                   re-orders to feature_names)
   ▼
model.predict_proba              ─▶ PD
score_to_rating, score_to_risk_band
explainer.shap_values            ─▶ waterfall (top 15 by |φ|)
```

Using the same `build_feature_matrix` function for training, batch
inference, and the single-row form guarantees there is **no skew** between
training-time and inference-time feature engineering. This is the silent
killer of most ML projects in production; here it is enforced by
construction (one function, two callers).

---

## Performance characteristics

| Operation                              | Time on a free Streamlit instance |
|----------------------------------------|------------------------------------|
| Cold start (artefacts load)            | ≈ 4 s                              |
| Single-row inference (PD only)         | < 5 ms                             |
| Single-row inference + SHAP waterfall  | ≈ 50 ms                            |
| Batch scoring of 1,000 contracts        | ≈ 200 ms                           |
| Global SHAP on 200-row subset          | ≈ 700 ms                           |
| `python -m src.train` end-to-end       | ≈ 8 s (CPU)                        |

---

## Reproducibility

- All randomness seeded with `random_state=42` (`train_test_split`, XGBoost,
  SHAP background sample).
- Pinned dependencies in `requirements.txt`.
- Model artefacts committed to the repo: anyone cloning the repo runs the
  app with the exact model that produced the metrics in `training_metrics.json`.
- `python -m src.data_prep && python -m src.train` reproduces those
  artefacts bit-for-bit from the raw CSV.

---

## Limitations

- The training data is Taiwanese credit-card data from 2005; the model
  must be recalibrated before being applied to any other population.
- Six months of payment history is short, `PAY_0` and `PAY_MAX` dominate
  the global feature importance.
- The probability output **is** calibrated to long-run frequencies (test
  ECE ≈ 0.01) because the model trains on the natural distribution; the
  remaining calibration risk is drift over time, monitored via the
  reliability diagram with isotonic recalibration held in reserve.
- No reject inference: the dataset only contains accepted contracts, so
  the model inherits the original lender's selection bias.

These are also discussed in the "About this model" tab in the app.
