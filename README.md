# ML Credit Scoring Engine, XGBoost + SHAP + Streamlit

![CI](https://github.com/MasTristan/credit-scoring-engine/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Community%20Cloud-FF4B4B.svg)

XGBoost credit-scoring model with full SHAP explainability, individual
scorer, portfolio analysis, and model performance dashboard. Trained on the
UCI *Default of Credit Card Clients* dataset (Taiwan, 2005).

> **Live demo**: https://tm-credit-scoring-engine.streamlit.app/

---

## Key features

- **Individual Scorer**, enter a borrower's profile and get a probability
  of default, an internal rating (Project 1 mapping AAA/AA → D), a risk band
  (LOW/MEDIUM/HIGH), a **three-tier decision** (auto-approve / manual review /
  auto-decline), a SHAP waterfall of the 15 most influential features,
  **plain-language reason codes** (GDPR Art. 22), and a **counterfactual
  explanation** showing the smallest change that would flip the decision.
- **Decision policy**, instead of one hard threshold, two PD cut-offs fit on
  validation split applicants into auto-approve (PD < 14%, ~9% default),
  manual review (the grey zone), and auto-decline (PD ≥ 36%, ~59% default).
  The confident tails are auto-decided; the uncertain middle goes to a human,
  which is how origination actually works and why a raw false-positive count at
  a single threshold is the wrong lens on the model.
- **Portfolio Analysis**, load 1,000 contracts from the public hold-out
  sample, score the batch and visualise PD distribution, scatter against
  credit limit, breakdown by current repayment status, and global SHAP
  importance.
- **Model Performance**, KPI row from the training run, ROC and PR
  curves, score distribution by actual outcome (KS plot), full metrics
  table, and an interactive **cost-sensitive thresholding** panel that
  turns the confusion matrix into a euro-denominated P&L.
- **Methodology**, recruiter-facing tab with the maths behind XGBoost and
  SHAP, justified technical choices, and "what I would do next in
  production".
- **Governance**, business case, reliability diagram showing the model is
  calibrated by construction (ECE ≈ 0.01, isotonic overlay kept as a drift
  remedy), fairness audit (DI / EOD / per-group AUC) on SEX, AGE band, and
  EDUCATION, monitoring-dashboard mock-up, and links to the full governance
  pack in `docs/`.

---

## Model metrics (hold-out test set, n=6,000)

| Metric        | Value  | Benchmark (consumer credit) |
|---------------|--------|------------------------------|
| ROC-AUC       | 0.781  | ≥ 0.72                       |
| Gini          | 0.561  | ≥ 0.44                       |
| KS statistic  | 0.426  | ≥ 0.35                       |
| PR-AUC        | 0.557  | varies with default rate     |
| Brier score   | 0.135  | lower is better              |
| ECE           | 0.010  | lower is better              |

Model selection (early stopping) and the operating threshold are fixed on a
dedicated validation split; the test set is scored once, so these figures are
an unbiased out-of-sample estimate. Split: stratified 60 / 20 / 20
(train / validation / test, seed = 42).

**Calibration.** The model trains on the natural class distribution (no
`scale_pos_weight`, no resampling), so its predicted PDs are calibrated as
long-run default frequencies: mean predicted PD 0.221 against a 0.221 base
rate, ECE 0.010. Class re-weighting was benchmarked and rejected, it doubled
every PD versus the base rate (ECE 0.19) without improving ranking, and a
post-hoc isotonic pass did not beat the natively calibrated model on
validation.

---

## Repository structure

```
LendingClubCreditScoring/
├── CLAUDE.md                     - project brief
├── README.md                     - this file
├── ARCHITECTURE.md               - technical design
├── LICENSE                       - MIT
├── requirements.txt
├── data/
│   ├── raw/                      - not committed (place CSV here)
│   ├── processed/                - not committed
│   └── sample/sample_1000.csv    - 1,000-row public demo sample
├── models/
│   ├── xgboost_model.json
│   ├── feature_names.json
│   ├── feature_importance.csv
│   ├── shap_background.parquet
│   └── training_metrics.json
├── docs/                         - governance pack (12 documents)
│   ├── BUSINESS_CASE.md
│   ├── PERSONAS_AND_USER_STORIES.md
│   ├── RACI.md
│   ├── PROCESS_FLOW.md
│   ├── ROADMAP.md
│   ├── REGULATORY_MAPPING.md
│   ├── MODEL_CARD.md
│   ├── RISK_REGISTER.md
│   ├── DATA_DICTIONARY.md
│   ├── MONITORING_PLAN.md
│   ├── AB_TEST_DESIGN.md
│   └── GLOSSARY.md
├── src/
│   ├── data_prep.py              - cleaning + feature engineering
│   ├── train.py                  - XGBoost training + metrics
│   ├── predict.py                - inference + rating mapping
│   ├── explain.py                - SHAP helpers
│   ├── calibration.py            - reliability diagram + isotonic
│   ├── fairness.py               - DI / EOD / per-group AUC
│   ├── counterfactuals.py        - actionable what-if search
│   ├── cost_analysis.py          - euro-denominated P&L sweep
│   └── reason_codes.py           - GDPR Art. 22 adverse-action text
├── streamlit_app.py              - the application
└── tests/                        - 34 pytest tests
```

---

## Setup

```bash
pip install -r requirements.txt
```

The model artefacts in `models/` are committed, so the Streamlit app runs
out of the box:

```bash
streamlit run streamlit_app.py
```

To re-train end-to-end:

```bash
# 1. Place the raw CSV at data/raw/uci_credit_card.csv
#    (mirrored on multiple public GitHub repos as UCI_Credit_Card.csv)
python -m src.data_prep
python -m src.train
```

To run the test suite (and the linter), install the dev dependencies first:

```bash
pip install -r requirements-dev.txt
ruff check src tests streamlit_app.py
pytest tests/ -v
```

---

## Data

| Field          | Value                                                            |
|----------------|------------------------------------------------------------------|
| Source         | UCI ML Repository, *Default of Credit Card Clients Data Set*    |
| Period         | April–September 2005, Taiwan                                     |
| Rows           | 30,000                                                           |
| Raw features   | 23 explanatory + 1 binary target                                 |
| Final features | 34 after engineering and one-hot encoding                        |
| Default rate   | 22.12%                                                           |
| Split          | Stratified 60 / 20 / 20 train / val / test (random, seed = 42)   |

The original project brief targeted the Lending Club 2015–2018 dataset,
which Kaggle removed. The UCI Taiwan dataset is the standard freely
available substitute in credit-risk literature.

### Feature engineering

```
PAY_MEAN, PAY_MAX            - average / worst delinquency status across 6 months
DELINQ_COUNT                 - count of months with PAY_x > 0
BILL_MEAN, PAY_AMT_MEAN      - 6-month averages
UTILIZATION                  - BILL_AMT1 / LIMIT_BAL
PAYMENT_RATIO                - PAY_AMT1 / max(BILL_AMT1, 1)
LIMIT_PER_AGE                - LIMIT_BAL / AGE
SEX_MALE, EDUCATION_*, MARRIAGE_*  - one-hot encodings (drop_first=True)
```

---

## Explainability

The application uses `shap.TreeExplainer` with
`feature_perturbation="tree_path_dependent"`. Under this mode, SHAP values
satisfy the **local-accuracy** property exactly:

> `sum(shap_values) + base_value == logit(P(default))`

This invariant is asserted in `tests/test_explain.py::test_shap_local_accuracy`
to within `1e-3` tolerance.

The Methodology tab inside the app explains the Shapley-value formula and
the tree-path-dependent algorithm in detail.

---

## Tests

```
tests/test_data_prep.py        - cleaning, feature engineering, leakage, split
tests/test_predict.py          - rating buckets, probability bounds, alignment
tests/test_explain.py          - SHAP shape, local accuracy, waterfall, global imp.
tests/test_calibration.py      - reliability table, Brier decomposition, isotonic
tests/test_fairness.py         - per-group metrics, disparate impact, EOD
tests/test_cost_analysis.py    - confusion matrix, P&L sweep, optimal threshold
tests/test_reason_codes.py     - adverse-action notice generation
tests/test_counterfactuals.py  - counterfactual search on the trained model
```

`pytest tests/ -v` → **34 passed**.

---

## Governance pack

The `docs/` folder contains the artefacts a model risk committee, an
internal audit team, or a regulator would request alongside the model
itself.

| Artefact | What it answers |
|---|---|
| [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) | Why fund this, € loss reduction, payback, sensitivities |
| [`docs/PERSONAS_AND_USER_STORIES.md`](docs/PERSONAS_AND_USER_STORIES.md) | Who uses it, with explicit acceptance criteria |
| [`docs/RACI.md`](docs/RACI.md) | Who is Accountable / Responsible / Consulted / Informed |
| [`docs/PROCESS_FLOW.md`](docs/PROCESS_FLOW.md) | Where the model sits in the origination workflow |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phased delivery MVP → v1 → v2 |
| [`docs/REGULATORY_MAPPING.md`](docs/REGULATORY_MAPPING.md) | Basel III / EBA / ECB / SR 11-7 / GDPR Art. 22 / EU AI Act |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Intended use, training data, performance, limitations |
| [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md) | Top 20 model risks scored × mitigated |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Every feature: definition, source, owner, PII flag |
| [`docs/MONITORING_PLAN.md`](docs/MONITORING_PLAN.md) | PSI / CSI / calibration drift / fairness cadence |
| [`docs/AB_TEST_DESIGN.md`](docs/AB_TEST_DESIGN.md) | Pre-registered champion-challenger A/B test |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Plain-language definitions for non-experts |

The **Governance** tab in the Streamlit app surfaces the business case
summary, an interactive reliability diagram (the model sits on the diagonal;
an isotonic overlay is available as a diagnostic), the fairness audit on the
public sample, the cost-sensitive thresholding panel, a production-monitoring
dashboard mock-up, and the full list of governance documents.

---

## Author

**Tristan Mas**, Business Analyst, Risk & Finance IT
[GitHub](https://github.com/MasTristan) · [LinkedIn](https://linkedin.com/in/tristan-mas)

---

## License

MIT, see [LICENSE](LICENSE).
