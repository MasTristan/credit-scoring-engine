# CLAUDE.md — ML Credit Scoring Engine with Explainability

Project brief for Claude Code. Adapted from the original Lending Club brief to
use the **UCI Default of Credit Card Clients** dataset (Taiwan, 2005), which
is freely downloadable and well-known in credit risk literature.

---

## Project context

XGBoost credit scoring model trained on the UCI "Default of Credit Card
Clients" dataset, with full explainability via SHAP. Public Streamlit
application with two modes: scoring an individual contract (SHAP waterfall)
and analysing a portfolio (PD distribution, risk segmentation, global
feature importance).

This project closes a three-project portfolio:
- Project 1: regulatory PL/SQL engine (Oracle, Basel III)
- Project 2: BI dashboard on public data (Power BI, EBA)
- Project 3: explainable ML in production (XGBoost, SHAP, Streamlit)

**Hard constraint: zero paid license.**
Stack: Python 3.10+, Streamlit Community Cloud (free), scikit-learn, XGBoost,
SHAP, pandas, plotly. Public GitHub.

---

## Dataset — UCI Default of Credit Card Clients

### Source

UCI ML Repository: "Default of Credit Card Clients Data Set" (Taiwan,
Apr–Sep 2005). 30,000 records, 23 explanatory features + binary target.
Originally published by Yeh & Lien (2009).

Mirrored on multiple public GitHub repos as `UCI_Credit_Card.csv`.

### Target variable

`default.payment.next.month` → `IS_DEFAULT` (1 = default, 0 = no default).
Default rate in the raw data: ~22.1%.

### Features (raw)

```
LIMIT_BAL          : credit limit (NT$)
SEX                : 1 = male, 2 = female
EDUCATION          : 1 = grad school, 2 = university, 3 = high school,
                     4 = others, (5, 6 unknown → mapped to 4)
MARRIAGE           : 1 = married, 2 = single, 3 = others (0 → 3)
AGE                : years
PAY_0..PAY_6       : repayment status for the last 6 months
                     -2 = no consumption, -1 = paid in full,
                      0 = revolving credit, 1..9 = months past due
BILL_AMT1..BILL_AMT6 : bill amounts (NT$) for the last 6 months
PAY_AMT1..PAY_AMT6   : payments made (NT$) for the last 6 months
```

### Feature engineering

```
PAY_MEAN, PAY_MAX            : average / worst delinquency status across 6m
DELINQ_COUNT                 : count of months with PAY_x > 0
BILL_MEAN, PAY_AMT_MEAN      : 6-month averages
UTILIZATION                  : BILL_AMT1 / LIMIT_BAL
PAYMENT_RATIO                : PAY_AMT1 / max(BILL_AMT1, 1)
LIMIT_PER_AGE                : LIMIT_BAL / AGE
EDUCATION_*, MARRIAGE_*, SEX : one-hot (drop_first=True)
```

Note: this dataset has no temporal information beyond the 6-month payment
window, so we use a stratified 80/20 random split (not a temporal one as in
the original brief).

---

## Architecture

```
LendingClubCreditScoring/
├── CLAUDE.md                        <- this file
├── README.md                        <- public documentation (EN)
├── ARCHITECTURE.md                  <- technical documentation
├── LICENSE                          <- MIT
├── .gitignore
├── requirements.txt                 <- pinned dependencies
├── data/
│   ├── raw/                         <- not committed
│   │   └── uci_credit_card.csv
│   ├── processed/                   <- not committed
│   │   ├── train.parquet
│   │   ├── test.parquet
│   │   └── feature_names.json
│   └── sample/                      <- committed
│       └── sample_1000.csv
├── models/
│   ├── xgboost_model.json
│   ├── feature_importance.csv
│   ├── shap_background.parquet
│   └── training_metrics.json
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── train.py
│   ├── predict.py
│   └── explain.py
├── streamlit_app.py
└── tests/
    ├── test_data_prep.py
    ├── test_predict.py
    └── test_explain.py
```

---

## Pipeline

### Data prep (`src/data_prep.py`)

1. Load `data/raw/uci_credit_card.csv`
2. Rename target → `IS_DEFAULT`
3. Drop `ID` column
4. Clean `EDUCATION` (map 0, 5, 6 → 4 "others") and `MARRIAGE` (0 → 3)
5. Compute engineered features
6. One-hot encode `SEX`, `EDUCATION`, `MARRIAGE` (drop_first=True)
7. Stratified 80/20 split on `IS_DEFAULT` (random_state=42)
8. Save `train.parquet`, `test.parquet`, `feature_names.json`
9. Build `data/sample/sample_1000.csv` from the test set (raw schema +
   `IS_DEFAULT_TRUE`)

### Training (`src/train.py`)

- XGBoost classifier with `scale_pos_weight = N_neg / N_pos`
- Early stopping on validation AUC
- Metrics: ROC-AUC, PR-AUC, Gini, KS, Brier, log-loss, F1, precision, recall
  at the Youden-J optimal threshold
- Outputs:
  - `models/xgboost_model.json`
  - `models/feature_importance.csv` (gain, weight, cover)
  - `models/shap_background.parquet` (500 rows stratified)
  - `models/training_metrics.json`

### Inference (`src/predict.py`)

- `load_model`, `predict_proba`
- `score_to_rating(pd)` → AAA/AA, A, BBB, BB, B, CCC, D (Project 1 table)
- `score_to_risk_band(pd)` → LOW < 0.05, MEDIUM < 0.15, HIGH ≥ 0.15

### Explainability (`src/explain.py`)

- `shap.TreeExplainer` with the background dataset
- `get_waterfall_data` returns top-N features (signed SHAP value)
- `get_global_importance` returns mean(|SHAP|) per feature

---

## Streamlit app — 4 tabs

1. **Individual Scorer** — form with sliders for the main features,
   PD + rating + risk band metric row, SHAP waterfall (top 15 features).
2. **Portfolio Analysis** — load `data/sample/sample_1000.csv`, score the
   batch, show KPI row, PD histogram (coloured by risk band), scatter
   PD vs `LIMIT_BAL` sized by bill amount, breakdown by repayment status,
   global SHAP importance on a 200-row subset.
3. **Model Performance** — KPI row from `training_metrics.json`, ROC curve,
   PR curve, score distribution by actual outcome (KS plot), full metrics
   table.
4. **About this model** — algorithm, training/test split, why SHAP, model
   limitations, training details (raw JSON).

---

## Validation checklist

- [x] ROC-AUC ≥ 0.72 on test set (typical for this dataset: ~0.78)
- [x] No post-origination leakage (target column dropped before training)
- [x] SHAP local accuracy test (`sum(shap_values) + base_value ≈ logit`)
- [x] Streamlit app starts locally without errors
- [x] Tab 1 (Individual Scorer) renders waterfall chart
- [x] Tab 2 (Portfolio) renders all four panels on `sample_1000.csv`
- [x] Tab 3 (Model Performance) renders ROC + PR + KS plot
- [x] Public artefacts committed: `models/*`, `data/sample/sample_1000.csv`
- [x] README contains live-demo link and screenshots

---

## Run locally

```bash
pip install -r requirements.txt

# 1. Place raw CSV at data/raw/uci_credit_card.csv
python -m src.data_prep
python -m src.train
streamlit run streamlit_app.py

# 2. Tests
pytest tests/
```
