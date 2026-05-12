# Data dictionary & lineage

> Every feature consumed or produced by the model, with its definition,
> source system, type, refresh cadence, owner, and a PII flag. The
> "lineage" section traces each output back to its raw input.

---

## Raw features (input to `src/data_prep.build_feature_matrix`)

| Name          | Type    | Definition                                                            | Source system     | Refresh   | Owner   | PII | Protected? |
|---------------|---------|-----------------------------------------------------------------------|--------------------|-----------|---------|-----|------------|
| `LIMIT_BAL`   | int     | Credit limit in NT$                                                    | Card system        | Monthly   | DataEng | No  | No         |
| `SEX`         | cat     | 1 = male, 2 = female                                                   | KYC                | Static    | DataEng | Yes | **Yes**    |
| `EDUCATION`   | cat     | 1 = graduate, 2 = university, 3 = high school, 4 = other (0/5/6 → 4)   | KYC                | Annual    | DataEng | Yes | Maybe      |
| `MARRIAGE`    | cat     | 1 = married, 2 = single, 3 = other (0 → 3)                              | KYC                | Annual    | DataEng | Yes | No         |
| `AGE`         | int     | Age in years (derived from date of birth)                              | KYC                | Annual    | DataEng | Yes | Possibly   |
| `PAY_0`       | int     | Repayment status — current month                                       | Billing system     | Monthly   | DataEng | No  | No         |
| `PAY_2`..`PAY_6` | int  | Repayment status — months t-1 to t-5                                   | Billing system     | Monthly   | DataEng | No  | No         |
| `BILL_AMT1`   | int     | Bill amount — current month (NT$)                                       | Billing system     | Monthly   | DataEng | No  | No         |
| `BILL_AMT2..6`| int     | Bill amount — months t-1 to t-5 (NT$)                                    | Billing system     | Monthly   | DataEng | No  | No         |
| `PAY_AMT1`    | int     | Payment made — current month (NT$)                                      | Billing system     | Monthly   | DataEng | No  | No         |
| `PAY_AMT2..6` | int     | Payment made — months t-1 to t-5 (NT$)                                  | Billing system     | Monthly   | DataEng | No  | No         |

### `PAY_*` code book

| Code | Meaning                                |
|------|----------------------------------------|
| -2   | No consumption that month               |
| -1   | Paid in full                            |
|  0   | Use of revolving credit (no delinquency) |
|  1   | 1 month past due                        |
|  2..8| 2 to 8 months past due                  |

---

## Engineered features (output of `engineer_features`)

| Name             | Formula                                                                  | Why                                                                |
|------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------|
| `PAY_MEAN`        | mean of `PAY_0..PAY_6`                                                    | Average delinquency status — smooths month-to-month noise            |
| `PAY_MAX`         | max of `PAY_0..PAY_6`                                                     | Worst delinquency over the window — strong default predictor          |
| `DELINQ_COUNT`    | count of `PAY_x > 0`                                                       | Persistence of delinquency                                            |
| `BILL_MEAN`       | mean of `BILL_AMT1..6`                                                    | Average usage of revolving credit                                     |
| `PAY_AMT_MEAN`    | mean of `PAY_AMT1..6`                                                     | Average ability/willingness to pay                                    |
| `UTILIZATION`     | `BILL_AMT1 / LIMIT_BAL` (clipped to [-2, 5])                              | Credit-line stress                                                    |
| `PAYMENT_RATIO`   | `PAY_AMT1 / max(BILL_AMT1, 1)` (clipped to [0, 10])                       | Coverage ratio of the most recent payment                              |
| `LIMIT_PER_AGE`   | `LIMIT_BAL / max(AGE, 18)`                                                | Proxy for granted-limit-vs-age relationship                            |

---

## Encoded features (output of `encode_categoricals`)

| Name            | Source                       |
|-----------------|------------------------------|
| `SEX_MALE`       | `(SEX == 1).astype(int)`      |
| `EDUCATION_2..4` | one-hot from `EDUCATION`, drop_first=True |
| `MARRIAGE_2..3`  | one-hot from `MARRIAGE`, drop_first=True   |

Final feature vector has **34 columns** — see `models/feature_names.json`.

---

## Target variable

| Field         | Value                                                            |
|---------------|------------------------------------------------------------------|
| Name          | `IS_DEFAULT`                                                      |
| Type          | binary {0, 1}                                                    |
| Definition    | Did the client default on the next month's payment?               |
| Source column | `default.payment.next.month` (in the raw CSV)                     |
| Class balance | 22.1% positive (default)                                          |
| Horizon       | 1 month (point-in-time)                                           |

---

## Lineage

```
   ┌────────────────────────────────────┐
   │  data/raw/uci_credit_card.csv      │  (not committed — open dataset)
   └─────────────────┬──────────────────┘
                     │
                     ▼   load_raw + clean_categoricals
   ┌────────────────────────────────────┐
   │  cleaned raw frame                 │
   │  (renamed target, EDUCATION/MARR.) │
   └─────────────────┬──────────────────┘
                     │
                     ▼   engineer_features
   ┌────────────────────────────────────┐
   │  + 8 engineered numeric features    │
   └─────────────────┬──────────────────┘
                     │
                     ▼   encode_categoricals
   ┌────────────────────────────────────┐
   │  35 columns (34 features + target)  │
   └─────────────────┬──────────────────┘
                     │
                     ▼   train_test_split (stratified 80/20)
   ┌──────────────────┬──────────────────┐
   │  train.parquet   │  test.parquet    │
   │  (24,000 rows)   │  (6,000 rows)    │
   └──────────────────┴──────────────────┘
                     │
                     ▼   XGBClassifier.fit
   ┌────────────────────────────────────┐
   │  models/xgboost_model.json          │
   │  models/training_metrics.json       │
   │  models/feature_importance.csv      │
   │  models/shap_background.parquet     │
   │  models/feature_names.json          │
   └────────────────────────────────────┘
                     │
                     ▼   (downstream — Streamlit app)
   ┌────────────────────────────────────┐
   │  PD, RATING, RISK_BAND, SHAP         │
   └────────────────────────────────────┘
```

---

## Data quality controls

| Control                                                  | Tested in                                                  |
|----------------------------------------------------------|-------------------------------------------------------------|
| `EDUCATION` ∈ {1, 2, 3, 4} after cleaning                | `tests/test_data_prep.py::test_clean_categoricals_education` |
| `MARRIAGE` ∈ {1, 2, 3} after cleaning                    | `tests/test_data_prep.py::test_clean_categoricals_marriage`  |
| `SEX` ∈ {1, 2} after cleaning                            | `tests/test_data_prep.py::test_clean_categoricals_sex`       |
| No nulls in feature matrix                               | `tests/test_data_prep.py::test_no_null_features`             |
| `DELINQ_COUNT` ∈ [0, 6]                                  | `tests/test_data_prep.py::test_engineer_features_ranges`      |
| Stratified split preserves default rate within 2 pp       | `tests/test_data_prep.py::test_stratified_split_rates`        |
| Feature ordering at inference matches training            | `tests/test_predict.py::test_align_features_ordering`          |
| SHAP local accuracy within 1e-3                          | `tests/test_explain.py::test_shap_local_accuracy`             |

---

## Refresh & ownership

| Dataset                                          | Refresh cadence | Owner   | Storage location           |
|--------------------------------------------------|-----------------|---------|----------------------------|
| `data/raw/uci_credit_card.csv`                    | Static (public) | DataEng | Not committed; .gitignored   |
| `data/processed/{train,test}.parquet`             | Per training    | DataEng | Not committed; .gitignored   |
| `data/sample/sample_1000.csv`                     | Per training    | DataEng | Committed (public demo)      |
| `models/*`                                       | Per training    | DataSci | Committed (public artefacts) |
| Production inference logs (designed)              | Real-time       | DataEng | Bank's data lake             |
