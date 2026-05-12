# Process flow — where the model fits

> The model is **one step** in a longer credit-origination process. This
> diagram shows the upstream and downstream activities, the data it
> consumes, and the systems it integrates with. Without this picture the
> model is a notebook; with it, it's a system.

---

## End-to-end origination (BPMN-style)

```
   ┌──────────────┐
   │  Applicant   │ submits application via web / branch / phone
   └──────┬───────┘
          │
          ▼
   ┌─────────────────────────┐
   │ 1. Capture & validate   │  client-facing channel + ID document check
   │    KYC / AML            │
   └──────┬──────────────────┘
          │
          ▼
   ┌─────────────────────────┐
   │ 2. Credit-bureau pull   │  external bureau API (e.g. Experian, Banque
   │                         │  de France)
   └──────┬──────────────────┘
          │
          ▼
   ┌─────────────────────────┐
   │ 3. Build feature vector │  data engineering — joins KYC, bureau, and
   │    (this codebase)      │  internal product history into one row
   └──────┬──────────────────┘
          │
          ▼
   ┌─────────────────────────┐
   │ 4. Score the contract   │  ◀────── XGBoost model + SHAP
   │    PD, rating, band     │           (src/predict.py, src/explain.py)
   └──────┬──────────────────┘
          │
          ▼
   ┌─────────────────────────┐
   │ 5. Decisioning engine   │  combines PD, policy rules, regulatory
   │                         │  caps, fraud signals → ACCEPT / REJECT /
   │                         │  REFER-TO-HUMAN
   └──────┬──────────────────┘
          │
          ▼
        ╱─────────────╲
       │  decision?    │
        ╲──────┬──────╱
       │       │       │
       ▼       ▼       ▼
   ┌────────┐ ┌────────────┐ ┌────────────┐
   │ Accept │ │   Refer    │ │   Reject   │
   └───┬────┘ └─────┬──────┘ └─────┬──────┘
       │            │              │
       ▼            ▼              ▼
   ┌────────┐ ┌────────────┐ ┌──────────────────┐
   │ Booking│ │  Manual    │ │ Adverse-action   │
   │ system │ │  review    │ │ notice (GDPR     │
   │        │ │  by P1     │ │ Art. 22)         │
   └───┬────┘ └─────┬──────┘ │ — top 3 SHAP     │
       │            │        │   reason codes   │
       │            │        └──────────────────┘
       │            │
       ▼            ▼
   ┌──────────────────────────────────────────┐
   │ Loan management & collections system      │
   │ (behaviour data feeds back to feature     │
   │  store for the next training cycle)       │
   └──────────────────────────────────────────┘
```

---

## Data flow into the feature vector

```
KYC system ──────────► AGE, EDUCATION, MARRIAGE, SEX
                            │
Bureau API ─────────► PAY_0..PAY_6 (delinquency over the last 6 cycles)
                            │
Internal billing  ──► BILL_AMT1..BILL_AMT6, PAY_AMT1..PAY_AMT6,
                       LIMIT_BAL
                            │
                            ▼
                  src/data_prep.build_feature_matrix
                            │
                            ▼
                    34-dimensional feature row
                            │
                            ▼
                    XGBoost model + SHAP
                            │
                            ▼
                   PD, rating, top-15 SHAP, reason codes
```

The **single source of truth** for feature engineering is
`src/data_prep.build_feature_matrix` — the **same function** is used at
training time, by the bundled 1,000-row sample, and by the single-row
form in Tab 1. This is the architectural guarantee against
training-serving skew.

---

## SLAs by step

| Step | SLA (95th percentile) | Owner       |
|------|------------------------|-------------|
| 1. KYC / AML        | 30 s        | KYC vendor  |
| 2. Bureau pull       | 800 ms     | Bureau API  |
| 3. Feature assembly  | 50 ms      | Data Eng    |
| 4. Score + SHAP      | **80 ms**   | This system |
| 5. Decisioning       | 10 ms      | Risk Eng    |
| Total (auto path)    | < 2 s      |             |
| Manual review path   | < 24 h     | P1 (Marie)  |
| Adverse-action mail  | < 5 days   | Compliance  |

---

## Failure modes & fallbacks

| Failure                              | Mitigation                              |
|--------------------------------------|------------------------------------------|
| Bureau API down                       | Use last-known-good bureau response if < 7 days old, else refer to human |
| Model service down                    | Decisioning engine falls back to policy rules (manual scorecard) |
| Feature out of domain (e.g. AGE < 18) | Schema validation in step 3, REFER decision |
| PD ∈ {NaN, ∞}                         | REFER decision, alert raised             |
| PSI breach on any feature (> 0.25)    | Score still returned, drift alert raised |
