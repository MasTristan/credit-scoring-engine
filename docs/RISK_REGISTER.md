# Model risk register

> Top model risks identified before go-live, scored on a standard
> 1-to-5 likelihood × impact matrix, with mitigations and an owner.
> Reviewed quarterly by the Model Risk Committee.

---

## Risk scoring grid

```
   Impact ▲
        5│  H   H   C   C   C
        4│  M   H   H   C   C
        3│  L   M   H   H   C
        2│  L   L   M   H   H
        1│  L   L   L   M   M
         └──────────────────────► Likelihood
            1   2   3   4   5

   L = Low (accept), M = Medium (monitor),
   H = High (mitigate), C = Critical (mitigate or stop)
```

---

## Risk register

| #  | Risk                                                                                            | L | I | Score | Owner          | Mitigation                                                                                                                              |
|----|-------------------------------------------------------------------------------------------------|---|---|-------|----------------|------------------------------------------------------------------------------------------------------------------------------------------|
| R-01 | **Data drift** — input distribution shifts away from training distribution                       | 4 | 4 | C     | Data Eng        | Daily PSI on all 34 features, alert > 0.10, retrain > 0.25 (`docs/MONITORING_PLAN.md`)                                                  |
| R-02 | **Concept drift** — relationship between features and default changes (macro break, policy change) | 3 | 5 | C     | Data Science    | Weekly performance monitoring on a one-month rolling window; champion-challenger framework active (`docs/AB_TEST_DESIGN.md`)             |
| R-03 | **Training-serving skew** — feature engineering differs between training and inference            | 2 | 5 | H     | Data Science    | Single `build_feature_matrix` function shared between training and the Streamlit form; verified by test `test_align_features_ordering`    |
| R-04 | **Selection bias / no reject inference** — model under-predicts on rejected-applicant tail        | 5 | 3 | H     | Data Science    | Flagged in `docs/MODEL_CARD.md` Limitations; reject inference in roadmap Phase 1                                                          |
| R-05 | **Disparate impact on protected attribute (sex)** — DI ratio falls outside [0.8, 1.25]            | 3 | 5 | C     | Compliance      | Pre-deployment audit (`src/fairness.py`); daily monitor on prod scores; pre-deployment removal of `SEX_MALE` in EU production            |
| R-06 | **Calibration drift** — predicted PD diverges from observed default frequency                     | 4 | 3 | H     | Risk Mgmt       | Monthly reliability diagram review; isotonic recalibration available in `src/calibration.py`; recalibrate when Brier > 0.20               |
| R-07 | **Over-fit to PAY_0 and PAY_MAX** — model collapses to a delinquency-window heuristic              | 3 | 3 | H     | Data Science    | Feature-importance audit per release; ablation tests on hold-out (planned)                                                                 |
| R-08 | **SHAP attribution incorrect / inconsistent**                                                     | 1 | 5 | H     | Data Science    | Local-accuracy invariant verified by `tests/test_explain.py::test_shap_local_accuracy`                                                    |
| R-09 | **Model service unavailable**                                                                     | 3 | 4 | H     | IT Production   | Fallback to legacy scorecard in decisioning engine; on-call playbook; 99.9% target                                                         |
| R-10 | **GDPR Art. 22 complaint** — applicant disputes an automated decision                             | 3 | 4 | H     | Compliance      | Refer-to-human path active (`docs/PROCESS_FLOW.md`); reason codes + counterfactuals available; logging of inputs and SHAP per decision    |
| R-11 | **Adversarial input** — applicant crafts inputs to game the score                                 | 2 | 3 | M     | Compliance      | Schema validation (range checks) on every input; out-of-range → manual review                                                              |
| R-12 | **Loss of model artefact** (git history rewrite, registry corruption)                              | 1 | 4 | M     | IT Production   | Artefacts committed to git; immutable storage in production model registry                                                                 |
| R-13 | **Bureau-data outage** propagates through pipeline                                                  | 3 | 3 | H     | Data Eng        | Last-known-good fallback up to 7 days; manual review beyond                                                                                 |
| R-14 | **Hyperparameter / training non-reproducibility**                                                  | 1 | 3 | L     | Data Science    | All seeds fixed; deps pinned; reproducibility verified in CI                                                                                |
| R-15 | **Stale training data** (e.g. label leakage from forward-looking features added by mistake)        | 2 | 5 | H     | Data Science    | No post-origination features in feature set; verified by `tests/test_data_prep.py::test_no_leakage_columns_present`                          |
| R-16 | **EU AI Act non-compliance** — high-risk system without conformity assessment                      | 4 | 5 | C     | Compliance      | Conformity assessment in Phase 1 roadmap; technical documentation already in `docs/`                                                        |
| R-17 | **Dependency vulnerability** (XGBoost / SHAP CVE)                                                  | 3 | 2 | M     | IT Production   | Pinned deps with monthly Dependabot review; security advisory subscription                                                                  |
| R-18 | **Key-person dependency** (only one person can retrain)                                            | 3 | 3 | H     | PO              | Documentation in this repo; pair-programming on every release                                                                                |
| R-19 | **Model staleness during a credit cycle inflection** (e.g. recession start)                         | 3 | 5 | C     | Risk Mgmt       | Macro-aware monitoring; automatic shrinking of approval rate on PSI alert (policy lever, not model change)                                   |
| R-20 | **Public communication of model output** (e.g. credit score sent to wrong applicant)                | 1 | 5 | H     | IT Production   | Encryption in transit & at rest; auditable access logs                                                                                       |

---

## Top risks summary

The four **Critical (C)** risks at go-live are:

1. R-01 — data drift
2. R-02 — concept drift
3. R-05 — disparate impact on protected attribute
4. R-16 — EU AI Act conformity
5. R-19 — model staleness through a macro inflection

All five have a mitigation path active or planned in Phase 1 of the
roadmap. Three open critical risks remain at MVP stage (R-01, R-02, R-16)
because they are addressed only by *production* deployment activities.

---

## Acceptance criteria for go-live

The Model Risk Committee will sign off only if:

- All **Critical** risks have a mitigation in place and a tested
  fallback.
- All **High** risks have a documented owner and a target date for
  closure.
- Fairness audit passes (`DI ∈ [0.8, 1.25]` on `sex`, AUC delta < 0.05
  across age bands).
- Independent validation report signed.
- A/B test ramp plan agreed.
