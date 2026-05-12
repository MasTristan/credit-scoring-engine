# Regulatory mapping

> A credit-scoring model in the EU sits at the intersection of prudential
> regulation (Basel III / CRR), supervisory guidance on internal models
> (EBA, ECB, local NCA), model-risk management (Federal Reserve SR 11-7
> for cross-border banks), consumer-protection law (GDPR Art. 22, Consumer
> Credit Directive), and the forthcoming EU AI Act. This document maps
> each requirement to **the specific control implemented in this repo**.

---

## 1. Prudential — Basel III / CRR

### Basel III §§ 144–162 — IRB framework (PD estimation)

| Requirement                                           | Implementation in this repo                                      |
|-------------------------------------------------------|-------------------------------------------------------------------|
| One-year horizon for PD                               | Target = `default.payment.next.month` (effectively MOB+1)         |
| Min. 5-year historical data window                    | **Gap** — UCI data is 6 months; explicitly called out in `docs/MODEL_CARD.md` Limitations |
| Distinct PD grades, properly differentiated           | 7-bucket rating scale `AAA/AA … D` in `src/predict.py::score_to_rating` |
| Consistency of definitions over time                   | Target definition documented in `docs/DATA_DICTIONARY.md`         |
| Model validation independent of model development     | Validation activities described in `docs/RACI.md` (2nd line)      |

### CRR Article 174 — model validation

| Requirement                                           | Implementation                                                    |
|-------------------------------------------------------|-------------------------------------------------------------------|
| Quantitative validation (discrimination, calibration) | ROC-AUC, Gini, KS, Brier, log-loss in `training_metrics.json`     |
| Stability of grades / migration matrices              | Designed in `docs/MONITORING_PLAN.md`; PSI / CSI monitors          |
| Use of model in business decisions documented         | `docs/PROCESS_FLOW.md` step 4                                     |

---

## 2. Supervisory — EBA, ECB, local NCA

### EBA/GL/2017/16 — Guidelines on PD/LGD estimation

| Requirement                                                  | Implementation                                                    |
|--------------------------------------------------------------|-------------------------------------------------------------------|
| §§ 60–65 — selection of risk drivers                           | Engineered features documented in `docs/DATA_DICTIONARY.md`, justified in `docs/MODEL_CARD.md` |
| §§ 71–75 — observation period / margin of conservatism         | **Gap** — short observation period flagged in MODEL_CARD          |
| §§ 80–85 — treatment of data, default definition               | Default = `IS_DEFAULT` (90+ DPD equivalent for this dataset)      |
| §§ 105–115 — calibration                                       | Reliability diagram + isotonic recalibration in `src/calibration.py` |
| §§ 130–137 — model overrides / human-in-the-loop                | Manual review path in `docs/PROCESS_FLOW.md`                       |

### ECB Targeted Review of Internal Models (TRIM, 2017–2019 + ongoing)

| Theme                              | Implementation                                                    |
|------------------------------------|-------------------------------------------------------------------|
| Reproducibility                    | All seeds fixed; `python -m src.data_prep && python -m src.train` reproduces metrics |
| Documentation completeness          | This `docs/` folder + `ARCHITECTURE.md` + inline docstrings        |
| Independent challenger model        | Designed in `docs/AB_TEST_DESIGN.md`                              |

### ECB Guide on internal models (2024 update) — section on ML

> ECB has explicitly accepted gradient-boosted trees provided the
> bank can demonstrate (a) interpretability per decision, (b) stability
> over time, (c) the absence of disparate impact.

| Requirement              | Implementation                                                                 |
|--------------------------|---------------------------------------------------------------------------------|
| Interpretability          | SHAP `TreeExplainer` per decision (Tab 1); local accuracy verified in tests    |
| Stability                 | PSI / CSI / calibration drift monitors in `docs/MONITORING_PLAN.md`            |
| Disparate impact          | Fairness audit in `src/fairness.py` (and Governance tab in the app)            |

---

## 3. Cross-border model risk — Fed SR 11-7

| Requirement                                  | Implementation                                                        |
|----------------------------------------------|------------------------------------------------------------------------|
| Sound development, implementation, use       | Test suite (`tests/`), data prep + train scripts reproducible           |
| Effective challenge — independent validation | 2nd-line activities in `docs/RACI.md`; designed handover artefacts     |
| Ongoing monitoring                           | `docs/MONITORING_PLAN.md` — daily PSI/CSI, weekly performance, monthly recalibration check |
| Model inventory                              | `docs/MODEL_CARD.md` is the registry entry for this model               |
| Documentation supports independent review     | `ARCHITECTURE.md` + `docs/` + reproducible artefacts                    |

---

## 4. Consumer protection

### GDPR Article 22 — automated individual decision-making

> "The data subject shall have the right not to be subject to a decision
> based solely on automated processing … which produces legal effects
> concerning him or her or similarly significantly affects him or her."
> With derogations, the data subject still has the right to:
> (i) obtain human intervention, (ii) express his/her point of view,
> and (iii) **contest the decision**.

| Requirement                                  | Implementation                                                  |
|----------------------------------------------|------------------------------------------------------------------|
| Right to human intervention                  | "Refer to human" path in `docs/PROCESS_FLOW.md` step 5            |
| Right to explanation                         | SHAP waterfall (Tab 1) + plain-language reason codes (Governance tab) |
| Logging / auditability                       | Designed: each score logged with model version hash, top-15 SHAP, timestamp |
| Special-category data prohibition            | No religion / political opinion / health / biometric in the feature set |

### EU AI Act (Regulation 2024/1689) — high-risk system

Credit scoring is classified high-risk under Annex III §5(b). Obligations:

| Requirement                                                | Implementation                                                  |
|------------------------------------------------------------|------------------------------------------------------------------|
| Risk management system (Art. 9)                            | `docs/RISK_REGISTER.md`                                          |
| Data and data governance (Art. 10)                         | `docs/DATA_DICTIONARY.md`; representativeness flagged in `docs/MODEL_CARD.md` |
| Technical documentation (Art. 11 + Annex IV)               | This whole `docs/` folder + `ARCHITECTURE.md` + `MODEL_CARD.md`  |
| Record-keeping (Art. 12)                                   | Designed: structured logging of scores, inputs, SHAP, decisions  |
| Transparency to deployers (Art. 13)                        | `docs/MODEL_CARD.md`                                             |
| Human oversight (Art. 14)                                  | Refer-to-human path; override authority in `docs/RACI.md`        |
| Accuracy, robustness, cybersecurity (Art. 15)              | Test suite, drift monitoring, model artefacts under access control |
| Fundamental rights impact assessment (Art. 27, deployers)  | Fairness audit in `src/fairness.py`                              |

### Consumer Credit Directive (2008/48/EC, recast 2023/2225)

| Requirement                                              | Implementation                                                |
|----------------------------------------------------------|----------------------------------------------------------------|
| Creditworthiness assessment (Art. 18 of recast)          | The model is exactly this                                      |
| Information to consumers on rejection                    | Adverse-action notice with top-3 reason codes (Tab 1 + Governance tab) |
| Database consultation must be relevant and proportionate | Bureau pull justified in `docs/PROCESS_FLOW.md`                |

---

## 5. Documentation crosswalk

For each external requester, the right combination of artefacts:

| Audience                        | Read in this order                                                          |
|---------------------------------|------------------------------------------------------------------------------|
| ECB / local NCA (model review)  | `MODEL_CARD.md` → `ARCHITECTURE.md` → `MONITORING_PLAN.md` → metrics + tests |
| Internal Audit                  | `RACI.md` → this file → `RISK_REGISTER.md` → sample SHAP outputs              |
| Model Validation (2nd line)     | `MODEL_CARD.md` → `src/train.py` → `tests/` → fairness/calibration outputs    |
| Consumer / data subject (GDPR)  | SHAP waterfall + reason-codes block from the Governance tab                   |
| Hiring manager / recruiter (P4) | `README.md` → Methodology tab → this folder                                    |
