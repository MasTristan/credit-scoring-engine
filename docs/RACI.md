# Stakeholder map & RACI

> Who is **R**esponsible, **A**ccountable, **C**onsulted, **I**nformed
> for each life-cycle activity of the model. Aligned with the three-lines-
> of-defence framework used in EU/UK regulated banks.

---

## Stakeholders

| Role / function                           | Line of defence | Notes                                                 |
|-------------------------------------------|-----------------|-------------------------------------------------------|
| Business owner (Head of Retail Credit)    | 1st             | Sponsor; owns the P&L impact                          |
| Product Owner (this profile)              | 1st             | Translates business need to backlog and acceptance    |
| Data Scientist / ML Engineer              | 1st             | Builds, trains, and serves the model                  |
| Data Engineer                             | 1st             | Owns the feature store and inference pipeline         |
| IT Production                             | 1st             | Hosts the inference service, on-call                  |
| Model Validation (independent review)     | 2nd             | Independent challenge under SR 11-7                   |
| Risk Management / CRO office              | 2nd             | Risk appetite, limits, calibration sign-off           |
| Compliance / Legal                        | 2nd             | GDPR Art. 22, ECB / EBA / local regulator submissions |
| Internal Audit                            | 3rd             | Periodic audit, sample-based testing of decisions     |
| External Auditor / Regulator              | external        | On request                                            |

---

## RACI by life-cycle activity

Legend: **R**esponsible (does the work), **A**ccountable (one per row),
**C**onsulted (provides input), **I**nformed (kept in the loop).

| Activity                                     | Bus. owner | PO  | DS  | DE  | IT  | Validation | Risk | Compliance | Audit |
|----------------------------------------------|------------|-----|-----|-----|-----|------------|------|------------|-------|
| Define business need & success metrics        | A          | R   | C   |     |     |            | C    | C          | I     |
| Write user stories / acceptance criteria      |            | A/R | C   |     |     |            | C    | C          |       |
| Define target variable & default definition   | C          | R   | R   |     |     | C          | A    | C          |       |
| Source raw data & build feature store         |            | C   | C   | A/R |     |            |      |            |       |
| Train candidate model                         |            | C   | A/R | C   |     |            |      |            |       |
| Validate model (independent challenge)        | I          | I   | C   | C   |     | A/R        | C    | C          | I     |
| Fairness / bias audit                         |            | C   | R   |     |     | A          | C    | C          |       |
| Approve model for production                  | I          | I   |     |     |     | C          | A    | C          | I     |
| Deploy to production                          |            | I   | C   | C   | A/R | I          | I    | I          |       |
| Monitor performance & drift                   |            | C   | C   | R   | C   | I          | A    | I          |       |
| Investigate alerts (PSI breach, drift, etc.) |            | C   | R   | C   | C   | C          | A    | I          |       |
| Periodic recalibration                        |            | C   | A/R | C   |     | C          | C    | I          |       |
| Decommission / replace                        | A          | R   | C   | C   | C   | C          | C    | C          | I     |
| Respond to GDPR Art. 22 request               |            | C   | R   |     |     |            |      | A          | I     |
| Annual internal audit                         |            | I   | C   | C   | C   | C          | C    | C          | A/R   |
| Regulator request (ECB / EBA / local)         | I          | I   | C   | C   |     | C          | C    | A/R        | I     |

---

## Decision rights (escalation path)

```
   Day-to-day model decisions
              │
              ▼
   Product Owner ── Data Scientist ── Risk Management
              │
              │   (cannot resolve / material change)
              ▼
   Model Risk Committee  (chaired by CRO, monthly)
              │
              │   (≥ €50k expected P&L impact / regulatory issue)
              ▼
   Executive Risk Committee  (chaired by CEO, quarterly)
              │
              │   (regulator engagement / public disclosure)
              ▼
   Board Risk Committee
```

---

## Communication cadence

| Audience           | Channel              | Frequency  | Owner       |
|--------------------|----------------------|------------|-------------|
| Model team standup | Slack                | Daily      | DS          |
| Risk Mgmt          | Monitoring dashboard | Daily auto | DE          |
| Risk Mgmt          | Drift review meeting | Weekly     | DS / Risk   |
| CRO office         | KPI pack             | Monthly    | PO          |
| Model Risk Cttee   | Validation pack      | Quarterly  | Validation  |
| Internal Audit     | Annual review        | Annual     | Audit       |
| Regulator          | On request           | ad-hoc     | Compliance  |
