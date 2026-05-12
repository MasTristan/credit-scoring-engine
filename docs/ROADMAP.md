# Product roadmap

> Three phases from this MVP to a production-grade credit decisioning
> service. Each phase has a clear exit criterion (the bar that has to be
> met before moving on), an owner, and an indicative scope.

---

## Phase 0 — MVP (this repo, ✅ done)

**Goal**: prove that an explainable XGBoost model on a public credit
dataset hits the discrimination bar (Gini ≥ 0.44, KS ≥ 0.35) and that
the explanations satisfy local accuracy.

**Scope shipped**

- End-to-end training pipeline (`src/data_prep.py`, `src/train.py`)
- Inference (`src/predict.py`) + SHAP (`src/explain.py`)
- 5-tab Streamlit app (Scorer, Portfolio, Performance, Methodology, About)
- 15-test pytest suite incl. SHAP local-accuracy invariant
- Public artefacts committed (model + metrics + 1,000-row sample)
- 12 governance documents (this folder)

**Exit criterion (met)**

- Gini = 0.558 on hold-out > 0.44 benchmark.
- SHAP local-accuracy verified within `1e-3`.
- App runs on Streamlit Community Cloud with zero paid licenses.

---

## Phase 1 — Production-ready model (Q3 2026, 3 FTE × 3 months)

**Goal**: take the same modelling approach and make it suitable for an
internal pilot on **real EU consumer-credit data**.

**Scope**

1. **Data**
   - Replace the UCI dataset with the bank's own application data
     (≥ 24 months of bookings, post-origination tagged at 12-month MOB).
   - Build a proper feature store (Feast / in-house) with named features,
     refresh cadence, and ownership.
   - Add bureau-data joins (current account history, past accounts).

2. **Modelling**
   - Train on the bank data with the same code path; expected Gini in
     the 0.55–0.65 range.
   - Add **reject inference** (parcelling, augmentation) — the training
     data must approximate the through-the-door population, not the
     accepted-only population.
   - Add **isotonic calibration** (already designed in `src/calibration.py`):
     the operational PD must be interpretable as a long-run frequency.

3. **Fairness**
   - Audit on protected attributes (sex, age, nationality, postcode-as-proxy).
   - Pre-deployment sign-off on disparate-impact thresholds.
   - Document in `docs/MODEL_CARD.md` for each release.

4. **Operations**
   - MLflow model registry (replacing committed-to-git artefacts).
   - FastAPI inference service behind an internal API gateway.
   - Daily drift monitoring (PSI / CSI / calibration drift).

5. **Governance**
   - Independent validation sign-off (P3, Anna).
   - Model Risk Committee approval.

**Exit criterion**

- Validation report signed.
- Disparate-impact ratio ∈ [0.8, 1.25] for all monitored attributes.
- API p95 latency < 80 ms.
- Champion-challenger framework live (Phase 2 prerequisite).

---

## Phase 2 — Champion-challenger & full automation (Q1 2027, 4 FTE × 4 months)

**Goal**: progressively replace the legacy scorecard in production
through a controlled A/B test, while building the long-tail features
that turn the system from "model" into "product".

**Scope**

1. **Champion-challenger A/B test**
   - See `docs/AB_TEST_DESIGN.md`.
   - Start at 1% ramp, 5%, 25%, 100% with stop-loss guardrails.
   - Primary metric: net P&L per application; guardrail metrics:
     approval rate, complaint rate, fairness ratios.

2. **Counterfactual explanations**
   - "What would have to change to flip the decision?" — surfaces an
     actionable recommendation for refer / reject cases.
   - Integrated with the customer-service UI used by the credit officers.

3. **Customer-facing reason codes**
   - Beyond the credit officer; the applicant sees the top 3 reasons
     in plain language (in their preferred language).
   - GDPR Art. 22 hardening.

4. **Cost-sensitive thresholding UI**
   - Live € sliders for cost of FN / FP, threshold auto-tunes,
     deployment with policy-board approval.

5. **Adverse-event playbooks**
   - On-call procedures for: PSI breach, calibration drift, fairness
     ratio breach, model-service incident.

**Exit criterion**

- A/B test crosses statistical significance with the model ahead on
  net P&L per application.
- Customer-facing reason codes translated into all served languages.
- 30-day production stability with no P1 incidents.

---

## Phase 3 — Continuous lifecycle (ongoing)

**Goal**: keep the model in calibration without growing the team.

**Scope (ongoing, 0.5 FTE)**

- Quarterly recalibration cycle (data freeze → retrain → validate →
  challenger run → promote).
- Annual independent validation.
- Drift alerts integrated with the bank's incident management system.
- Roadmap reviews twice a year against business KPIs.

---

## Backlog (not yet prioritised)

- Multi-product support (auto loan, mortgage) — likely separate models
  rather than a unified one.
- Behavioural model on the **booked** population (i.e. PD on existing
  customers, for limit reviews / collections).
- LGD and EAD modelling for Basel III IRB advanced.
- Federated learning across affiliated entities to avoid centralising PII.

---

## Out of scope

- **Pricing** — owned by the pricing engine team.
- **Marketing scoring** — owned by the CRM team; very different optimisation
  function (uplift, not PD).
- **Fraud detection** — owned by a separate model, runs in parallel in
  step 5 of the process flow.
