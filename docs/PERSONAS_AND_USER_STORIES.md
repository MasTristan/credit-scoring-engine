# Personas & user stories

The four tabs of the Streamlit app are not "screens" — they are answers to
four distinct people asking four distinct questions. This document names
those people and writes their needs down as user stories with acceptance
criteria.

---

## Personas

### P1 — Marie, Credit Officer (front line)

- **Day-to-day**: handles 40–80 applications per day, mostly by phone.
- **Goals**: decide quickly; defend the decision to the applicant if rejected.
- **Pain points**: a black-box score she cannot explain; ad-hoc overrides
  that drift from policy.
- **Quote**: *"If the system says no, I need to be able to tell the customer
  the top three reasons in plain language."*
- **App tab**: **Individual Scorer**.

### P2 — Karim, Portfolio Manager (middle office)

- **Day-to-day**: monitors the active book, flags emerging concentrations,
  reports monthly to the CRO.
- **Goals**: understand risk-band composition; spot drift in PD distribution;
  isolate sub-segments that are deteriorating.
- **Pain points**: data lives in five systems; building a portfolio view
  takes a week.
- **Quote**: *"I want one page that tells me whether my portfolio is
  riskier today than last month."*
- **App tab**: **Portfolio Analysis**.

### P3 — Anna, Model Validator (second line of defence)

- **Day-to-day**: independent challenge of every production model under
  SR 11-7. Writes the validation report that the board signs off.
- **Goals**: verify metrics, replicate the model on holdout, audit the
  feature engineering for leakage, check for bias on protected attributes.
- **Pain points**: opaque pipelines; missing model cards; metrics not
  reproducible from committed artefacts.
- **Quote**: *"I need to re-run your training script from a clean checkout
  and get the same numbers as your README."*
- **App tab**: **Model Performance** + **Governance** (and the repo itself).

### P4 — Hugo, Recruiter / Hiring Manager / CRO

- **Day-to-day**: scanning portfolios in 5 minutes; deciding whether to
  spend 45 minutes in interview.
- **Goals**: assess seniority, business sense, regulatory awareness,
  communication.
- **Pain points**: candidates over-index on technical novelty and under-index
  on framing.
- **Quote**: *"Tell me in one paragraph why this matters and what you would
  do next."*
- **App tab**: **Methodology** + the README.

---

## User stories

Each story follows the standard `As a … I want … so that …` template,
with explicit acceptance criteria. Stories are sized in t-shirt units
(S/M/L) and prioritised MoSCoW (M = must, S = should, C = could).

---

### Epic 1 — Score an individual contract (P1, Marie)

#### US-1.1 — `M` `S` Compute a PD for an arbitrary applicant

> As a credit officer, I want to enter an applicant's profile and receive
> a probability of default, an internal rating, and a risk band, so that
> I can make a defensible accept / reject recommendation.

**Acceptance criteria**

- The form accepts at least: credit limit, age, sex, education, marriage,
  the last 6 months of `PAY_*`, the most recent and average bill and
  payment amounts.
- The PD is computed in **under 200 ms** on a free Streamlit instance.
- Results display: `PD %`, internal rating from the `AAA/AA … D` scale,
  risk band `LOW / MEDIUM / HIGH`, and a progress bar.
- The result is **reproducible** across sessions when the form inputs
  are identical (no random sampling at inference time).

#### US-1.2 — `M` `M` Explain the decision with SHAP

> As a credit officer, I want to see the top 15 factors that pushed the
> score up or down for this specific applicant, so that I can explain
> the decision verbally and document the reason for a rejection.

**Acceptance criteria**

- Waterfall chart of the **top 15 features** by `|SHAP value|`, sorted
  by impact magnitude, colour-coded by direction.
- Each bar shows `feature = value` on the y-axis and the signed SHAP
  value on the x-axis.
- Caption explains red = "increases risk", green = "decreases risk".
- The chart and the score are produced from the same `model.predict_proba`
  call (no skew between the displayed PD and the explanation).

#### US-1.3 — `S` `S` Reason codes for adverse-action notices

> As a credit officer, I want a plain-language sentence summarising the
> top 3 negative factors, so that I can paste it into the rejection
> letter without re-typing.

**Acceptance criteria**

- Three short sentences, one per top-3 negative SHAP feature.
- Mapping from feature name to human-readable phrase
  (`PAY_0` → "your current month is overdue by N months", etc.).
- Available as a copy-paste text block under the waterfall.
- Implemented in the **Governance** tab as part of the GDPR Art. 22
  treatment.

#### US-1.4 — `C` `M` Counterfactual explanation

> As a credit officer, I want to see *what change in the applicant's
> profile* would have flipped the decision, so that I can advise on a
> resubmission plan or up-sell the credit-builder product.

**Acceptance criteria**

- "What would have to change to reach `PD < 5%`?" panel beneath the
  waterfall.
- Displays the smallest single-feature change (in the model's
  feature space) that crosses the threshold, plus its plain-language
  description.
- If no such single-feature change exists, surface a 2-feature
  counterfactual.

---

### Epic 2 — Monitor and slice the active book (P2, Karim)

#### US-2.1 — `M` `S` Score a 1,000-contract sample

> As a portfolio manager, I want to upload (or load the bundled) sample
> of 1,000 contracts and see the PD distribution, so that I can
> characterise the risk profile of a recent cohort.

**Acceptance criteria**

- The sample loads from `data/sample/sample_1000.csv` in under 1 second.
- KPI row: portfolio size, average PD, share `PD > 15%`, median credit limit.
- PD histogram with risk-band colour coding.

#### US-2.2 — `M` `M` Drill into sub-segments

> As a portfolio manager, I want to see PD by current-month repayment
> status and a PD-versus-credit-limit scatter, so that I can identify
> the segments driving aggregate risk.

**Acceptance criteria**

- Bar chart `Avg_PD ~ PAY_0`, sorted.
- Scatter `PD ~ LIMIT_BAL`, bubble-sized by `BILL_AMT1`, coloured by
  risk band.
- Both charts have hover tooltips.

#### US-2.3 — `S` `M` Global SHAP importance on the sample

> As a portfolio manager, I want to see which features drive risk
> *across* the portfolio (not just for one borrower), so that I can
> communicate to the CRO the systemic risk factors.

**Acceptance criteria**

- Horizontal bar chart of mean(|SHAP|) on a 200-row stratified subset.
- Top 20 features displayed.
- Chart is computed on a cached sample so the tab loads under 1 second
  on subsequent navigations.

---

### Epic 3 — Validate the model (P3, Anna)

#### US-3.1 — `M` `S` Reproducible metrics

> As a model validator, I want to re-run the training script from a
> clean checkout and get the same metrics that the README claims, so
> that I can sign off on the validation report.

**Acceptance criteria**

- `python -m src.data_prep && python -m src.train` produces
  `models/training_metrics.json` whose values match the README
  exactly.
- All randomness is seeded.
- Dependencies are pinned in `requirements.txt`.

#### US-3.2 — `M` `S` ROC, PR, KS and Brier reported

> As a model validator, I want the standard credit-risk discrimination
> and calibration metrics, so that I can compare to the bank's existing
> models.

**Acceptance criteria**

- Metrics table with ROC-AUC, Gini, KS, PR-AUC, Brier, log-loss.
- ROC and PR curves rendered.
- KS plot (score distribution by actual outcome) rendered.

#### US-3.3 — `S` `M` Calibration assessment

> As a model validator, I want a reliability diagram and a Brier-score
> decomposition, so that I can assess whether the model's PDs can be
> taken as long-run frequencies.

**Acceptance criteria**

- Reliability diagram with 10 quantile bins.
- Brier decomposed into reliability + resolution − uncertainty.
- A one-click "re-calibrate with isotonic regression" toggle.
- Implemented in the **Governance** tab.

#### US-3.4 — `M` `M` Fairness audit on protected attributes

> As a model validator, I want to see the AUC, default rate, and
> selection rate broken down by sex, age band, and education, so that
> I can flag any disparate impact.

**Acceptance criteria**

- Table of disparate-impact ratio (DI), equal-opportunity difference
  (EOD), and AUC by sex, age band (`<30`, `30–45`, `>45`), and
  education level.
- Cells with `DI < 0.8` or `DI > 1.25` are highlighted.
- Implemented in the **Governance** tab.

---

### Epic 4 — Demonstrate maturity to a hiring decision (P4, Hugo)

#### US-4.1 — `M` `S` Methodology narrative

> As a recruiter, I want a single tab that explains the problem, the
> maths, the design choices, and what would come next in production,
> so that I can assess the candidate's seniority in 10 minutes.

**Acceptance criteria**

- "Why this project" section.
- LaTeX-rendered XGBoost loss and Shapley formula.
- "Technical choices, justified" table.
- "What I would do next in production" section.

#### US-4.2 — `S` `S` Governance artefacts in the repo

> As a recruiter, I want to see model card, RACI, risk register, and
> regulatory mapping artefacts in the repo, so that I can assess
> business-analyst maturity (not just code maturity).

**Acceptance criteria**

- `docs/` folder with: `BUSINESS_CASE.md`, `RACI.md`,
  `REGULATORY_MAPPING.md`, `RISK_REGISTER.md`, `MODEL_CARD.md`,
  `DATA_DICTIONARY.md`, `MONITORING_PLAN.md`, `AB_TEST_DESIGN.md`,
  `GLOSSARY.md`, `PROCESS_FLOW.md`, `ROADMAP.md`, this file.
- README links to each.
