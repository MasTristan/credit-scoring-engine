# Glossary

> Plain-language definitions of the terms used in this repo. Aimed at
> recruiters, business stakeholders, and credit officers, not at
> machine-learning practitioners.

---

## Credit-risk terms

**PD — Probability of Default.** The probability that a borrower will
fail to pay according to contract within a defined horizon (here:
one month). Range [0, 1]; expressed as a percentage. The output of
this model.

**LGD — Loss Given Default.** The fraction of exposure the lender
expects to lose when a default occurs. Typical value for unsecured
consumer credit: 60–70%. **Not modelled here** — taken as a policy
parameter.

**EAD — Exposure At Default.** The amount the borrower owes at the
moment of default. For revolving credit, this is approximately
the drawn balance plus a fraction of the unused limit. **Not modelled
here.**

**EL — Expected Loss.** `EL = PD × LGD × EAD`. The "central estimate"
of credit loss, the building block of provisions (IFRS 9) and
regulatory capital (Basel III).

**IRB — Internal Ratings-Based.** Basel III framework allowing banks
to use their own models for PD (and, in IRB-advanced, LGD and EAD)
to compute regulatory capital.

**Default rate.** The fraction of borrowers in a cohort that
defaulted within the horizon. Empirical, observed ex-post.

**Approval rate.** The fraction of applications that are accepted by
the lender. A policy choice; a tighter cut-off lowers approval rate
and lowers default rate.

**Internal rating.** A discrete grade (`AAA/AA … D`) derived from the
PD by step thresholds. Used in internal reporting and pricing.

---

## Model performance terms

**ROC-AUC.** Area under the Receiver Operating Characteristic curve.
The probability that a randomly chosen defaulter is ranked above a
randomly chosen non-defaulter. 0.5 = random; 1.0 = perfect.
**Industry floor for retail credit: 0.72.**

**Gini.** `2 × ROC-AUC − 1`. Range [0, 1]; same ranking content as
ROC-AUC but more familiar in retail-credit decks. **Industry floor:
0.44.**

**KS — Kolmogorov–Smirnov statistic.** Maximum vertical distance
between the cumulative distribution of scores for defaulters and
non-defaulters. **Industry floor: 0.35.**

**PR-AUC.** Area under the Precision-Recall curve. Robust to class
imbalance; useful when defaulters are rare.

**Brier score.** Mean squared error of the predicted PD vs. the
binary outcome. 0 = perfect; lower is better. Decomposes into
**reliability − resolution + uncertainty**, of which reliability
is the calibration error.

**Log-loss.** Negative log-likelihood; the loss the model is
actually trained on.

**Precision, Recall, F1.** Standard classifier metrics at a chosen
threshold. Precision = (correct positives) / (predicted positives);
recall = (correct positives) / (actual positives); F1 is their
harmonic mean.

**Youden-J threshold.** The threshold that maximises
`TPR − FPR`; the "best" threshold for a balanced cost of FP and FN.

---

## Explainability terms

**SHAP — SHapley Additive exPlanations.** Per-prediction attributions
that decompose the model output into a sum of feature contributions,
grounded in cooperative game theory (Shapley 1953). Used here to
explain individual decisions.

**Local accuracy.** SHAP property: the sum of feature attributions
plus the base value equals the model output exactly. Verified by
the test suite (`tests/test_explain.py::test_shap_local_accuracy`).

**Waterfall.** A horizontal bar chart of the top N features by
|SHAP value| for a single prediction. Standard form factor in the
SHAP library.

**Global SHAP importance.** Mean of `|SHAP|` across many predictions
per feature; ranks features by *typical* impact rather than by
training-time information gain.

**Counterfactual explanation.** "What is the smallest change in
feature space that would flip the decision?" Different from SHAP:
SHAP says *why this decision*, counterfactual says *what to change*.

**Reason code.** A short, plain-language sentence summarising one
of the top negative SHAP contributors. Used in adverse-action
notices.

---

## Monitoring terms

**PSI — Population Stability Index.** Distributional distance between
a baseline and a current sample on a single feature. PSI < 0.10
≈ no shift; 0.10–0.25 ≈ moderate; > 0.25 ≈ large shift.

**CSI — Characteristic Stability Index.** Same formula as PSI but
applied to the model **score** distribution rather than a single
feature.

**Data drift.** The input distribution shifts away from the training
distribution.

**Concept drift.** The relationship between features and target shifts
(e.g. a macro break, a policy change).

**Calibration drift.** The predicted PD no longer matches the
observed default frequency in the appropriate quantile.

**Reliability diagram.** A plot of observed default rate vs.
predicted PD, by quantile. A perfectly calibrated model lies on
the `y = x` line.

---

## Fairness terms

**Protected attribute.** A characteristic on which discrimination
is prohibited (sex, race, religion, age in some jurisdictions).

**Disparate impact (DI) ratio.** Ratio of selection rates between
two groups (e.g. female approval rate / male approval rate). The
"four-fifths rule" considers `DI ∈ [0.80, 1.25]` acceptable.

**Equal-opportunity difference (EOD).** Difference in true-positive
rates between two groups; ideally close to zero.

---

## Process terms

**KYC — Know Your Customer.** Identity-verification activity carried
out before any credit-risk assessment.

**AML — Anti-Money-Laundering.** Compliance checks against
sanctions lists, politically-exposed-person lists, etc.

**MOB — Months On Book.** Time since contract origination. "Default
at MOB+12" means a default observed 12 months after the loan
was booked.

**Three lines of defence.** Governance model used in regulated banks:
1st line (business / building) — 2nd line (risk, compliance) —
3rd line (internal audit).

---

## Regulatory terms

**Basel III / CRR.** Prudential framework for bank capital. Defines
the IRB regime for internal PD/LGD/EAD models.

**EBA.** European Banking Authority. Issues guidelines on internal
models (EBA/GL/2017/16 for PD/LGD).

**ECB.** European Central Bank. Single supervisor for significant
banks in the Eurozone; runs the TRIM exercise and publishes the
Guide on internal models.

**SR 11-7.** Federal Reserve / OCC supervisory guidance on model
risk management. Defines model validation, ongoing monitoring,
documentation expectations.

**GDPR Article 22.** EU General Data Protection Regulation. The
right not to be subject to a decision based solely on automated
processing, and the right to obtain a meaningful explanation.

**EU AI Act (Regulation 2024/1689).** Credit scoring is classified
high-risk (Annex III §5(b)); obligations on risk management,
data governance, transparency, human oversight, accuracy.

**SR 11-7 "effective challenge".** The principle that a model
must be subject to independent challenge by people with
sufficient skill, authority, and incentive to identify weaknesses.
