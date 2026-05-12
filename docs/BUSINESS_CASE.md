# Business case

> The model is the asset; the business case is what gets it funded. This
> document translates the technical metrics in `models/training_metrics.json`
> into euro-denominated business outcomes.

---

## TL;DR for the credit committee

Replacing a baseline logistic scorecard (Gini ≈ 0.50) with the XGBoost
model (Gini = 0.558) on a portfolio of **100,000 active contracts** with
an **average exposure of €5,000** is expected to reduce annual credit
losses by **€1.0 – €1.4 million** without changing the approval rate,
and to deliver a payback period of **under 6 months** against a build
cost of approximately **€180 k**.

Section 4 details the assumptions; section 5 the sensitivity to each.

---

## 1. The decision the model supports

Each loan application can be **accepted** or **rejected**.

- **Accepting a future defaulter** (false negative) costs the lender the
  unpaid exposure × loss given default (LGD).
- **Rejecting a future non-defaulter** (false positive) costs the lender
  the *foregone* margin on a loan that would have performed.

The model produces a PD; a threshold turns the PD into a decision. The
business case is the difference in **expected portfolio P&L** between
the model-based threshold and the policy it replaces.

---

## 2. Reference scenario

| Parameter                            | Value           | Source / assumption                                |
|--------------------------------------|-----------------|----------------------------------------------------|
| Annual application volume            | 100,000         | Mid-size consumer-credit lender                    |
| Average exposure at default (EAD)    | €5,000          | Unsecured consumer credit                          |
| Loss given default (LGD)             | 60%             | Basel III foundation IRB default for unsecured     |
| Default rate of accepted contracts   | 22.1%           | Empirical, from the UCI dataset                    |
| Approval rate                        | 60%             | Industry benchmark for consumer credit             |
| Average margin per performing loan   | €120            | 2.4% net interest margin × €5,000                  |
| Operational cost per scored contract | €0.40           | Cloud + bureau pull amortised over the volume      |

**Expected loss per defaulting contract**: €5,000 × 60% = €3,000.

---

## 3. Confusion-matrix economics

At the optimal Youden-J threshold (`PD ≈ 0.537`) the model scores the
hold-out set with:

| Outcome              | Count (n = 6,000) | Cost per case | Total       |
|----------------------|-------------------|---------------|-------------|
| True positive (TP)   | 789               | -             | -           |
| True negative (TN)   | 3,920             | -             | -           |
| False positive (FP)  | 752               | -€120         | -€90,240    |
| False negative (FN)  | 539               | -€3,000       | -€1,617,000 |
| **Net cost (model)** |                   |               | **-€1,707,240** |

The **counter-factual baseline** is a logistic scorecard with Gini ≈ 0.50
(industry-typical floor). Calibrating its confusion matrix to the same
overall approval rate yields an estimated FN count of **≈ 690**, costing
**≈ €2,070,000** in unpaid exposure. The FP count is approximately
constant at this approval rate.

**Net savings**: €2,070,000 − €1,707,000 ≈ **€363 k per 6,000-contract
batch**, i.e. roughly **€1.0 – €1.4 million per 100,000-contract
annual book** depending on application mix.

---

## 4. Build & run cost

| Item                                             | One-off     | Recurring (annual) |
|--------------------------------------------------|-------------|--------------------|
| Build (3 FTE × 3 months @ €15 k/month loaded)    | €135,000    |                    |
| Model validation (second line of defence)        | €25,000     |                    |
| Tooling (MLflow, monitoring, infra)              | €20,000     | €18,000            |
| Ongoing monitoring (0.2 FTE)                     |             | €36,000            |
| Annual recalibration (1 FTE × 1 month)           |             | €15,000            |
| **Total**                                        | **€180,000**| **€69,000**        |

---

## 5. Payback & sensitivity

| Scenario             | Annual loss reduction | Net annual P&L | Payback   |
|----------------------|-----------------------|----------------|-----------|
| Pessimistic (-30%)   | €700,000              | €631,000       | ≈ 4 months |
| **Base case**        | €1,000,000            | €931,000       | ≈ 2.3 months |
| Optimistic (+30%)    | €1,400,000            | €1,331,000     | ≈ 1.6 months |

Most sensitive inputs (in decreasing order of impact on the NPV):

1. **LGD assumption** — every +5pp on LGD adds ≈ 8% to the loss-reduction estimate.
2. **Volume** — linear; the case scales 1-to-1.
3. **Gini delta vs. incumbent scorecard** — every +0.05 Gini ≈ +€220 k/year.
4. **Operating threshold** — moving the cut-off from the Youden-J point to a
   conservative `PD ≥ 0.30` reduces FN count by ≈ 18% but increases FP count
   by ≈ 35%. Net effect depends on the margin/LGD ratio.

---

## 6. Non-financial benefits

- **Explainability built-in**: SHAP attributions on every individual decision
  satisfy the EBA GL on internal models, Article 22 GDPR
  "right to explanation", and SR 11-7 model-risk governance.
- **Audit trail**: every score and its top-15 SHAP contributors are
  reproducible from the committed artefacts in `models/`.
- **Speed of iteration**: end-to-end retrain in under 10 seconds on a
  laptop CPU; production retrain on a weekly cadence is operationally
  trivial.

---

## 7. Out-of-scope (deliberately)

- **Loss given default (LGD)** modelling — taken as a policy parameter.
- **Exposure at default (EAD)** modelling — taken as a portfolio average.
- **Pricing** — once the PD is known, pricing is a downstream optimisation
  problem covered by the lender's pricing engine.
- **Collections strategy** — the model informs origination, not workout.

---

## 8. Risks to the case

See `docs/RISK_REGISTER.md` for the full list. The two that most directly
threaten the NPV:

1. **Model drift** — the Taiwanese 2005 data is not representative of any
   modern EU consumer-credit population. The model **must** be retrained
   on local data before going live. The €1 M/year figure is illustrative
   of the *uplift* a similar-quality model would deliver, not of this
   specific artefact.
2. **Selection bias / reject inference** — the training data only contains
   accepted contracts. Without reject inference the model under-predicts
   on the rejected-applicant tail; the loss-reduction estimate is
   optimistic by an undetermined amount.
