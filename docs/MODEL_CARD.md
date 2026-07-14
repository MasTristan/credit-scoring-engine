# Model card

> Format adapted from Mitchell et al. (2019) and Hugging Face Model Cards.
> A single document that a model risk manager, an internal auditor, or a
> regulator can read in under 15 minutes to understand what this model
> is, what it isn't, and where it should not be used.

---

## Model details

| Field                | Value                                                                   |
|----------------------|--------------------------------------------------------------------------|
| Model name           | `xgboost-credit-pd-v1`                                                  |
| Version              | 1.0.0                                                                    |
| Release date         | 2026-05-12                                                              |
| Model type           | Binary classifier, gradient-boosted trees (XGBoost ≥ 2.0)               |
| Output               | Probability of default within 1 month (range [0, 1])                     |
| Number of features   | 34 after engineering and one-hot encoding                                |
| Model size           | ≈ 900 KB (`models/xgboost_model.json`)                                  |
| Owner (1st line)     | Data Science team                                                        |
| Owner (2nd line)     | Model Validation                                                         |
| Sponsor              | Head of Retail Credit                                                    |
| Inference latency    | < 80 ms p95 single-row (SHAP included)                                   |
| Training time        | ≈ 8 s on a single CPU core                                               |
| Source code          | https://github.com/MasTristan/LendingClubCreditScoring                   |
| Commit hash          | tracked in `models/training_metrics.json::training_date`                  |

---

## Intended use

### Primary use case

Estimate the **probability that a new credit-card / unsecured consumer
loan applicant will default in the next monthly billing cycle**, to
inform an accept / refer / reject decision in a regulated lender's
origination workflow.

### Primary users

- **Credit officers** (front line), to defend an individual decision
  and produce an adverse-action notice if rejected.
- **Portfolio managers** (middle office), to monitor PD distribution
  and concentration risks across a portfolio.
- **Model validation** (second line), for independent challenge.
- **Internal audit** (third line), for periodic review.

### Out-of-scope uses

- **Marketing / cross-sell scoring**, the optimisation function is
  different (uplift, not PD).
- **Pricing**, the lender's pricing engine should consume the PD,
  not be replaced by it.
- **Collections strategy**, this is an origination model, not a
  behavioural model on the booked book.
- **Compliance scoring** (AML / fraud), separate models, separate
  features, separate label.
- **Decisions on populations not represented in training**, see
  Limitations.

---

## Training data

| Field                | Value                                                                   |
|----------------------|--------------------------------------------------------------------------|
| Source               | UCI ML Repository, *Default of Credit Card Clients*                     |
| Provider             | Yeh, I.-C., & Lien, C.-h. (2009), Chung Hua University, Taiwan         |
| Population           | Cash and revolving credit card clients of a Taiwanese bank               |
| Period               | April – September 2005                                                   |
| Size                 | 30,000 contracts                                                          |
| Default rate         | 22.1%                                                                     |
| Train/val/test split | Stratified 60/20/20 (random, seed = 42); early stopping and threshold selected on validation, test scored once |
| Special-category data | None used, no race, religion, political opinion, health, biometric    |
| Protected attributes  | `SEX`, `AGE` are present (audited; see Fairness section)                |

### Why this dataset

This is a public, widely cited credit-scoring benchmark. The original
project brief targeted Lending Club 2015–2018, which is no longer
publicly available; the UCI Taiwan dataset is the standard substitute
in academic credit-risk literature (≈ 200 published papers).

### Pre-processing

See `src/data_prep.py`. In summary:

- `EDUCATION` codes 0, 5, 6 mapped to 4 ("other").
- `MARRIAGE` code 0 mapped to 3 ("other").
- 8 engineered features (`PAY_MEAN`, `PAY_MAX`, `DELINQ_COUNT`,
  `BILL_MEAN`, `PAY_AMT_MEAN`, `UTILIZATION`, `PAYMENT_RATIO`,
  `LIMIT_PER_AGE`).
- One-hot encoding of `SEX`, `EDUCATION`, `MARRIAGE` (`drop_first=True`).

---

## Training procedure

- **Algorithm**: XGBoost classifier, `tree_method="hist"`.
- **Hyperparameters**:
  ```python
  n_estimators=500, max_depth=6, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
  gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
  early_stopping_rounds=50,
  eval_metric=["auc","aucpr"], random_state=42
  ```
- **Class imbalance**: not re-weighted and not resampled. Training on the
  natural distribution keeps the PDs calibrated (see the Calibration section).
- **Loss**: binary cross-entropy, second-order Taylor approximation per
  XGBoost.
- **Decision threshold**: Youden-J optimal on the validation set, reported as a
  reference operating point. The shipped decision rule is the three-tier policy
  below, not this single cut-off.

### Decision policy (three tiers)

A single hard threshold forces a binary accept/reject on every applicant,
including the large middle band where the model is genuinely uncertain, which is
what produces an unconvincing false-positive count. The model instead auto-
decides only the confident tails and refers the grey zone to a human underwriter:

| Band | Rule | Share (test) | Actual default rate |
|------|------|-------------:|--------------------:|
| Auto-approve | PD < 0.14 | 45.9% | 8.7% |
| Manual review | 0.14 ≤ PD < 0.365 | 36.8% | 21.6% |
| Auto-decline | PD ≥ 0.365 | 17.3% | 58.8% |

Both cut-offs are fit on the validation split from interpretable risk targets
(approved band default rate ≤ 8%, declined band ≥ 60%), never on test, and are
stored in `models/decision_policy.json`. The auto-decided tails sit far from the
22% portfolio average; the review band is where discrimination is genuinely
ambiguous. This is why the model should be judged on ranking and calibration
(AUC 0.78, ECE 0.01), plus the risk separation of these bands, rather than a raw
confusion matrix at one threshold.

**Economic view.** `src/cost_analysis.py` prices the policy in euros. The
cost-driven break-even PD is `(margin + cost_fp) / (margin + cost_fp + cost_fn)`
≈ 7.4% under the default assumptions (margin €120, FP €120, FN €3,000); the
review band brackets it, which is where paying a human to make the call is
justified. `policy_pnl` scores the three-tier rule against approving everyone
and the best single threshold, with an explicit reviewer-effectiveness knob:
the review tier beats a plain threshold once reviewers catch roughly 85% of the
grey-zone defaulters, and loses below that. The Governance → *Cost-sensitive
thresholding* panel makes this interactive.

---

## Performance metrics (hold-out test set, n = 6,000)

| Metric            | Value  | Benchmark (consumer credit) |
|-------------------|--------|------------------------------|
| ROC-AUC           | 0.781  | ≥ 0.72                       |
| Gini              | 0.561  | ≥ 0.44                       |
| KS statistic      | 0.426  | ≥ 0.35                       |
| PR-AUC            | 0.557  | varies with default rate     |
| Brier score       | 0.135  | lower is better              |
| Log-loss          | 0.429  | lower is better              |
| ECE (10 bins)     | 0.010  | lower is better              |
| Precision @ J     | 0.484  |                              |
| Recall @ J        | 0.603  |                              |
| F1 @ J            | 0.537  |                              |

The Methodology tab in the app explains each metric in plain language.

### Calibration

The model is trained on the natural class distribution (no `scale_pos_weight`,
no resampling), so the predicted PDs are calibrated as long-run default
frequencies: mean predicted PD 0.221 against a 0.221 observed base rate, ECE
0.010, reliability (Murphy) ≈ 0.0001. An earlier `scale_pos_weight = n_neg /
n_pos ≈ 3.5` inflated every PD to roughly twice the base rate (ECE 0.19) with
no ranking benefit; a post-hoc isotonic recalibration was also benchmarked and
did not improve validation calibration over the native model, so neither is
used in the shipped pipeline.

### Performance by sub-group

See `src/fairness.py` and the Governance tab. At time of writing, the
disparate-impact ratio on `SEX` is within the [0.8, 1.25] regulator-
acceptable band; AUC is consistent across age bands within ±0.02.

### Calibration monitoring

The shipped model is calibrated out of the box (see the Calibration section
above). `src/calibration.py` additionally provides a reliability diagram and an
isotonic recalibrator, rendered in the Governance tab, kept as the operational
remedy if monitoring detects calibration drift.

---

## Ethical considerations

- **Sex as a feature**: `SEX` is encoded as `SEX_MALE`. In many EU
  jurisdictions this is a **protected attribute** for credit
  decisions; for a production deployment the feature would be
  removed and the model retrained, with the fairness audit checking
  that the removal does not re-introduce discrimination through
  correlated proxies.
- **Age**: present as a continuous feature. Not directly prohibited but
  monitored for adverse impact on young and elderly cohorts.
- **No marketing use**: the model and its outputs must not be used for
  customer segmentation, marketing scoring, or product cross-sell
  without a separate impact assessment.

---

## Limitations

- **Geographic & temporal generalisation**: trained on Taiwanese
  2005 data; will **not** generalise to a 2026 European population
  without retraining. The €1 M/year figure in `docs/BUSINESS_CASE.md`
  is illustrative of the *uplift* a similar-quality model would
  deliver, not of this specific artefact.
- **Short observation period**: 6 months of payment history; `PAY_0`
  and `PAY_MAX` dominate feature importance, which means the model
  has limited information for thin-file applicants.
- **No reject inference**: training data contains only accepted
  contracts; the model under-estimates risk on the rejected-applicant
  tail.
- **Calibration**: PDs are calibrated as long-run frequencies (test ECE
  0.010), so they can be used directly for loss-provisioning or pricing;
  monitor for drift and recalibrate if the reliability diagram degrades.

---

## Maintenance

- **Recalibration cadence**: quarterly, or earlier if any monitor
  triggers (see `docs/MONITORING_PLAN.md`).
- **Full retrain**: annually, or earlier on macro-economic break.
- **Deprecation**: end of life when Gini drops below 0.45 on three
  consecutive monthly evaluations, or when a successor model passes
  the champion-challenger A/B test (see `docs/AB_TEST_DESIGN.md`).

---

## Citation

If you use this work or this artefact:

```bibtex
@misc{mas2026xgbcredit,
  author       = {Mas, Tristan},
  title        = {{XGBoost + SHAP credit-scoring engine on the UCI
                  Default of Credit Card Clients dataset}},
  year         = {2026},
  howpublished = {\url{https://github.com/MasTristan/LendingClubCreditScoring}}
}
```

---

## Contact

**Tristan Mas**, Business Analyst, Risk & Finance IT
[GitHub](https://github.com/MasTristan) · [LinkedIn](https://linkedin.com/in/tristan-mas)
