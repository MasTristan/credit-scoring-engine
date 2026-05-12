# Champion-challenger A/B test design

> The model only earns its place in production by beating the incumbent
> on a randomised, statistically powered, fairness-aware A/B test. This
> document is the pre-registration of that test.

---

## 1. Hypothesis

**H₀ (null)**: the new model (challenger) produces the same expected
net P&L per application as the incumbent (champion).

**H₁ (alternative, one-sided)**: the new model produces a higher
expected net P&L per application than the incumbent, at the same
overall approval rate.

We test one-sided because a non-superior model has no business case
for replacement.

---

## 2. Population and randomisation

- **Eligible population**: all incoming applications that pass KYC.
- **Exclusions**: VIP segment, B2B, applications < age 21 (regulatory
  cap), applications above the policy maximum loan size.
- **Randomisation unit**: the application (independent across time).
- **Randomisation seed**: deterministic hash of application ID modulo
  100 → arm assignment. Recorded in the application record.
- **Allocation**: see ramp plan in §4. The ramp varies; arm boundaries
  are stable within a ramp phase.

---

## 3. Metrics

### Primary

| Name                        | Definition                                                                              |
|-----------------------------|------------------------------------------------------------------------------------------|
| Net P&L per application      | (margin on performing accepted loans − expected loss on defaulting accepted loans) / N    |

### Secondary

| Name                  | Definition                                                            |
|-----------------------|------------------------------------------------------------------------|
| Default rate (booked)  | defaults / booked, on the booked cohort                                 |
| Approval rate         | accepted / scored                                                       |
| Average PD (booked)   | mean PD of accepted contracts                                            |
| Gini (booked, with labels) | discrimination on labelled cohort                                       |

### Guardrails (must not deteriorate vs. champion)

| Name                           | Tolerance                                  |
|--------------------------------|---------------------------------------------|
| Approval rate                  | Must not move > ±2 pp                       |
| DI ratio on `SEX`              | Must remain in [0.80, 1.25]                |
| Customer complaint rate        | Must not increase > 0.5 pp                  |
| Manual-override rate           | Must not increase > 5 pp                    |
| p95 inference latency           | Must not exceed 150 ms                      |

Breach of **any guardrail** for two consecutive days triggers an
auto-pause and a Risk Mgmt review.

---

## 4. Ramp plan

| Phase | Challenger traffic | Duration       | Exit criterion to next phase                            |
|-------|--------------------|----------------|----------------------------------------------------------|
| Ramp 1 | 1%                | 2 weeks        | No guardrail breach; operational stability               |
| Ramp 2 | 5%                | 4 weeks        | No guardrail breach; preliminary net P&L not worse        |
| Ramp 3 | 25%               | 6 weeks        | Net P&L lift statistically significant @ α = 0.05         |
| Ramp 4 | 50%               | 4 weeks        | Confirm lift; fairness ratios stable                      |
| Cutover | 100%             | —              | Final sign-off by Model Risk Committee                    |

Total minimum duration: ≈ 4 months. Conservative; protects against
slow-emerging issues (e.g. macro inflection).

---

## 5. Statistical power

### Sample size

Sources of variance: per-application net P&L has standard deviation
≈ €1,200 (driven by default outcomes). To detect a **€5/application
lift** (≈ 8% of the mean margin) at:

- α = 0.05 (one-sided),
- power 0.80,

required sample size per arm:

$$n = \frac{2\sigma^2 (z_{1-\alpha} + z_{1-\beta})^2}{\delta^2}$$

$$n \approx \frac{2 \times 1200^2 \times (1.645 + 0.84)^2}{5^2} \approx
\text{≈ 715,000 applications per arm.}$$

At 100,000 monthly applications and a 25% allocation, this is reached
in 28 days (Ramp 3). The full ramp plan exceeds the minimum sample
size by ≈ 3×, providing margin against attrition and segmentation.

### Multiple testing

We monitor 5 secondary metrics + 5 guardrails. Bonferroni correction
applied: α_secondary = 0.05 / 5 = 0.01.

### Sequential testing

The pre-registered analysis is computed at the **end** of each ramp
phase (not continuously) to avoid alpha inflation. Mid-phase monitors
are for **operational** monitoring (PSI / latency / error rate) only.

---

## 6. Stop conditions

The test **must stop immediately** if any of the following occurs:

1. Net cumulative loss on the challenger arm exceeds €100k vs. champion.
2. Approval rate drift on any single day > 5 pp.
3. DI ratio on `SEX` outside [0.70, 1.40] on any single day.
4. Complaint rate exceeds 1% on any single day.
5. Two consecutive days of guardrail breach.
6. Active regulator request.

Stop = roll back to 100% champion within 30 minutes; conduct post-mortem
within 48 h.

---

## 7. Analysis plan

Pre-registered, performed by Validation (independent of the model build
team):

1. Frequentist 1-sided t-test on net P&L per application,
   α = 0.05 (Bonferroni-adjusted for secondaries).
2. Bootstrap confidence interval (10,000 resamples) on the net P&L
   lift as a sanity check.
3. Subgroup analysis — net P&L lift by age band, sex, education,
   geography. Pre-declared subgroups only.
4. Calibration analysis on the challenger arm (reliability diagram).
5. Final report submitted to the Model Risk Committee.

---

## 8. Roles

| Role               | Responsibility                                         |
|--------------------|---------------------------------------------------------|
| PO (this profile)   | Pre-registration document, exit-criterion sign-off       |
| Risk Mgmt           | Guardrail definitions, escalation on breach              |
| Data Science        | Implementation of the challenger; weekly health report   |
| Data Engineering    | Randomisation, arm-assignment integrity                  |
| Validation (2nd L)  | Final analysis, statistical significance assessment      |
| Compliance          | Fairness sign-off after each ramp                        |
| Model Risk Cttee    | Phase-gate approval; full cutover sign-off                |

---

## 9. Reporting cadence

- **Daily** during each ramp — operational monitor (latency, error,
  drift) by Data Engineering.
- **Weekly** — health report (metrics + fairness ratios) by Data Science.
- **End-of-phase** — pre-registered statistical analysis by Validation.
- **End-of-test** — full A/B test report submitted to Model Risk Cttee.

---

## 10. Communications

External (regulator, statutory auditor): on demand only.
Internal (CRO office, business owner): summary at every phase gate.
Public (annual report disclosures): aggregated, no per-applicant data.
