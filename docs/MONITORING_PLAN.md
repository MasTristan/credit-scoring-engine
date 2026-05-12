# Production monitoring plan

> What to monitor, at what cadence, with which alert thresholds, and
> who picks up the phone. Aligned with SR 11-7 "ongoing monitoring" and
> the ECB Guide on internal models (2024).

---

## Monitoring layers

```
   Layer 1 — Pre-inference  ──► input validation, schema, range checks
   Layer 2 — Data drift     ──► PSI per feature, daily
   Layer 3 — Score drift    ──► CSI on PD distribution, daily
   Layer 4 — Performance    ──► AUC / Gini / KS on label-tagged batches, weekly
   Layer 5 — Calibration    ──► reliability diagram + Brier, monthly
   Layer 6 — Fairness       ──► DI / EOD / AUC by sub-group, monthly
   Layer 7 — Operational    ──► p95 latency, error rate, availability
```

---

## Metrics, thresholds, and alerting

### Layer 1 — Pre-inference

| Metric                                        | Threshold        | Action                            |
|-----------------------------------------------|------------------|------------------------------------|
| Input out of schema (type / required missing)  | Any single occurrence | Reject score; refer to manual review |
| Input out of valid range (e.g. `AGE < 18`)     | Any single occurrence | Refer to manual review            |

### Layer 2 — Data drift (PSI)

The Population Stability Index quantifies the distributional difference
between a baseline (training data) and a current batch:

$$\mathrm{PSI} = \sum_{i=1}^{B} (p_i - q_i) \ln\frac{p_i}{q_i}$$

| PSI    | Interpretation       | Action                                                |
|--------|----------------------|--------------------------------------------------------|
| < 0.10 | No significant shift  | No action                                              |
| 0.10–0.25 | Moderate shift      | Investigation by Data Science within 5 working days     |
| > 0.25 | Large shift           | Page Risk Mgmt; freeze approval-rate change until reviewed |

**Cadence**: daily, on the previous day's inference logs. Run per
feature; aggregate dashboard flags any feature breaching.

### Layer 3 — Score drift (CSI on PD)

Same formula as PSI but applied to the PD distribution itself.

| CSI    | Action                                              |
|--------|------------------------------------------------------|
| < 0.10 | No action                                            |
| 0.10–0.25 | Trigger calibration check                          |
| > 0.25 | Page Risk Mgmt; champion-challenger comparison       |

### Layer 4 — Performance (requires labels)

Labels arrive with a lag (label = "default in month t+1"). Performance
monitors run on a **rolling 30-day cohort** that has reached label
maturity.

| Metric    | Floor               | Action below floor                                   |
|-----------|---------------------|-------------------------------------------------------|
| Gini      | 0.40                | Investigate within 10 working days                    |
| Gini      | 0.30                | Trigger emergency recalibration                       |
| KS        | 0.25                | Investigate within 10 working days                    |
| ROC-AUC   | 0.65                | Trigger emergency recalibration                       |

### Layer 5 — Calibration

Monthly reliability diagram on the matured cohort.

| Metric                                | Threshold                           | Action                          |
|---------------------------------------|--------------------------------------|----------------------------------|
| Brier score (model)                   | > 0.20                              | Trigger isotonic recalibration   |
| Reliability deviation                 | any decile > ±5pp from y=x          | Refit calibrator                  |
| Calibration intercept (HL test)        | p-value < 0.01                      | Investigate                       |

### Layer 6 — Fairness

| Metric                                                | Acceptable band   | Action outside band              |
|-------------------------------------------------------|-------------------|-----------------------------------|
| Disparate-impact ratio (DI) on `SEX`                   | [0.80, 1.25]      | Compliance escalation              |
| Equal-opportunity difference (EOD) on `SEX`            | ±0.10             | Compliance escalation              |
| AUC by age band                                       | within ±0.05      | Investigate                        |
| Approval-rate ratio by sub-group                       | [0.80, 1.25]      | Compliance escalation              |

### Layer 7 — Operational

| Metric                  | Target  | Alert    |
|-------------------------|---------|----------|
| p95 inference latency   | < 80 ms | > 150 ms |
| Error rate              | < 0.1%  | > 1%     |
| Availability (monthly)   | 99.9%   | < 99.5%  |

---

## Dashboard mock-up

```
   ┌────────────────────────────────────────────────────────────────┐
   │  CREDIT-SCORING MODEL — PROD HEALTH                  ▼ today    │
   ├────────────────────────────────────────────────────────────────┤
   │                                                                │
   │  Scores today      Approvals      p95 latency    Availability  │
   │     8,412            61.3%         42 ms          100%         │
   │                                                                │
   │  ─── DRIFT ────────────────────────────────────────────────── │
   │  Max PSI (PAY_0)    Max CSI (PD)   Features in alert            │
   │     0.08             0.04            0                          │
   │                                                                │
   │  ─── PERFORMANCE (30-day rolling, label-mature cohort) ─────── │
   │  Gini   0.541  ▏      KS  0.421  ▏    Brier  0.182  ▏           │
   │                                                                │
   │  ─── FAIRNESS (last month) ────────────────────────────────── │
   │  DI(sex) 1.04 ✔     AUC(<30) 0.74  AUC(30-45) 0.76  AUC(>45) 0.77 │
   │                                                                │
   │  ─── ALERTS ─────────────────────────────────────────────────  │
   │  ●  No active alerts                                           │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘
```

This static layout is rendered in the Governance tab of the Streamlit
app as a screenshot mock-up; in production it would be a live dashboard
backed by the model's inference logs.

---

## On-call playbook

For every alert type, the on-call runbook specifies:

1. **Triage** — who picks up first (Data Eng → Data Science → Risk Mgmt
   escalation).
2. **Investigation steps** — log queries, drift attribution, feature-
   level diagnostics.
3. **Decision tree** — when to keep serving, when to freeze approvals,
   when to roll back to the legacy scorecard.
4. **Communications** — who to inform (CRO, Compliance, business owner).
5. **Post-mortem template** — root-cause analysis within 48 h.

The runbooks live in `docs/runbooks/` in a production deployment; they
are stubbed in this MVP as `docs/runbooks/README.md` (placeholder).

---

## Cadence summary

| Cadence       | Activity                                                              |
|---------------|------------------------------------------------------------------------|
| Real-time     | Layer 1 input validation; Layer 7 operational metrics                  |
| Daily         | Layer 2 PSI per feature; Layer 3 CSI on PD                              |
| Weekly        | Layer 4 performance on matured cohort                                   |
| Monthly       | Layer 5 calibration; Layer 6 fairness; full dashboard review with CRO   |
| Quarterly     | Independent validation revalidation; recalibration if needed             |
| Annually      | Full retrain with refreshed feature store and rerun of validation report |

---

## Tooling

- **Metrics store**: Prometheus + Grafana (or the bank's existing stack).
- **Drift computation**: nightly batch job in the data lake.
- **Fairness audit**: monthly notebook executed by Validation team.
- **Alerts**: PagerDuty (on-call) + email (Risk Mgmt + Compliance).
- **Log storage**: append-only object store, immutable for ≥ 7 years
  (regulatory retention).
