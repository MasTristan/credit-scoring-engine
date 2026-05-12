"""ML Credit Scoring Engine — Streamlit application.

Four tabs: Individual Scorer, Portfolio Analysis, Model Performance,
About this model.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data_prep import build_feature_matrix
from src.explain import (
    compute_shap_values,
    get_explainer,
    get_global_importance,
    get_waterfall_data,
)
from src.predict import (
    align_features,
    load_feature_names,
    load_model,
    predict_proba,
    score_to_rating,
    score_to_risk_band,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

BAND_COLOURS = {"LOW": "#00B050", "MEDIUM": "#FFC000", "HIGH": "#FF4444"}
MODELS_DIR = Path("models")
SAMPLE_PATH = Path("data/sample/sample_1000.csv")

st.set_page_config(
    page_title="ML Credit Scoring Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = load_model(MODELS_DIR / "xgboost_model.json")
    background = pd.read_parquet(MODELS_DIR / "shap_background.parquet")
    explainer = get_explainer(model, background)
    feature_names = load_feature_names(MODELS_DIR / "feature_names.json")
    with open(MODELS_DIR / "training_metrics.json") as f:
        metrics = json.load(f)
    return model, explainer, feature_names, metrics


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data
def score_sample(_model_id: str, feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Return the sample with PD / RATING / RISK_BAND columns added."""
    df = load_sample().copy()
    raw_for_features = df.rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
    X = build_feature_matrix(raw_for_features)
    X = align_features(X.drop(columns=["IS_DEFAULT"]), list(feature_names))
    model = load_artifacts()[0]
    df["PD"] = predict_proba(model, X)
    df["RATING"] = df["PD"].apply(score_to_rating)
    df["RISK_BAND"] = df["PD"].apply(score_to_risk_band)
    return df


model, explainer, feature_names, metrics = load_artifacts()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("ML Credit Scoring Engine")
    st.markdown("---")
    st.markdown("**Model summary**")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.metric("Gini", f"{metrics['gini']:.3f}")
    st.metric("KS statistic", f"{metrics['ks_statistic']:.3f}")
    st.metric("Training observations", f"{metrics['n_train']:,}")
    st.markdown("---")
    st.markdown(
        "**Tristan Mas** — Business Analyst Risk & Finance IT\n\n"
        "[GitHub](https://github.com/MasTristan) · "
        "[LinkedIn](https://linkedin.com/in/tristan-mas)"
    )


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_scorer, tab_portfolio, tab_model, tab_methodology, tab_about = st.tabs(
    [
        "Individual Scorer",
        "Portfolio Analysis",
        "Model Performance",
        "Methodology",
        "About this model",
    ]
)


# --------------------------------------------------------------------------
# Tab 1 — Individual Scorer
# --------------------------------------------------------------------------

with tab_scorer:
    st.header("Score an individual contract")
    st.caption(
        "Fill the form with the client's credit profile and click "
        "**Score this client** to compute the probability of default and the "
        "SHAP feature attribution."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account")
        limit_bal = st.number_input(
            "Credit limit (NT$)", 10_000, 1_000_000, 200_000, 10_000
        )
        age = st.slider("Age", 21, 79, 35)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        education = st.selectbox(
            "Education",
            ["Graduate school", "University", "High school", "Other"],
            index=1,
        )
        marriage = st.selectbox("Marriage", ["Married", "Single", "Other"], index=1)

    with col2:
        st.subheader("Repayment status (last 6 months)")
        st.caption("−2 = no consumption · −1 = paid in full · 0 = revolving · 1..8 = months late")
        pay_0 = st.slider("Current month (PAY_0)", -2, 8, 0)
        pay_2 = st.slider("1 month ago (PAY_2)", -2, 8, 0)
        pay_3 = st.slider("2 months ago (PAY_3)", -2, 8, 0)
        pay_4 = st.slider("3 months ago (PAY_4)", -2, 8, 0)
        pay_5 = st.slider("4 months ago (PAY_5)", -2, 8, 0)
        pay_6 = st.slider("5 months ago (PAY_6)", -2, 8, 0)

    with col3:
        st.subheader("Bills and payments")
        st.caption("Average bill and payment amounts (NT$) over the last 6 months")
        bill_mean = st.number_input("Average bill amount", 0, 500_000, 40_000, 1000)
        pay_amt_mean = st.number_input("Average payment amount", 0, 500_000, 5000, 500)
        bill_amt1 = st.number_input("Most recent bill (BILL_AMT1)", 0, 1_000_000, int(bill_mean), 1000)
        pay_amt1 = st.number_input("Most recent payment (PAY_AMT1)", 0, 1_000_000, int(pay_amt_mean), 500)

    if st.button("Score this client", type="primary"):
        # Build a one-row raw frame and run it through the same pipeline.
        bill_cols = {f"BILL_AMT{i}": bill_mean for i in range(1, 7)}
        bill_cols["BILL_AMT1"] = bill_amt1
        pay_amt_cols = {f"PAY_AMT{i}": pay_amt_mean for i in range(1, 7)}
        pay_amt_cols["PAY_AMT1"] = pay_amt1
        education_map = {
            "Graduate school": 1,
            "University": 2,
            "High school": 3,
            "Other": 4,
        }
        marriage_map = {"Married": 1, "Single": 2, "Other": 3}
        raw_row = pd.DataFrame(
            [{
                "LIMIT_BAL": limit_bal,
                "SEX": 1 if sex == "Male" else 2,
                "EDUCATION": education_map[education],
                "MARRIAGE": marriage_map[marriage],
                "AGE": age,
                "PAY_0": pay_0,
                "PAY_2": pay_2,
                "PAY_3": pay_3,
                "PAY_4": pay_4,
                "PAY_5": pay_5,
                "PAY_6": pay_6,
                **bill_cols,
                **pay_amt_cols,
                "IS_DEFAULT": 0,  # placeholder, dropped below
            }]
        )
        X_one = build_feature_matrix(raw_row).drop(columns=["IS_DEFAULT"])
        X_one = align_features(X_one, feature_names)

        pd_value = float(predict_proba(model, X_one)[0])
        rating = score_to_rating(pd_value)
        band = score_to_risk_band(pd_value)

        st.markdown("### Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability of default", f"{pd_value:.1%}")
        m2.metric("Internal rating", rating)
        m3.metric("Risk band", band)
        st.progress(min(pd_value, 1.0))

        # SHAP waterfall
        shap_vals = compute_shap_values(explainer, X_one)
        waterfall = get_waterfall_data(
            shap_vals[0], X_one.iloc[0], feature_names, top_n=15
        )
        fig = go.Figure(
            go.Bar(
                x=waterfall["SHAP_VALUE"],
                y=waterfall["FEATURE_LABEL"],
                orientation="h",
                marker_color=[
                    "#FF4444" if v > 0 else "#00B050"
                    for v in waterfall["SHAP_VALUE"]
                ],
                text=[f"{v:+.3f}" for v in waterfall["SHAP_VALUE"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="SHAP contribution — Top 15 features",
            xaxis_title="SHAP value (impact on log-odds of default)",
            height=500,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=80, t=60, b=40),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Red bars push the prediction towards default. Green bars push it "
            "away. Bar length = magnitude of impact on the model output."
        )


# --------------------------------------------------------------------------
# Tab 2 — Portfolio Analysis
# --------------------------------------------------------------------------

with tab_portfolio:
    st.header("Portfolio analysis")
    st.caption(
        "Scores a public sample of 1,000 contracts drawn from the hold-out "
        "test set."
    )

    scored = score_sample("xgb-v1", tuple(feature_names))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio size", f"{len(scored):,} contracts")
    c2.metric("Average PD", f"{scored['PD'].mean():.1%}")
    high_n = int((scored["PD"] > 0.15).sum())
    c3.metric(
        "High risk (PD>15%)",
        f"{high_n:,}",
        f"{(scored['PD'] > 0.15).mean():.1%}",
    )
    c4.metric("Median credit limit", f"NT${scored['LIMIT_BAL'].median():,.0f}")

    st.subheader("PD distribution")
    fig = px.histogram(
        scored,
        x="PD",
        nbins=50,
        color="RISK_BAND",
        color_discrete_map=BAND_COLOURS,
        labels={"PD": "Probability of default", "count": "Number of contracts"},
        title="PD distribution across the portfolio",
    )
    fig.add_vline(x=0.05, line_dash="dash", line_color="gray",
                  annotation_text="LOW / MEDIUM")
    fig.add_vline(x=0.15, line_dash="dash", line_color="gray",
                  annotation_text="MEDIUM / HIGH")
    st.plotly_chart(fig, width="stretch")

    st.subheader("PD vs credit limit")
    scatter_sample = scored.sample(min(500, len(scored)), random_state=42)
    scatter_sample = scatter_sample.assign(
        BILL_AMT1_SIZE=scatter_sample["BILL_AMT1"].clip(lower=1000)
    )
    fig = px.scatter(
        scatter_sample,
        x="LIMIT_BAL",
        y="PD",
        size="BILL_AMT1_SIZE",
        color="RISK_BAND",
        color_discrete_map=BAND_COLOURS,
        hover_data=["AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"],
        title="PD vs credit limit (bubble size = most recent bill amount)",
        labels={"LIMIT_BAL": "Credit limit (NT$)", "PD": "Probability of default"},
        opacity=0.65,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Default rate by current repayment status (PAY_0)")
    pay0_stats = (
        scored.groupby("PAY_0")
        .agg(Avg_PD=("PD", "mean"), Count=("PD", "count"))
        .reset_index()
        .sort_values("PAY_0")
    )
    fig = px.bar(
        pay0_stats,
        x="PAY_0",
        y="Avg_PD",
        color="Avg_PD",
        color_continuous_scale=["#00B050", "#FFC000", "#FF4444"],
        hover_data=["Count"],
        title="Average PD by current month's repayment status",
        labels={"Avg_PD": "Average PD", "PAY_0": "Repayment status (current month)"},
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Global SHAP feature importance (sample of 200)")
    sub = scored.sample(min(200, len(scored)), random_state=42)
    sub_raw = sub.rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
    X_sub = build_feature_matrix(sub_raw)
    X_sub = align_features(X_sub.drop(columns=["IS_DEFAULT"]), feature_names)
    shap_matrix = compute_shap_values(explainer, X_sub)
    global_imp = get_global_importance(shap_matrix, feature_names).head(20)
    fig = px.bar(
        global_imp,
        x="MEAN_ABS_SHAP",
        y="FEATURE",
        orientation="h",
        title="Global feature importance — top 20 (mean |SHAP|)",
        labels={"MEAN_ABS_SHAP": "Mean |SHAP value|", "FEATURE": ""},
        color="MEAN_ABS_SHAP",
        color_continuous_scale=["#C8E6FA", "#0070C0"],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Tab 3 — Model Performance
# --------------------------------------------------------------------------

with tab_model:
    st.header("Model performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c2.metric("Gini", f"{metrics['gini']:.4f}")
    c3.metric("KS statistic", f"{metrics['ks_statistic']:.4f}")
    c4.metric(
        "Brier score",
        f"{metrics['brier_score']:.4f}",
        help="Lower is better. Perfect model = 0.",
    )

    scored = score_sample("xgb-v1", tuple(feature_names))
    y_true = scored["IS_DEFAULT_TRUE"].astype(int).values
    y_proba = scored["PD"].values

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name="XGBoost", line=dict(color="#0070C0", width=2))
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                       line=dict(color="gray", dash="dash"))
        )
        fig.update_layout(
            title=f"ROC — sample (AUC≈{metrics['roc_auc']:.3f} on full test)",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.subheader("Precision-Recall curve")
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=recall, y=precision, name="XGBoost",
                       line=dict(color="#0070C0", width=2))
        )
        fig.add_hline(
            y=metrics["default_rate_test"],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Baseline ({metrics['default_rate_test']:.1%})",
        )
        fig.update_layout(
            title=f"PR — sample (PR-AUC≈{metrics['pr_auc']:.3f} on full test)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Score distribution by actual outcome")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=y_proba[y_true == 0],
            name="Non-default",
            opacity=0.7,
            marker_color="#00B050",
            nbinsx=50,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=y_proba[y_true == 1],
            name="Default",
            opacity=0.7,
            marker_color="#FF4444",
            nbinsx=50,
        )
    )
    fig.update_layout(
        barmode="overlay",
        title=f"Predicted PD distribution by actual outcome (KS={metrics['ks_statistic']:.3f})",
        xaxis_title="Predicted PD",
        yaxis_title="Count",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Full metrics")
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": list(metrics.keys()),
                "Value": [
                    f"{v:.4f}" if isinstance(v, float) else str(v)
                    for v in metrics.values()
                ],
            }
        ).set_index("Metric"),
        width="stretch",
    )


# --------------------------------------------------------------------------
# Tab 4 — Methodology (maths, design choices, recruiter-facing narrative)
# --------------------------------------------------------------------------

with tab_methodology:
    st.header("Methodology")
    st.caption(
        "Why this project matters, the maths behind the model and the "
        "explainability layer, and the engineering choices made along the way."
    )

    st.subheader("Why this project")
    st.markdown(
        """
        Credit scoring sits at the intersection of three constraints that are
        usually in tension:

        - **Predictive performance** — the model must rank borrowers well
          enough to materially reduce expected loss versus a baseline policy.
        - **Explainability** — under SR 11-7 (Federal Reserve) and EBA
          Guidelines on internal models, a bank must be able to justify
          every individual credit decision and demonstrate ongoing model
          governance. A black-box scorecard is not acceptable.
        - **Operational simplicity** — the model is recomputed on every
          contract, sometimes thousands of times per second; inference
          latency and infrastructure cost matter.

        This app demonstrates a stack that resolves all three: a
        gradient-boosted tree ensemble (XGBoost) for performance, SHAP for
        per-decision explanations grounded in cooperative game theory, and a
        single-binary Streamlit deployment that scales to public traffic at
        zero cost.
        """
    )

    st.subheader("The credit scoring problem")
    st.markdown(
        r"""
        Given a borrower described by a feature vector $x \in \mathbb{R}^d$,
        the model estimates the probability of default within a defined
        horizon:

        $$
        \widehat{PD}(x) = \mathbb{P}\big(\text{IS\_DEFAULT}=1 \mid X = x\big)
        $$

        From this PD we derive:

        - the **expected loss** $EL = PD \times LGD \times EAD$ (LGD and EAD
          are policy parameters, not modelled here);
        - the **internal rating** via a step function on PD (Project 1
          mapping: AAA/AA, A, BBB, BB, B, CCC, D);
        - the **risk band** for UI colour coding (LOW < 5%, MEDIUM < 15%,
          HIGH ≥ 15%).
        """
    )

    st.subheader("Why XGBoost over a logistic regression scorecard")
    st.markdown(
        r"""
        A classical PD scorecard fits a logistic regression on
        Weight-of-Evidence-transformed bins. It is transparent but typically
        leaves 5–10 Gini points on the table because it cannot model
        interactions or non-linearities without explicit feature engineering.

        Gradient boosting fits an additive ensemble

        $$
        f(x) = \sum_{m=1}^{M} \eta\, T_m(x)
        $$

        where each $T_m$ is a shallow decision tree minimising a second-order
        approximation of the logistic loss with $\ell_1$/$\ell_2$
        regularisation on the leaf weights:

        $$
        \mathcal{L}^{(t)} = \sum_i \left[ g_i T_t(x_i) +
        \tfrac{1}{2} h_i T_t(x_i)^2 \right]
        + \gamma\,\lvert T_t \rvert
        + \tfrac{1}{2}\lambda\sum_j w_j^2
        $$

        with $g_i$ and $h_i$ the first/second derivatives of the loss at the
        current prediction. Two consequences matter for credit risk:

        1. **Interactions are learned automatically.** A logistic
           scorecard requires us to know in advance that PAY_0×UTILIZATION
           matters; the tree ensemble discovers it.
        2. **Monotonicity / regularisation are knobs**, not assumptions.
           `max_depth=6`, `min_child_weight=5`, `gamma=0.1`,
           `subsample=0.8`, and `early_stopping_rounds=50` together
           control the bias–variance trade-off.

        The class imbalance (≈22% default rate) is handled with
        `scale_pos_weight = n_neg / n_pos` rather than over-sampling, so the
        training distribution is preserved.
        """
    )

    st.subheader("SHAP — the maths of the explanations")
    st.markdown(
        r"""
        SHAP (SHapley Additive exPlanations) assigns to each feature $i$ of
        an observation $x$ a contribution $\phi_i(x)$ such that

        $$
        f(x) = \phi_0 + \sum_{i=1}^{d} \phi_i(x)
        $$

        where $\phi_0 = \mathbb{E}[f(X)]$ is the model's average prediction
        and $f(x)$ is the predicted log-odds of default. The $\phi_i$ are
        the Shapley values from cooperative game theory:

        $$
        \phi_i(x) = \sum_{S \subseteq F \setminus \{i\}}
        \frac{|S|!\,(d - |S| - 1)!}{d!}
        \big[ f_{S \cup \{i\}}(x) - f_S(x) \big]
        $$

        i.e. the average marginal contribution of feature $i$ across all
        possible orderings of the feature set $F$.

        Three properties make this the standard for model governance:

        - **Local accuracy** — the explanations literally add up to the
          prediction; there is no residual.
        - **Consistency** — if a feature's true contribution grows after a
          model update, its SHAP value cannot decrease.
        - **Missingness** — features that are not used by the model
          receive a SHAP value of zero.

        For tree ensembles, the naive computation is exponential in $d$.
        Lundberg et al. (2018) showed it can be computed in
        $O(T L D^2)$ where $T$ is the number of trees, $L$ the maximum
        number of leaves per tree, and $D$ the maximum depth. That is what
        the **Individual Scorer** tab uses — every waterfall is computed
        from scratch in well under a second.

        We verify the local-accuracy property in the test suite:
        `sum(shap_values) + base_value == logit(predict_proba)` within
        `1e-3`. See `tests/test_explain.py::test_shap_local_accuracy`.
        """
    )

    st.subheader("Metrics, in plain language")
    st.markdown(
        "Credit scoring uses a small set of standard metrics that capture "
        "different aspects of model quality:"
    )
    st.markdown(
        f"""
| Metric        | What it measures                                                                  | This model |
|---------------|------------------------------------------------------------------------------------|------------|
| **ROC-AUC**   | Probability that a random defaulter is ranked above a random non-defaulter         | {metrics['roc_auc']:.3f}        |
| **Gini**      | $2 \\cdot \\mathrm{{AUC}} - 1$ — Lorenz-curve area, used in most retail-credit decks  | {metrics['gini']:.3f}           |
| **KS**        | Maximum vertical distance between the CDFs of defaulters and non-defaulters        | {metrics['ks_statistic']:.3f}   |
| **PR-AUC**    | Average precision — robust to class imbalance                                       | {metrics['pr_auc']:.3f}         |
| **Brier**     | Mean squared error of the probability output; 0 = perfect, lower is better         | {metrics['brier_score']:.3f}    |
| **Log-loss**  | Negative log-likelihood; the loss the model is actually trained on                 | {metrics['log_loss']:.3f}       |

Industry rule-of-thumb benchmarks for unsecured consumer credit:
ROC-AUC ≥ 0.72, Gini ≥ 0.44, KS ≥ 0.35. This model clears all
three on the hold-out test set.
        """
    )

    st.subheader("Technical choices, justified")
    st.markdown(
        """
        | Choice | Alternative | Why this one |
        |---|---|---|
        | XGBoost (CPU, `tree_method=hist`) | LightGBM, CatBoost | Mature, single-file model serialisation, deterministic, well-supported by SHAP, fits on a free Streamlit instance. |
        | Stratified 80/20 random split | Temporal split | The UCI dataset only contains 6 months of payment data and no contract issue date, so a meaningful temporal split is not available. The original Lending Club brief planned 2015-2017 train / 2018 test. |
        | `scale_pos_weight = n_neg / n_pos` | SMOTE, random over-sampling | Preserves the empirical default rate, keeps the trained model directly interpretable as conditional probabilities, avoids the leakage risks of synthetic minority oversampling. |
        | Early stopping on `aucpr` | Fixed `n_estimators` | Cuts training time, prevents over-fit. The best iteration is logged. |
        | SHAP `TreeExplainer`, `tree_path_dependent` | `interventional`, KernelSHAP | Exact and fast for tree ensembles; satisfies the local-accuracy property by construction. |
        | Streamlit Community Cloud | Flask/FastAPI + a frontend | Zero infra, one-file deploy, free public URL, `@st.cache_resource` keeps the model warm across requests. |
        | Parquet for train/test/background | CSV | 5–10× smaller, preserves dtypes, columnar reads. |
        | All model artefacts committed to git | A model registry | The artefact set is small (≈1 MB) and a public repo is the cheapest registry possible. |
        """
    )

    st.subheader("Architecture in one diagram")
    st.code(
        """
        data/raw/uci_credit_card.csv  (30,000 rows, raw)
                       │
                       ▼
          src/data_prep.py
            • clean categoricals (EDUCATION → {1..4}, MARRIAGE → {1..3})
            • engineer 8 features (PAY_MEAN, DELINQ_COUNT, UTILIZATION, …)
            • one-hot encode SEX / EDUCATION / MARRIAGE (drop_first)
            • stratified 80/20 split
                       │
                       ▼
          data/processed/{train,test}.parquet + feature_names.json
                       │
                       ▼
          src/train.py     (XGBoost + early stopping)
                       │
                       ▼
          models/{xgboost_model.json, feature_importance.csv,
                  training_metrics.json, shap_background.parquet}
                       │
                       ▼
          streamlit_app.py
            • Tab 1: Individual Scorer  (PD + SHAP waterfall)
            • Tab 2: Portfolio Analysis (1,000-row sample)
            • Tab 3: Model Performance  (ROC + PR + KS)
            • Tab 4: Methodology        (this tab)
            • Tab 5: About this model
        """,
        language="text",
    )

    st.subheader("What I would do next in a production setting")
    st.markdown(
        """
        - **Population stability index (PSI)** monitoring on the input
          distribution, with automated alerts when a feature drifts beyond a
          governance threshold.
        - **Calibration** with isotonic regression or Platt scaling, and
          a reliability diagram in this tab.
        - **Reject-inference** — the training data only contains accepted
          contracts; a production model needs to correct for selection bias.
        - **Counterfactual explanations** in addition to SHAP attributions:
          "what is the smallest change in feature space that would flip the
          decision?" — closer to what a lender's customer-facing explainer
          actually needs.
        - **Model registry + CI** — currently the artefacts are committed to
          git; in production I would push to MLflow / SageMaker / Vertex and
          version both the model and the training data hash.
        - **A/B-test the threshold** in production against the existing
          policy on a small fraction of traffic before broad rollout.
        """
    )

    st.subheader("References")
    st.markdown(
        """
        - Yeh, I.-C., & Lien, C.-h. (2009). *The comparisons of data mining
          techniques for the predictive accuracy of probability of default of
          credit card clients.* Expert Systems with Applications, 36(2).
        - Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting
          System.* KDD '16.
        - Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to
          Interpreting Model Predictions.* NeurIPS.
        - Lundberg, S. M. et al. (2020). *From local explanations to global
          understanding with explainable AI for trees.* Nature Machine
          Intelligence 2.
        - Federal Reserve SR 11-7 — *Guidance on Model Risk Management.*
        - EBA (2017). *Guidelines on PD estimation, LGD estimation and the
          treatment of defaulted exposures.*
        """
    )


# --------------------------------------------------------------------------
# Tab 5 — About
# --------------------------------------------------------------------------

with tab_about:
    st.header("Model documentation")

    with st.expander("Model overview", expanded=True):
        st.markdown(
            f"""
            **Algorithm**: XGBoost (eXtreme Gradient Boosting)
            **Training data**: UCI Default of Credit Card Clients (Taiwan, Apr–Sep 2005)
            **Split**: stratified 80 / 20 (random, seed = 42)
            **Target**: binary default indicator (1 = default next month)
            **Features**: {metrics['n_features']} after engineering and encoding
            **Explainability**: SHAP `TreeExplainer` (Lundberg & Lee, 2017)
            """
        )

    with st.expander("Why SHAP?"):
        st.markdown(
            """
            SHAP (SHapley Additive exPlanations) decomposes each prediction
            into the contribution of each feature, grounded in cooperative
            game theory (Shapley values). Key properties:

            - **Local accuracy** — SHAP values sum to the model output.
            - **Consistency** — if a feature's contribution increases, its
              SHAP value increases.
            - **Missingness** — features with no effect get a SHAP value of 0.

            This is the standard explainability framework referenced in
            model-governance guidelines (SR 11-7, EBA GL on internal models).
            """
        )

    with st.expander("Model limitations"):
        st.markdown(
            """
            - Training data is Taiwanese credit-card payments from 2005, not
              EU consumer lending — recalibration is required before any
              operational use.
            - Six months of payment history is short for a behavioural model;
              `PAY_0` and `PAY_MAX` dominate the feature importance.
            - The target is "default next month", so the model is point-in-time
              and does not account for macroeconomic cycles.
            - Class imbalance is handled with `scale_pos_weight`, not by
              resampling — predicted probabilities are calibrated towards
              detecting defaults rather than calibrated as long-run frequencies.
            """
        )

    with st.expander("Training details"):
        st.json(metrics)
