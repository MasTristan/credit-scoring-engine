"""ML Credit Scoring Engine — Streamlit application.

Six tabs: Individual Scorer, Portfolio Analysis, Model Performance,
Methodology, Governance, About this model.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src.calibration import (
    apply_calibrator,
    assess_calibration,
    expected_calibration_error,
    fit_isotonic,
)
from src.cost_analysis import (
    CostInputs,
    breakeven_pd,
    confusion_at_threshold,
    optimal_threshold,
    policy_pnl,
    portfolio_pnl,
    sweep_thresholds,
)
from src.counterfactuals import find_counterfactual
from src.data_prep import build_feature_matrix
from src.decision_policy import APPROVE, DECLINE, REVIEW, evaluate_policy, load_policy
from src.explain import (
    compute_shap_values,
    get_explainer,
    get_global_importance,
    get_waterfall_data,
)
from src.fairness import fairness_summary
from src.i18n import LANGUAGES, set_language, t
from src.predict import (
    align_features,
    load_feature_names,
    load_model,
    predict_proba,
    score_to_rating,
    score_to_risk_band,
)
from src.reason_codes import format_adverse_action_block, top_negative_reasons

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
    policy = load_policy(MODELS_DIR / "decision_policy.json")
    return model, explainer, feature_names, metrics, policy


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
    df["DECISION"] = load_artifacts()[4].decide_batch(df["PD"].values)
    return df


model, explainer, feature_names, metrics, policy = load_artifacts()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    # Language selector — must run before any t() call below or in the tabs.
    lang_code = st.radio(
        "Language / Langue",
        options=list(LANGUAGES.keys()),
        format_func=lambda c: LANGUAGES[c],
        horizontal=True,
        key="lang",
    )
    set_language(lang_code)

    st.title("ML Credit Scoring Engine")
    st.markdown("---")
    st.markdown(t("**Model summary**", "**Résumé du modèle**"))
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.metric("Gini", f"{metrics['gini']:.3f}")
    st.metric("KS statistic", f"{metrics['ks_statistic']:.3f}")
    st.metric(
        t("Training observations", "Observations d'entraînement"),
        f"{metrics['n_train']:,}",
    )
    st.markdown("---")
    st.markdown(
        t(
            "**Tristan Mas** — Business Analyst Risk & Finance IT\n\n",
            "**Tristan Mas** — Business Analyst Risk & Finance IT\n\n",
        )
        + "[GitHub](https://github.com/MasTristan) · "
        "[LinkedIn](https://linkedin.com/in/tristan-mas)"
    )


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

(
    tab_scorer,
    tab_portfolio,
    tab_model,
    tab_methodology,
    tab_governance,
    tab_about,
) = st.tabs(
    [
        t("Individual Scorer", "Scoring individuel"),
        t("Portfolio Analysis", "Analyse de portefeuille"),
        t("Model Performance", "Performance du modèle"),
        t("Methodology", "Méthodologie"),
        t("Governance", "Gouvernance"),
        t("About this model", "À propos du modèle"),
    ]
)


# --------------------------------------------------------------------------
# Tab 1 — Individual Scorer
# --------------------------------------------------------------------------

with tab_scorer:
    st.header(t("Score an individual contract", "Scorer un contrat individuel"))
    st.caption(
        t(
            "Fill the form with the client's credit profile and click "
            "**Score this client** to compute the probability of default and the "
            "SHAP feature attribution.",
            "Renseignez le profil de crédit du client et cliquez sur "
            "**Scorer ce client** pour calculer la probabilité de défaut et "
            "l'attribution SHAP des variables.",
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(t("Account", "Compte"))
        limit_bal = st.number_input(
            t("Credit limit (NT$)", "Plafond de crédit (NT$)"),
            10_000, 1_000_000, 200_000, 10_000,
        )
        age = st.slider(t("Age", "Âge"), 21, 79, 35)
        sex = st.radio(
            t("Sex", "Sexe"), ["Male", "Female"], horizontal=True,
            format_func=lambda v: {"Male": t("Male", "Homme"),
                                    "Female": t("Female", "Femme")}[v],
        )
        education = st.selectbox(
            t("Education", "Éducation"),
            ["Graduate school", "University", "High school", "Other"],
            index=1,
            format_func=lambda v: {
                "Graduate school": t("Graduate school", "Master/Doctorat"),
                "University": t("University", "Université"),
                "High school": t("High school", "Lycée"),
                "Other": t("Other", "Autre"),
            }[v],
        )
        marriage = st.selectbox(
            t("Marriage", "Situation familiale"),
            ["Married", "Single", "Other"], index=1,
            format_func=lambda v: {
                "Married": t("Married", "Marié(e)"),
                "Single": t("Single", "Célibataire"),
                "Other": t("Other", "Autre"),
            }[v],
        )

    with col2:
        st.subheader(t("Repayment status (last 6 months)",
                       "Statut de remboursement (6 derniers mois)"))
        st.caption(t(
            "−2 = no consumption · −1 = paid in full · 0 = revolving · 1..8 = months late",
            "−2 = pas de consommation · −1 = payé intégralement · 0 = crédit renouvelable · 1..8 = mois de retard",
        ))
        pay_0 = st.slider(t("Current month (PAY_0)", "Mois courant (PAY_0)"), -2, 8, 0)
        pay_2 = st.slider(t("1 month ago (PAY_2)", "Il y a 1 mois (PAY_2)"), -2, 8, 0)
        pay_3 = st.slider(t("2 months ago (PAY_3)", "Il y a 2 mois (PAY_3)"), -2, 8, 0)
        pay_4 = st.slider(t("3 months ago (PAY_4)", "Il y a 3 mois (PAY_4)"), -2, 8, 0)
        pay_5 = st.slider(t("4 months ago (PAY_5)", "Il y a 4 mois (PAY_5)"), -2, 8, 0)
        pay_6 = st.slider(t("5 months ago (PAY_6)", "Il y a 5 mois (PAY_6)"), -2, 8, 0)

    with col3:
        st.subheader(t("Bills and payments", "Factures et paiements"))
        st.caption(t(
            "Average bill and payment amounts (NT$) over the last 6 months",
            "Montants moyens de facture et de paiement (NT$) sur les 6 derniers mois",
        ))
        bill_mean = st.number_input(
            t("Average bill amount", "Montant moyen de facture"), 0, 500_000, 40_000, 1000)
        pay_amt_mean = st.number_input(
            t("Average payment amount", "Montant moyen de paiement"), 0, 500_000, 5000, 500)
        bill_amt1 = st.number_input(
            t("Most recent bill (BILL_AMT1)", "Dernière facture (BILL_AMT1)"),
            0, 1_000_000, int(bill_mean), 1000)
        pay_amt1 = st.number_input(
            t("Most recent payment (PAY_AMT1)", "Dernier paiement (PAY_AMT1)"),
            0, 1_000_000, int(pay_amt_mean), 500)

    if st.button(t("Score this client", "Scorer ce client"), type="primary"):
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

        st.markdown(t("### Result", "### Résultat"))
        m1, m2, m3 = st.columns(3)
        m1.metric(t("Probability of default", "Probabilité de défaut"), f"{pd_value:.1%}")
        m2.metric(t("Internal rating", "Notation interne"), rating)
        m3.metric(t("Risk band", "Bande de risque"), band)
        st.progress(min(pd_value, 1.0))

        decision = policy.decide(pd_value)
        if decision == APPROVE:
            st.success(t(
                f"**Decision: AUTO-APPROVE** — PD {pd_value:.1%} is below the "
                f"approve cut-off ({policy.approve_below:.0%}). Low-risk, no "
                "manual underwriting needed.",
                f"**Décision : ACCEPTATION AUTOMATIQUE** — PD {pd_value:.1%} sous le "
                f"seuil d'acceptation ({policy.approve_below:.0%}). Faible risque, "
                "pas d'analyse manuelle nécessaire.",
            ))
        elif decision == DECLINE:
            st.error(t(
                f"**Decision: AUTO-DECLINE** — PD {pd_value:.1%} is at or above "
                f"the decline cut-off ({policy.decline_at_or_above:.0%}). "
                "High-risk, clear reject.",
                f"**Décision : REFUS AUTOMATIQUE** — PD {pd_value:.1%} au-dessus du "
                f"seuil de refus ({policy.decline_at_or_above:.0%}). "
                "Risque élevé, refus net.",
            ))
        else:
            st.warning(t(
                f"**Decision: MANUAL REVIEW** — PD {pd_value:.1%} sits in the grey "
                f"zone ({policy.approve_below:.0%}–{policy.decline_at_or_above:.0%}). "
                "The model defers to a human underwriter rather than force a "
                "binary call. See the reason codes below.",
                f"**Décision : ÉTUDE MANUELLE** — PD {pd_value:.1%} dans la zone grise "
                f"({policy.approve_below:.0%}–{policy.decline_at_or_above:.0%}). "
                "Le modèle laisse la main à un analyste plutôt que de forcer une "
                "décision binaire. Voir les motifs ci-dessous.",
            ))

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
            title=t("SHAP contribution — Top 15 features",
                    "Contribution SHAP — Top 15 des variables"),
            xaxis_title=t("SHAP value (impact on log-odds of default)",
                          "Valeur SHAP (impact sur le log-odds de défaut)"),
            height=500,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=80, t=60, b=40),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(t(
            "Red bars push the prediction towards default. Green bars push it "
            "away. Bar length = magnitude of impact on the model output.",
            "Les barres rouges poussent la prédiction vers le défaut, les vertes "
            "l'en éloignent. La longueur = ampleur de l'impact sur la sortie du modèle.",
        ))

        # --- Reason codes (GDPR Art. 22 plain-language explanation) -------
        st.markdown(t("### Reason codes", "### Motifs de décision"))
        reasons = top_negative_reasons(waterfall, top_n=3)
        block = format_adverse_action_block(reasons)
        st.code(block, language="text")
        st.caption(t(
            "Auto-generated from the top-3 positive SHAP contributors. "
            "Copy-paste into an adverse-action notice. See docs/REGULATORY_MAPPING.md.",
            "Généré automatiquement à partir des 3 principaux contributeurs SHAP "
            "positifs. À reprendre dans une notification de refus. "
            "Voir docs/REGULATORY_MAPPING.md. (Texte du modèle en anglais.)",
        ))

        # --- Counterfactual ----------------------------------------------
        st.markdown(t("### Counterfactual explanation", "### Explication contrefactuelle"))
        st.caption(t(
            "What single change in the applicant's profile would bring the "
            "PD below 15%?",
            "Quel changement unique du profil ferait passer la PD sous 15% ?",
        ))
        cf = find_counterfactual(model, raw_row, feature_names, threshold=0.15)
        if cf is None:
            if pd_value < 0.15:
                st.success(t(
                    "This applicant is already below the 15% risk threshold; "
                    "no counterfactual needed.",
                    "Ce dossier est déjà sous le seuil de risque de 15% ; "
                    "aucun contrefactuel nécessaire.",
                ))
            else:
                st.info(t(
                    "No single-feature change in the actionable feature set "
                    "would push this applicant below 15%. A 2-feature search "
                    "or policy override would be needed.",
                    "Aucun changement d'une seule variable actionnable ne fait "
                    "passer ce dossier sous 15%. Il faudrait une recherche à "
                    "2 variables ou une dérogation.",
                ))
        else:
            st.success(t(
                f"**{cf.description}** → predicted PD drops from "
                f"{pd_value:.1%} to {cf.new_pd:.1%}.",
                f"**{cf.description}** → la PD prédite passe de "
                f"{pd_value:.1%} à {cf.new_pd:.1%}.",
            ))


# --------------------------------------------------------------------------
# Tab 2 — Portfolio Analysis
# --------------------------------------------------------------------------

with tab_portfolio:
    st.header(t("Portfolio analysis", "Analyse de portefeuille"))
    st.caption(t(
        "Scores a public sample of 1,000 contracts drawn from the hold-out "
        "test set.",
        "Score un échantillon public de 1 000 contrats issus du jeu de test "
        "mis de côté.",
    ))

    scored = score_sample("xgb-v1", tuple(feature_names))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Portfolio size", "Taille du portefeuille"),
              f"{len(scored):,} " + t("contracts", "contrats"))
    c2.metric(t("Average PD", "PD moyenne"), f"{scored['PD'].mean():.1%}")
    high_n = int((scored["PD"] > 0.15).sum())
    c3.metric(
        t("High risk (PD>15%)", "Risque élevé (PD>15%)"),
        f"{high_n:,}",
        f"{(scored['PD'] > 0.15).mean():.1%}",
    )
    c4.metric(t("Median credit limit", "Plafond de crédit médian"),
              f"NT${scored['LIMIT_BAL'].median():,.0f}")

    st.subheader(t("PD distribution", "Distribution des PD"))
    fig = px.histogram(
        scored,
        x="PD",
        nbins=50,
        color="RISK_BAND",
        color_discrete_map=BAND_COLOURS,
        labels={"PD": t("Probability of default", "Probabilité de défaut"),
                "count": t("Number of contracts", "Nombre de contrats")},
        title=t("PD distribution across the portfolio",
                "Distribution des PD sur le portefeuille"),
    )
    fig.add_vline(x=0.05, line_dash="dash", line_color="gray",
                  annotation_text="LOW / MEDIUM")
    fig.add_vline(x=0.15, line_dash="dash", line_color="gray",
                  annotation_text="MEDIUM / HIGH")
    st.plotly_chart(fig, width="stretch")

    st.subheader(t("PD vs credit limit", "PD vs plafond de crédit"))
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
        title=t("PD vs credit limit (bubble size = most recent bill amount)",
                "PD vs plafond (taille = dernière facture)"),
        labels={"LIMIT_BAL": t("Credit limit (NT$)", "Plafond de crédit (NT$)"),
                "PD": t("Probability of default", "Probabilité de défaut")},
        opacity=0.65,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader(t("Default rate by current repayment status (PAY_0)",
                   "Taux de défaut par statut de remboursement courant (PAY_0)"))
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
        title=t("Average PD by current month's repayment status",
                "PD moyenne par statut de remboursement du mois courant"),
        labels={"Avg_PD": t("Average PD", "PD moyenne"),
                "PAY_0": t("Repayment status (current month)",
                           "Statut de remboursement (mois courant)")},
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader(t("Global SHAP feature importance (sample of 200)",
                   "Importance globale SHAP (échantillon de 200)"))
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
        title=t("Global feature importance — top 20 (mean |SHAP|)",
                "Importance globale des variables — top 20 (|SHAP| moyen)"),
        labels={"MEAN_ABS_SHAP": t("Mean |SHAP value|", "|Valeur SHAP| moyenne"),
                "FEATURE": ""},
        color="MEAN_ABS_SHAP",
        color_continuous_scale=["#C8E6FA", "#0070C0"],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Tab 3 — Model Performance
# --------------------------------------------------------------------------

with tab_model:
    st.header(t("Model performance", "Performance du modèle"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c2.metric("Gini", f"{metrics['gini']:.4f}")
    c3.metric("KS statistic", f"{metrics['ks_statistic']:.4f}")
    c4.metric(
        "Brier score",
        f"{metrics['brier_score']:.4f}",
        help=t("Lower is better. Perfect model = 0.",
               "Plus bas = mieux. Modèle parfait = 0."),
    )

    scored = score_sample("xgb-v1", tuple(feature_names))
    y_true = scored["IS_DEFAULT_TRUE"].astype(int).values
    y_proba = scored["PD"].values

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("ROC curve", "Courbe ROC"))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name="XGBoost", line=dict(color="#0070C0", width=2))
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name=t("Random", "Aléatoire"),
                       line=dict(color="gray", dash="dash"))
        )
        fig.update_layout(
            title=t(f"ROC — sample (AUC≈{metrics['roc_auc']:.3f} on full test)",
                    f"ROC — échantillon (AUC≈{metrics['roc_auc']:.3f} sur test complet)"),
            xaxis_title=t("False positive rate", "Taux de faux positifs"),
            yaxis_title=t("True positive rate", "Taux de vrais positifs"),
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.subheader(t("Precision-Recall curve", "Courbe précision-rappel"))
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
            annotation_text=t(f"Baseline ({metrics['default_rate_test']:.1%})",
                              f"Référence ({metrics['default_rate_test']:.1%})"),
        )
        fig.update_layout(
            title=t(f"PR — sample (PR-AUC≈{metrics['pr_auc']:.3f} on full test)",
                    f"PR — échantillon (PR-AUC≈{metrics['pr_auc']:.3f} sur test complet)"),
            xaxis_title=t("Recall", "Rappel"),
            yaxis_title=t("Precision", "Précision"),
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader(t("Score distribution by actual outcome",
                   "Distribution des scores par résultat réel"))
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=y_proba[y_true == 0],
            name=t("Non-default", "Sans défaut"),
            opacity=0.7,
            marker_color="#00B050",
            nbinsx=50,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=y_proba[y_true == 1],
            name=t("Default", "Défaut"),
            opacity=0.7,
            marker_color="#FF4444",
            nbinsx=50,
        )
    )
    fig.update_layout(
        barmode="overlay",
        title=t(
            f"Predicted PD distribution by actual outcome (KS={metrics['ks_statistic']:.3f})",
            f"Distribution des PD prédites par résultat réel (KS={metrics['ks_statistic']:.3f})",
        ),
        xaxis_title=t("Predicted PD", "PD prédite"),
        yaxis_title=t("Count", "Effectif"),
        height=400,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader(t("Cost-sensitive evaluation", "Évaluation sensible au coût"))
    st.caption(t(
        "Translate the confusion matrix into a euro-denominated P&L. "
        "Move the sliders to reflect your portfolio economics; the threshold "
        "auto-tunes to maximise net P&L.",
        "Traduit la matrice de confusion en P&L en euros. Ajustez les curseurs "
        "selon l'économie de votre portefeuille ; le seuil s'optimise pour "
        "maximiser le P&L net.",
    ))
    cc1, cc2, cc3 = st.columns(3)
    margin = cc1.number_input(
        t("Margin per correctly approved good (€)", "Marge par bon accepté (€)"),
        0, 1000, 120, 10,
        help=t("Net interest margin on a performing loan",
               "Marge nette d'intérêt sur un prêt performant"),
    )
    fn_cost = cc2.number_input(
        t("Cost of a false negative (€)", "Coût d'un faux négatif (€)"),
        0, 20000, 3000, 100,
        help=t("Unpaid exposure × LGD on an accepted defaulter",
               "Exposition impayée × LGD sur un défaillant accepté"),
    )
    fp_cost = cc3.number_input(
        t("Cost of a false positive (€)", "Coût d'un faux positif (€)"),
        0, 1000, 120, 10,
        help=t("Foregone margin on a rejected non-defaulter",
               "Marge perdue sur un bon client refusé"),
    )
    costs = CostInputs(margin_per_tn=margin, cost_per_fn=fn_cost, cost_per_fp=fp_cost)
    sweep = sweep_thresholds(y_true, y_proba, costs, n_steps=101)
    best_t = float(sweep.loc[sweep["net"].idxmax(), "threshold"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["threshold"], y=sweep["net"], mode="lines",
        line=dict(color="#0070C0", width=2), name="Net P&L",
    ))
    fig.add_vline(
        x=best_t, line_dash="dash", line_color="#FF4444",
        annotation_text=t(f"Optimum @ {best_t:.2f}", f"Optimum @ {best_t:.2f}"),
    )
    fig.update_layout(
        title=t("Portfolio net P&L vs. decision threshold",
                "P&L net du portefeuille vs seuil de décision"),
        xaxis_title=t("Decision threshold (PD ≥ this → reject)",
                      "Seuil de décision (PD ≥ ce seuil → refus)"),
        yaxis_title=t("Net P&L on the 1,000-row sample (€)",
                      "P&L net sur l'échantillon de 1 000 (€)"),
        height=400,
    )
    st.plotly_chart(fig, width="stretch")

    conf = confusion_at_threshold(y_true, y_proba, best_t)
    pnl = portfolio_pnl(conf, costs)
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric(t("True positives (correctly rejected)",
                 "Vrais positifs (bien refusés)"), f"{conf['tp']}")
    cm2.metric(t("False negatives (accepted defaulters)",
                 "Faux négatifs (défaillants acceptés)"), f"{conf['fn']}",
               delta=f"-€{pnl['fn_loss']:,.0f}", delta_color="inverse")
    cm3.metric(t("False positives (rejected goods)",
                 "Faux positifs (bons refusés)"), f"{conf['fp']}",
               delta=f"-€{pnl['fp_loss']:,.0f}", delta_color="inverse")
    cm4.metric(t("Net P&L on the sample", "P&L net sur l'échantillon"),
               f"€{pnl['net']:,.0f}")

    st.subheader(t("Full metrics", "Métriques complètes"))
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
    st.header(t("Methodology", "Méthodologie"))
    st.caption(t(
        "Why this project matters, the maths behind the model and the "
        "explainability layer, and the engineering choices made along the way.",
        "Pourquoi ce projet, les mathématiques derrière le modèle et la couche "
        "d'explicabilité, et les choix d'ingénierie faits en chemin. "
        "(Les démonstrations mathématiques restent en anglais.)",
    ))

    st.subheader(t("Why this project", "Pourquoi ce projet"))
    st.markdown(t(
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
        """,
        """
        Le credit scoring se situe à l'intersection de trois contraintes
        généralement en tension :

        - **Performance prédictive** — le modèle doit classer les emprunteurs
          assez bien pour réduire sensiblement la perte attendue face à une
          politique de référence.
        - **Explicabilité** — sous SR 11-7 (Federal Reserve) et les orientations
          EBA sur les modèles internes, une banque doit pouvoir justifier chaque
          décision de crédit individuelle et démontrer une gouvernance continue.
          Un scorecard boîte noire n'est pas acceptable.
        - **Simplicité opérationnelle** — le modèle est recalculé sur chaque
          contrat, parfois des milliers de fois par seconde ; la latence
          d'inférence et le coût d'infrastructure comptent.

        Cette app démontre une stack qui résout les trois : un ensemble d'arbres
        boostés (XGBoost) pour la performance, SHAP pour des explications par
        décision fondées sur la théorie des jeux coopératifs, et un déploiement
        Streamlit mono-fichier qui passe à l'échelle du trafic public à coût nul.
        """,
    ))

    st.subheader(t("The credit scoring problem", "Le problème du credit scoring"))
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

    st.subheader(t("Why XGBoost over a logistic regression scorecard",
                   "Pourquoi XGBoost plutôt qu'un scorecard logistique"))
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

        The class imbalance (≈22% default rate) is left untouched: the model
        trains on the natural distribution with no `scale_pos_weight` and no
        resampling, so the predicted PDs come out calibrated as long-run default
        frequencies (test ECE ≈ 0.01). Re-weighting the positive class was
        benchmarked and rejected — it inflated every PD to ~2x the base rate
        without improving ranking.
        """
    )

    st.subheader(t("SHAP — the maths of the explanations",
                   "SHAP — les mathématiques des explications"))
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

    st.subheader(t("Metrics, in plain language", "Les métriques, en clair"))
    st.markdown(t(
        "Credit scoring uses a small set of standard metrics that capture "
        "different aspects of model quality:",
        "Le credit scoring s'appuie sur un petit jeu de métriques standard qui "
        "capturent différents aspects de la qualité du modèle :",
    ))
    # Cells are precomputed as variables: f-string expressions cannot contain
    # backslashes (the Gini row carries LaTeX) on Python < 3.12.
    _mcol = t("What it measures", "Ce qu'elle mesure")
    _mmod = t("This model", "Ce modèle")
    _d_auc = t("Probability that a random defaulter is ranked above a random non-defaulter",
               "Probabilité qu'un défaillant au hasard soit classé au-dessus d'un non-défaillant")
    _gini_math = "$2 \\cdot \\mathrm{AUC} - 1$"
    _d_gini = t(f"{_gini_math} — Lorenz-curve area, standard in retail-credit decks",
                f"{_gini_math} — aire de la courbe de Lorenz, standard en crédit retail")
    _d_ks = t("Maximum vertical distance between the CDFs of defaulters and non-defaulters",
              "Distance verticale maximale entre les CDF des défaillants et non-défaillants")
    _d_pr = t("Average precision — robust to class imbalance",
              "Précision moyenne — robuste au déséquilibre de classes")
    _d_brier = t("Mean squared error of the probability output; 0 = perfect, lower is better",
                 "Erreur quadratique moyenne de la probabilité ; 0 = parfait, plus bas = mieux")
    _d_ll = t("Negative log-likelihood; the loss the model is actually trained on",
              "Log-vraisemblance négative ; la perte réellement optimisée")
    _bench = t(
        "Industry rule-of-thumb benchmarks for unsecured consumer credit: "
        "ROC-AUC ≥ 0.72, Gini ≥ 0.44, KS ≥ 0.35. This model clears all three "
        "on the hold-out test set.",
        "Repères usuels du crédit à la consommation non garanti : ROC-AUC ≥ 0.72, "
        "Gini ≥ 0.44, KS ≥ 0.35. Ce modèle dépasse les trois sur le jeu de test.",
    )
    st.markdown(
        f"""
| Metric | {_mcol} | {_mmod} |
|---|---|---|
| **ROC-AUC**  | {_d_auc}   | {metrics['roc_auc']:.3f} |
| **Gini**     | {_d_gini}  | {metrics['gini']:.3f} |
| **KS**       | {_d_ks}    | {metrics['ks_statistic']:.3f} |
| **PR-AUC**   | {_d_pr}    | {metrics['pr_auc']:.3f} |
| **Brier**    | {_d_brier} | {metrics['brier_score']:.3f} |
| **Log-loss** | {_d_ll}    | {metrics['log_loss']:.3f} |

{_bench}
        """
    )

    st.subheader(t("Technical choices, justified", "Choix techniques, justifiés"))
    _c_choice = t("Choice", "Choix")
    _c_alt = t("Alternative", "Alternative")
    _c_why = t("Why this one", "Pourquoi celui-ci")
    st.markdown(
        f"""
        | {_c_choice} | {_c_alt} | {_c_why} |
        |---|---|---|
        | XGBoost (CPU, `tree_method=hist`) | LightGBM, CatBoost | {t("Mature, single-file model serialisation, deterministic, well-supported by SHAP, fits on a free Streamlit instance.", "Mature, sérialisation mono-fichier, déterministe, bien supporté par SHAP, tient sur une instance Streamlit gratuite.")} |
        | {t("Stratified 60/20/20 train/val/test split", "Split stratifié 60/20/20 train/val/test")} | {t("Temporal split", "Split temporel")} | {t("The UCI dataset only contains 6 months of payment data and no contract issue date, so a meaningful temporal split is not available.", "Le dataset UCI ne contient que 6 mois de paiements et aucune date d'octroi, donc un split temporel pertinent n'est pas possible.")} |
        | {t("Early stopping and threshold selected on validation", "Early stopping et seuil choisis sur la validation")} | {t("Selecting on test", "Sélection sur le test")} | {t("Keeps the test set genuinely unseen, so the reported metrics are an unbiased out-of-sample estimate.", "Garde le test réellement inconnu, donc les métriques rapportées sont une estimation hors échantillon non biaisée.")} |
        | {t("No class re-weighting (natural distribution)", "Pas de repondération de classe (distribution naturelle)")} | `scale_pos_weight`, SMOTE | {t("Keeps the predicted PDs calibrated as long-run frequencies (test ECE ≈ 0.01). Re-weighting was benchmarked and rejected.", "Garde les PD calibrées comme des fréquences long terme (ECE test ≈ 0.01). La repondération a été testée puis rejetée.")} |
        | SHAP `TreeExplainer`, `tree_path_dependent` | `interventional`, KernelSHAP | {t("Exact and fast for tree ensembles; satisfies the local-accuracy property by construction.", "Exact et rapide pour les ensembles d'arbres ; satisfait la propriété de local accuracy par construction.")} |
        | Streamlit Community Cloud | Flask/FastAPI + frontend | {t("Zero infra, one-file deploy, free public URL, the model stays warm across requests.", "Zéro infra, déploiement mono-fichier, URL publique gratuite, le modèle reste chaud entre requêtes.")} |
        | {t("Parquet for the datasets", "Parquet pour les jeux de données")} | CSV | {t("5–10× smaller, preserves dtypes, columnar reads.", "5 à 10× plus petit, préserve les types, lectures colonnaires.")} |
        | {t("Model artefacts committed to git", "Artefacts modèle commités dans git")} | {t("A model registry", "Un registre de modèles")} | {t("The artefact set is small (≈1 MB) and a public repo is the cheapest registry possible.", "Les artefacts sont petits (≈1 Mo) et un repo public est le registre le moins cher possible.")} |
        """
    )

    st.subheader(t("Architecture in one diagram", "L'architecture en un schéma"))
    st.code(
        """
        data/raw/uci_credit_card.csv  (30,000 rows, raw)
                       │
                       ▼
          src/data_prep.py
            • clean categoricals (EDUCATION → {1..4}, MARRIAGE → {1..3})
            • engineer 8 features (PAY_MEAN, DELINQ_COUNT, UTILIZATION, …)
            • one-hot encode SEX / EDUCATION / MARRIAGE (drop_first)
            • stratified 60/20/20 train/val/test split
                       │
                       ▼
          data/processed/{train,val,test}.parquet + feature_names.json
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

    st.subheader(t("What I would do next in a production setting",
                   "Ce que je ferais ensuite en production"))
    st.markdown(t(
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
        """,
        """
        - **Suivi du Population Stability Index (PSI)** sur la distribution des
          entrées, avec alertes automatiques dès qu'une variable dérive au-delà
          d'un seuil de gouvernance.
        - **Calibration** par régression isotonique ou Platt scaling, avec un
          diagramme de fiabilité dans cet onglet.
        - **Reject-inference** — les données d'entraînement ne contiennent que
          des contrats acceptés ; un modèle en production doit corriger ce biais
          de sélection.
        - **Explications contrefactuelles** en complément des attributions SHAP :
          « quel est le plus petit changement qui inverserait la décision ? »,
          plus proche de ce dont un explicateur client a réellement besoin.
        - **Registre de modèles + CI** — aujourd'hui les artefacts sont dans git ;
          en production je pousserais vers MLflow / SageMaker / Vertex et
          versionnerais le modèle et le hash des données d'entraînement.
        - **A/B-tester le seuil** en production contre la politique existante sur
          une fraction du trafic avant un déploiement large.
        """,
    ))

    st.subheader(t("References", "Références"))
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
# Tab 5 — Governance (business case, calibration, fairness, monitoring)
# --------------------------------------------------------------------------

with tab_governance:
    st.header(t("Governance & model risk", "Gouvernance & risque modèle"))
    st.caption(t(
        "The artefacts a model validator, an internal auditor, or a "
        "regulator would expect to see alongside the model itself. "
        "Every panel is backed by a markdown document in `docs/`.",
        "Les artefacts qu'un validateur de modèle, un auditeur interne ou un "
        "régulateur attendrait à côté du modèle. Chaque panneau s'appuie sur un "
        "document markdown dans `docs/`.",
    ))

    _gov_labels = {
        "Business case": t("Business case", "Analyse de rentabilité"),
        "Decision policy": t("Decision policy", "Politique de décision"),
        "Calibration": t("Calibration", "Calibration"),
        "Fairness audit": t("Fairness audit", "Audit d'équité"),
        "Cost-sensitive thresholding": t("Cost-sensitive thresholding",
                                         "Seuil sensible au coût"),
        "Monitoring (mock-up)": t("Monitoring (mock-up)", "Monitoring (maquette)"),
        "Documents": t("Documents", "Documents"),
    }
    gov_section = st.radio(
        "Section",
        list(_gov_labels.keys()),
        format_func=lambda k: _gov_labels[k],
        horizontal=True,
    )

    scored = score_sample("xgb-v1", tuple(feature_names))
    y_true_sample = scored["IS_DEFAULT_TRUE"].astype(int).values
    y_proba_sample = scored["PD"].values

    if gov_section == "Business case":
        st.subheader(t("€ impact per 100k-contract annual book",
                       "Impact € sur un encours annuel de 100k contrats"))
        st.markdown(t(
            """
            Replacing a baseline logistic scorecard (Gini ≈ 0.50) with the
            XGBoost model (Gini = 0.558) on a portfolio of **100,000 active
            contracts** with an **average exposure of €5,000** is expected to
            reduce annual credit losses by **€1.0 – €1.4 million** at the
            same approval rate, with a payback period of **under 6 months**
            against a build cost of approximately **€180 k**.
            """,
            """
            Remplacer un scorecard logistique de référence (Gini ≈ 0.50) par le
            modèle XGBoost (Gini = 0.558) sur un portefeuille de **100 000 contrats
            actifs** avec une **exposition moyenne de 5 000 €** devrait réduire les
            pertes de crédit annuelles de **1,0 à 1,4 M€** à taux d'acceptation
            constant, avec un retour sur investissement en **moins de 6 mois** pour
            un coût de construction d'environ **180 k€**.
            """,
        ))
        bc = pd.DataFrame(
            {
                t("Scenario", "Scénario"): [
                    t("Pessimistic (-30%)", "Pessimiste (-30%)"),
                    t("Base case", "Cas de base"),
                    t("Optimistic (+30%)", "Optimiste (+30%)"),
                ],
                t("Annual loss reduction", "Réduction de perte annuelle"):
                    ["€700,000", "€1,000,000", "€1,400,000"],
                t("Net annual P&L", "P&L annuel net"):
                    ["€631,000", "€931,000", "€1,331,000"],
                t("Payback", "Retour sur invest."):
                    [t("≈ 4 months", "≈ 4 mois"), t("≈ 2.3 months", "≈ 2,3 mois"),
                     t("≈ 1.6 months", "≈ 1,6 mois")],
            }
        ).set_index(t("Scenario", "Scénario"))
        st.dataframe(bc, width="stretch")
        st.caption(t(
            "Full assumptions and sensitivities in `docs/BUSINESS_CASE.md`.",
            "Hypothèses et sensibilités complètes dans `docs/BUSINESS_CASE.md`.",
        ))

    elif gov_section == "Decision policy":
        st.subheader(t("Three-tier decision policy", "Politique de décision à 3 niveaux"))
        st.markdown(t(
            f"""
            A single hard threshold forces a binary accept/reject on every
            applicant, including the large middle band where the model is
            genuinely uncertain — that is what inflates the false-positive count.
            Origination instead auto-decides only the confident tails and sends
            the grey zone to a human:

            - **Auto-approve** when PD < **{policy.approve_below:.0%}**
            - **Manual review** when **{policy.approve_below:.0%} ≤ PD < {policy.decline_at_or_above:.0%}**
            - **Auto-decline** when PD ≥ **{policy.decline_at_or_above:.0%}**

            Both cut-offs are fit on the validation split from risk targets
            (approved band ≤ 8% default, declined band ≥ 60%), never on test.
            """,
            f"""
            Un seuil unique force une décision binaire accepter/refuser sur chaque
            dossier, y compris la large bande centrale où le modèle est réellement
            incertain — c'est ce qui gonfle le nombre de faux positifs. L'octroi
            décide automatiquement seulement les extrêmes sûrs et envoie la zone
            grise à un humain :

            - **Acceptation auto** si PD < **{policy.approve_below:.0%}**
            - **Étude manuelle** si **{policy.approve_below:.0%} ≤ PD < {policy.decline_at_or_above:.0%}**
            - **Refus auto** si PD ≥ **{policy.decline_at_or_above:.0%}**

            Les deux seuils sont calés sur la validation à partir de cibles de
            risque (bande acceptée ≤ 8% de défaut, refusée ≥ 60%), jamais sur le test.
            """,
        ))

        _band_lbl = {APPROVE: t("Auto-approve", "Acceptation auto"),
                     REVIEW: t("Manual review", "Étude manuelle"),
                     DECLINE: t("Auto-decline", "Refus auto")}
        table = evaluate_policy(y_true_sample, y_proba_sample, policy)
        disp = table.copy()
        disp["decision"] = disp["decision"].map(_band_lbl)
        disp["share"] = (disp["share"] * 100).map("{:.1f}%".format)
        disp["default_rate"] = (disp["default_rate"] * 100).map("{:.1f}%".format)
        disp = disp.rename(
            columns={
                "decision": t("Decision", "Décision"),
                "n": t("Contracts", "Contrats"),
                "share": t("Share of book", "Part de l'encours"),
                "default_rate": t("Actual default rate", "Taux de défaut réel"),
            }
        ).set_index(t("Decision", "Décision"))
        st.dataframe(disp, width="stretch")

        colours = {APPROVE: "#00B050", REVIEW: "#FFC000", DECLINE: "#FF4444"}
        fig = go.Figure(
            go.Bar(
                x=table["default_rate"] * 100,
                y=[_band_lbl[d] for d in table["decision"]],
                orientation="h",
                marker_color=[colours[d] for d in table["decision"]],
                text=[f"{r*100:.1f}%" for r in table["default_rate"]],
                textposition="outside",
            )
        )
        fig.add_vline(
            x=y_true_sample.mean() * 100,
            line=dict(color="gray", dash="dash"),
            annotation_text=t(f"portfolio {y_true_sample.mean():.1%}",
                              f"portefeuille {y_true_sample.mean():.1%}"),
        )
        fig.update_layout(
            title=t("Actual default rate by decision band (public sample)",
                    "Taux de défaut réel par bande de décision (échantillon public)"),
            xaxis_title=t("Default rate (%)", "Taux de défaut (%)"),
            height=280,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=60, t=50, b=40),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(t(
            "The auto-decided tails are far from the portfolio average; the "
            "review band is where the model is honestly uncertain. This is why a "
            "raw false-positive count at one threshold is the wrong lens on the "
            "model.",
            "Les extrêmes auto-décidés sont loin de la moyenne du portefeuille ; la "
            "bande de revue est là où le modèle est honnêtement incertain. C'est "
            "pourquoi un décompte brut de faux positifs à un seuil unique est le "
            "mauvais angle de lecture.",
        ))

    elif gov_section == "Calibration":
        st.subheader(t("Reliability diagram", "Diagramme de fiabilité"))
        st.caption(t(
            "Predicted PDs (quantile-binned) against observed default rate. "
            "A perfectly calibrated model lies on the y=x diagonal. This model "
            "trains on the natural class distribution (no `scale_pos_weight`), "
            "so it is calibrated by construction and sits close to the diagonal.",
            "PD prédites (regroupées par quantiles) vs taux de défaut observé. Un "
            "modèle parfaitement calibré est sur la diagonale y=x. Ce modèle "
            "s'entraîne sur la distribution naturelle (sans `scale_pos_weight`), "
            "donc il est calibré par construction et colle à la diagonale.",
        ))
        result = assess_calibration(y_true_sample, y_proba_sample, n_bins=10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash"),
            name=t("Perfect calibration", "Calibration parfaite"),
        ))
        fig.add_trace(go.Scatter(
            x=result.bins["mean_predicted"],
            y=result.bins["observed_rate"],
            mode="markers+lines",
            marker=dict(size=10, color="#0070C0"),
            name=t("Model", "Modèle"),
            text=[f"n={n}" for n in result.bins["n"]],
        ))

        if st.checkbox(t("Overlay post-hoc isotonic (diagnostic)",
                         "Superposer l'isotonic post-hoc (diagnostic)"), value=False):
            iso = fit_isotonic(y_true_sample, y_proba_sample)
            y_recal = apply_calibrator(iso, y_proba_sample)
            result_recal = assess_calibration(y_true_sample, y_recal, n_bins=10)
            fig.add_trace(go.Scatter(
                x=result_recal.bins["mean_predicted"],
                y=result_recal.bins["observed_rate"],
                mode="markers+lines",
                marker=dict(size=10, color="#00B050"),
                name="+ isotonic",
            ))
            st.caption(t(
                "Isotonic is fit here for illustration only; on validation it did "
                "not improve calibration over the native model, so it is not part "
                "of the shipped pipeline.",
                "L'isotonic est ajusté ici à titre d'illustration ; sur la validation "
                "il n'a pas amélioré la calibration du modèle natif, donc il ne fait "
                "pas partie du pipeline livré.",
            ))

        fig.update_layout(
            xaxis=dict(title=t("Predicted PD (bin mean)", "PD prédite (moyenne du bin)"),
                       range=[0, 1]),
            yaxis=dict(title=t("Observed default rate", "Taux de défaut observé"),
                       range=[0, 1]),
            height=450,
        )
        st.plotly_chart(fig, width="stretch")

        ece = expected_calibration_error(y_true_sample, y_proba_sample, n_bins=10)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ECE ↓", f"{ece:.4f}")
        c2.metric("Brier score", f"{result.brier:.4f}")
        c3.metric("Reliability ↓", f"{result.reliability:.4f}")
        c4.metric("Resolution ↑", f"{result.resolution:.4f}")
        c5.metric(t("Uncertainty", "Incertitude"), f"{result.uncertainty:.4f}")
        st.caption(t(
            "Murphy decomposition: Brier = Reliability − Resolution + Uncertainty. "
            "Lower reliability ⇒ better calibrated. Higher resolution ⇒ "
            "better discrimination. See `src/calibration.py`.",
            "Décomposition de Murphy : Brier = Reliability − Resolution + Uncertainty. "
            "Reliability plus basse ⇒ mieux calibré. Resolution plus haute ⇒ "
            "meilleure discrimination. Voir `src/calibration.py`.",
        ))

    elif gov_section == "Fairness audit":
        st.subheader(t("Disparate-impact ratios & per-group AUC",
                       "Ratios d'impact disparate & AUC par groupe"))
        st.caption(t(
            "Audit on SEX, AGE band, and EDUCATION. Disparate-impact (DI) "
            "ratio is the selection rate vs. the largest reference group; "
            "the four-fifths rule considers DI ∈ [0.80, 1.25] acceptable. "
            "Audit threshold is the trained model's Youden-J cut-off.",
            "Audit sur SEXE, tranche d'ÂGE et ÉDUCATION. Le ratio d'impact disparate "
            "(DI) est le taux de sélection vs le plus grand groupe de référence ; "
            "la règle des quatre cinquièmes considère DI ∈ [0.80, 1.25] acceptable. "
            "Le seuil d'audit est le seuil Youden-J du modèle.",
        ))

        scored_features = scored.rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
        feat_full = build_feature_matrix(scored_features)
        audit_df = feat_full[["SEX_MALE", "AGE"]].copy()
        audit_df["EDUCATION"] = scored_features["EDUCATION"].values

        threshold = float(metrics.get("threshold_optimal", 0.5))
        summary = fairness_summary(audit_df, y_true_sample, y_proba_sample, threshold)

        for attr, sub in summary.items():
            st.markdown(f"**{attr}**")
            display = sub[
                [
                    "group", "n", "base_rate", "selection_rate",
                    "tpr", "fpr", "auc", "di_ratio", "eod", "reference",
                ]
            ].copy()
            for col in ["base_rate", "selection_rate", "tpr", "fpr", "auc", "di_ratio", "eod"]:
                display[col] = display[col].map(
                    lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                )
            st.dataframe(display.set_index("group"), width="stretch")

        st.caption(t(
            "Audit code in `src/fairness.py`. For an EU production deployment, "
            "`SEX_MALE` would be removed from the feature set and the audit "
            "re-run against correlated proxies.",
            "Code d'audit dans `src/fairness.py`. Pour un déploiement en production "
            "UE, `SEX_MALE` serait retiré des variables et l'audit relancé contre "
            "les proxys corrélés.",
        ))

    elif gov_section == "Cost-sensitive thresholding":
        st.subheader(t("Cost of the decision policy vs. a single threshold",
                       "Coût de la politique de décision vs un seuil unique"))
        st.markdown(t(
            "Puts the three-tier policy on a euro footing. Set your portfolio "
            "economics; the panel compares approving everyone, the best single "
            "hard threshold, and the three-tier policy — and shows how much of "
            "the policy's value depends on the manual-review layer.",
            "Met la politique à 3 niveaux sur une base en euros. Réglez l'économie "
            "de votre portefeuille ; le panneau compare accepter tout le monde, le "
            "meilleur seuil unique et la politique à 3 niveaux — et montre à quel "
            "point la valeur de la politique dépend de la couche de revue manuelle.",
        ))
        e1, e2, e3, e4 = st.columns(4)
        margin = e1.number_input(t("Margin / good (€)", "Marge / bon (€)"), 0, 1000, 120, 10)
        fn_cost = e2.number_input(t("False negative (€)", "Faux négatif (€)"), 0, 20000, 3000, 100)
        fp_cost = e3.number_input(t("False positive (€)", "Faux positif (€)"), 0, 2000, 120, 10)
        review_cost = e4.number_input(
            t("Manual review / case (€)", "Revue manuelle / dossier (€)"), 0, 1000, 50, 10)
        costs = CostInputs(
            margin_per_tn=margin,
            cost_per_fn=fn_cost,
            cost_per_fp=fp_cost,
            cost_per_review=review_cost,
        )
        review_eff = st.slider(
            t("Reviewer effectiveness (fraction of grey-zone defaulters caught)",
              "Efficacité du reviewer (fraction des défaillants de la zone grise détectés)"),
            0.0, 1.0, 0.8, 0.05,
        )

        be = breakeven_pd(costs)
        st.info(t(
            f"**Break-even PD = {be:.1%}** — below it accepting is profitable, "
            f"above it declining is. The policy's approve cut-off is "
            f"{policy.approve_below:.0%} and its decline cut-off is "
            f"{policy.decline_at_or_above:.0%}: the review band brackets the "
            "break-even point, which is exactly where a human call earns its cost.",
            f"**PD de break-even = {be:.1%}** — en dessous accepter est rentable, "
            f"au-dessus refuser l'est. Le seuil d'acceptation de la politique est "
            f"{policy.approve_below:.0%} et son seuil de refus {policy.decline_at_or_above:.0%} : "
            "la bande de revue encadre le point de break-even, exactement là où une "
            "décision humaine rentabilise son coût.",
        ))

        y_s, p_s = y_true_sample, y_proba_sample
        approve_all = portfolio_pnl(
            {"tp": 0, "fp": 0, "tn": int((y_s == 0).sum()), "fn": int((y_s == 1).sum())},
            costs,
        )["net"]
        best_t = optimal_threshold(y_s, p_s, costs, n_steps=201)
        binary = portfolio_pnl(confusion_at_threshold(y_s, p_s, best_t), costs)["net"]
        pol_pnl = policy_pnl(
            y_s, p_s, policy.approve_below, policy.decline_at_or_above, costs, review_eff
        )

        strategies = [
            t("Approve everyone (no model)", "Accepter tout le monde (sans modèle)"),
            t(f"Best single threshold (@ {best_t:.2f})",
              f"Meilleur seuil unique (@ {best_t:.2f})"),
            t(f"Three-tier policy (review eff. {review_eff:.0%})",
              f"Politique 3 niveaux (revue eff. {review_eff:.0%})"),
        ]
        pnl_values = [approve_all, binary, pol_pnl["net"]]
        fig = go.Figure(
            go.Bar(
                x=pnl_values,
                y=strategies,
                orientation="h",
                marker_color=["#999999", "#0070C0", "#00B050"],
                text=[f"€{v:,.0f}" for v in pnl_values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=t("Net P&L on the 1,000-row sample",
                    "P&L net sur l'échantillon de 1 000"),
            xaxis_title=t("€ (higher is better)", "€ (plus haut = mieux)"),
            height=260,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=80, t=50, b=40),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(t(
            f"The manual-review tier handles **{pol_pnl['n_review']} cases** at "
            f"**€{pol_pnl['review_cost']:,.0f}**. Slide reviewer effectiveness to "
            "0 (rubber-stamp) and the policy loses to a plain threshold; above "
            "roughly 0.85 it wins — that break-even is the business case for the "
            "human layer, not a modelling opinion.",
            f"La couche de revue manuelle traite **{pol_pnl['n_review']} dossiers** pour "
            f"**{pol_pnl['review_cost']:,.0f} €**. Mettez l'efficacité du reviewer à "
            "0 (tampon aveugle) et la politique perd face à un simple seuil ; au-dessus "
            "d'environ 0,85 elle gagne — ce break-even est la justification économique "
            "de la couche humaine, pas une opinion de modélisation.",
        ))

    elif gov_section == "Monitoring (mock-up)":
        st.subheader(t("Production monitoring dashboard — mock-up",
                       "Tableau de bord de monitoring production — maquette"))
        st.caption(t(
            "What the model risk team would see daily in production. "
            "Values below are static placeholders for the static MVP; "
            "the design is documented in `docs/MONITORING_PLAN.md`.",
            "Ce que l'équipe risque modèle verrait chaque jour en production. "
            "Les valeurs ci-dessous sont des placeholders statiques pour ce MVP ; "
            "le design est documenté dans `docs/MONITORING_PLAN.md`.",
        ))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("Scores today", "Scores aujourd'hui"), "8,412")
        m2.metric(t("Approval rate", "Taux d'acceptation"), "61.3%")
        m3.metric(t("p95 latency", "Latence p95"), "42 ms")
        m4.metric(t("Availability (30d)", "Disponibilité (30j)"), "100.0%")

        st.markdown(t("**Drift**", "**Dérive**"))
        d1, d2, d3 = st.columns(3)
        d1.metric("Max PSI (PAY_0)", "0.08", "✓")
        d2.metric("Max CSI (PD)", "0.04", "✓")
        d3.metric(t("Features in alert", "Variables en alerte"), "0", "✓")

        st.markdown(t("**Performance (30-day rolling, label-mature)**",
                      "**Performance (30 jours glissants, labels matures)**"))
        p1, p2, p3 = st.columns(3)
        p1.metric("Gini", "0.541", t("vs. 0.40 floor", "vs plancher 0.40"))
        p2.metric("KS", "0.421", t("vs. 0.25 floor", "vs plancher 0.25"))
        p3.metric("Brier", "0.182", t("vs. 0.20 alert", "vs alerte 0.20"))

        st.markdown(t("**Fairness (last month)**", "**Équité (mois dernier)**"))
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("DI(sex)", "1.04", "✓")
        f2.metric("AUC(<30)", "0.74")
        f3.metric("AUC(30-45)", "0.76")
        f4.metric("AUC(>45)", "0.77")

        st.success(t("● No active alerts.", "● Aucune alerte active."))

    elif gov_section == "Documents":
        st.subheader(t("Governance artefacts", "Artefacts de gouvernance"))
        st.caption(t(
            "Every link below points to a markdown document in this repo's "
            "`docs/` folder. These are the artefacts a model risk committee "
            "or a regulator would request.",
            "Chaque ligne pointe vers un document markdown du dossier `docs/`. Ce "
            "sont les artefacts qu'un comité risque modèle ou un régulateur demanderait.",
        ))
        _c_art = t("Artefact", "Artefact")
        _c_sum = t("Summary", "Résumé")
        docs_table = pd.DataFrame(
            [
                ("Business case", "docs/BUSINESS_CASE.md", t("€ impact, payback, sensitivities", "Impact €, ROI, sensibilités")),
                ("Personas & user stories", "docs/PERSONAS_AND_USER_STORIES.md", t("Who uses what, acceptance criteria", "Qui utilise quoi, critères d'acceptation")),
                ("RACI", "docs/RACI.md", t("Stakeholder map, decision rights", "Carte des parties prenantes, droits de décision")),
                ("Process flow", "docs/PROCESS_FLOW.md", t("BPMN — where the model sits", "BPMN — où se situe le modèle")),
                ("Roadmap", "docs/ROADMAP.md", t("Phased delivery MVP → v1 → v2", "Livraison par phases MVP → v1 → v2")),
                ("Regulatory mapping", "docs/REGULATORY_MAPPING.md", "Basel III, GDPR, EU AI Act, SR 11-7"),
                ("Model card", "docs/MODEL_CARD.md", t("Intended use, limitations, citation", "Usage prévu, limites, citation")),
                ("Risk register", "docs/RISK_REGISTER.md", t("Top 20 model risks with mitigations", "Top 20 des risques modèle et mitigations")),
                ("Data dictionary", "docs/DATA_DICTIONARY.md", t("Features, lineage, ownership", "Variables, lignage, propriété")),
                ("Monitoring plan", "docs/MONITORING_PLAN.md", t("PSI / CSI / calibration cadence", "PSI / CSI / cadence de calibration")),
                ("A/B test design", "docs/AB_TEST_DESIGN.md", t("Champion-challenger pre-registration", "Pré-enregistrement champion-challenger")),
                ("Glossary", "docs/GLOSSARY.md", t("Plain-language definitions", "Définitions en langage clair")),
            ],
            columns=[_c_art, "Path", _c_sum],
        ).set_index(_c_art)
        st.dataframe(docs_table, width="stretch")


# --------------------------------------------------------------------------
# Tab 6 — About
# --------------------------------------------------------------------------

with tab_about:
    st.header(t("Model documentation", "Documentation du modèle"))

    with st.expander(t("Model overview", "Vue d'ensemble du modèle"), expanded=True):
        st.markdown(t(
            f"""
            **Algorithm**: XGBoost (eXtreme Gradient Boosting)
            **Training data**: UCI Default of Credit Card Clients (Taiwan, Apr–Sep 2005)
            **Split**: stratified 60 / 20 / 20 train / val / test (random, seed = 42)
            **Target**: binary default indicator (1 = default next month)
            **Features**: {metrics['n_features']} after engineering and encoding
            **Explainability**: SHAP `TreeExplainer` (Lundberg & Lee, 2017)
            """,
            f"""
            **Algorithme** : XGBoost (eXtreme Gradient Boosting)
            **Données** : UCI Default of Credit Card Clients (Taïwan, avr.–sept. 2005)
            **Split** : stratifié 60 / 20 / 20 train / val / test (aléatoire, seed = 42)
            **Cible** : indicateur binaire de défaut (1 = défaut le mois suivant)
            **Variables** : {metrics['n_features']} après feature engineering et encodage
            **Explicabilité** : SHAP `TreeExplainer` (Lundberg & Lee, 2017)
            """,
        ))

    with st.expander(t("Why SHAP?", "Pourquoi SHAP ?")):
        st.markdown(t(
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
            """,
            """
            SHAP (SHapley Additive exPlanations) décompose chaque prédiction en
            la contribution de chaque variable, fondée sur la théorie des jeux
            coopératifs (valeurs de Shapley). Propriétés clés :

            - **Local accuracy** — les valeurs SHAP somment à la sortie du modèle.
            - **Consistency** — si la contribution d'une variable augmente, sa
              valeur SHAP augmente.
            - **Missingness** — une variable sans effet a une valeur SHAP de 0.

            C'est le cadre d'explicabilité standard référencé dans les
            orientations de gouvernance (SR 11-7, orientations EBA modèles internes).
            """,
        ))

    with st.expander(t("Model limitations", "Limites du modèle")):
        st.markdown(t(
            """
            - Training data is Taiwanese credit-card payments from 2005, not
              EU consumer lending — recalibration is required before any
              operational use.
            - Six months of payment history is short for a behavioural model;
              `PAY_0` and `PAY_MAX` dominate the feature importance.
            - The target is "default next month", so the model is point-in-time
              and does not account for macroeconomic cycles.
            - Class imbalance is left untouched (no `scale_pos_weight`, no
              resampling) so the PDs are calibrated as long-run frequencies
              (test ECE ≈ 0.01); the trade-off is that rare-default recall at a
              fixed threshold relies on the operating point chosen on validation.
            """,
            """
            - Les données sont des paiements de cartes taïwanaises de 2005, pas du
              crédit conso UE — une recalibration est requise avant tout usage
              opérationnel.
            - Six mois d'historique de paiement, c'est court pour un modèle
              comportemental ; `PAY_0` et `PAY_MAX` dominent l'importance.
            - La cible est « défaut le mois prochain », donc le modèle est
              ponctuel et n'intègre pas les cycles macroéconomiques.
            - Le déséquilibre de classes est laissé tel quel (sans `scale_pos_weight`,
              sans rééchantillonnage) donc les PD sont calibrées comme des fréquences
              long terme (ECE test ≈ 0.01) ; en contrepartie, le rappel des défauts
              rares à seuil fixe dépend du point de fonctionnement choisi en validation.
            """,
        ))

    with st.expander(t("Training details", "Détails d'entraînement")):
        st.json(metrics)
