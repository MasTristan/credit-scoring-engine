"""GDPR Art. 22-style reason codes.

Maps the top-N negative SHAP contributors of a single decision into
plain-language sentences suitable for an adverse-action notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Each entry maps a feature name to a short, customer-facing template.
# ``{value}`` is replaced by the formatted feature value.
TEMPLATES: dict[str, str] = {
    "PAY_0":         "Your most recent month shows a delinquency status of {value}.",
    "PAY_2":         "The month before that showed a delinquency status of {value}.",
    "PAY_3":         "Three months ago, your account showed a delinquency status of {value}.",
    "PAY_4":         "Four months ago, your account showed a delinquency status of {value}.",
    "PAY_5":         "Five months ago, your account showed a delinquency status of {value}.",
    "PAY_6":         "Six months ago, your account showed a delinquency status of {value}.",
    "PAY_MAX":       "The worst delinquency status observed in the last 6 months was {value}.",
    "PAY_MEAN":      "Your average delinquency status across the last 6 months is {value}.",
    "DELINQ_COUNT":  "You had {value} month(s) of late payment in the last 6 months.",
    "BILL_AMT1":     "Your most recent bill amount of NT${value} is elevated relative to peers.",
    "BILL_MEAN":     "Your average bill amount over the last 6 months is NT${value}.",
    "PAY_AMT1":      "Your most recent payment of NT${value} is low relative to your bill.",
    "PAY_AMT_MEAN":  "Your average payment over the last 6 months is NT${value}.",
    "UTILIZATION":   "Your credit utilisation ratio is {value}.",
    "PAYMENT_RATIO": "Your payment-to-bill ratio is {value}.",
    "LIMIT_BAL":     "Your credit limit of NT${value} is a contributing factor.",
    "LIMIT_PER_AGE": "Your credit-limit-per-age ratio is {value}.",
    "AGE":           "Your age is a contributing factor.",
}


def _format_value(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.2f}"
    return str(v)


def top_negative_reasons(
    waterfall_df: pd.DataFrame, top_n: int = 3
) -> list[str]:
    """Return up to ``top_n`` plain-language sentences from a waterfall.

    The waterfall is expected to have FEATURE, VALUE, SHAP_VALUE columns
    as produced by :func:`src.explain.get_waterfall_data`. Only features
    with a **positive** SHAP value (push the prediction toward default)
    are eligible; features without a template are skipped.
    """
    sentences: list[str] = []
    eligible = waterfall_df[waterfall_df["SHAP_VALUE"] > 0].copy()
    eligible = eligible.sort_values("SHAP_VALUE", ascending=False)
    for _, row in eligible.iterrows():
        feature = row["FEATURE"]
        template = TEMPLATES.get(feature)
        if template is None:
            continue
        sentence = template.format(value=_format_value(row["VALUE"]))
        sentences.append(sentence)
        if len(sentences) >= top_n:
            break
    return sentences


def format_adverse_action_block(reasons: list[str]) -> str:
    """Format the reason list as a copy-paste block for an adverse-action notice."""
    if not reasons:
        return (
            "We are unable to approve this application. A credit officer will "
            "review the decision on request."
        )
    header = (
        "We are unable to approve this application at this time. The "
        "principal factors that contributed to this decision are:"
    )
    bullets = "\n".join(f"  • {s}" for s in reasons)
    footer = (
        "You have the right to obtain human intervention and to contest "
        "this decision (GDPR Article 22)."
    )
    return f"{header}\n\n{bullets}\n\n{footer}"
