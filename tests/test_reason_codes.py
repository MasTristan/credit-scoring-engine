import pandas as pd

from src.reason_codes import format_adverse_action_block, top_negative_reasons


def test_top_negative_reasons_only_positive_shap():
    wf = pd.DataFrame(
        {
            "FEATURE": ["PAY_0", "PAY_2", "LIMIT_BAL", "UTILIZATION"],
            "VALUE":   [2, -1, 100000, 0.75],
            "SHAP_VALUE": [0.50, -0.30, 0.15, -0.05],
        }
    )
    reasons = top_negative_reasons(wf, top_n=3)
    assert len(reasons) == 2  # only positive-SHAP, recognised features
    assert "delinquency status" in reasons[0]


def test_top_negative_reasons_caps_at_top_n():
    wf = pd.DataFrame(
        {
            "FEATURE": ["PAY_0", "PAY_2", "PAY_3", "PAY_4"],
            "VALUE":   [2, 2, 2, 2],
            "SHAP_VALUE": [0.4, 0.3, 0.2, 0.1],
        }
    )
    reasons = top_negative_reasons(wf, top_n=2)
    assert len(reasons) == 2


def test_format_adverse_action_block_no_reasons():
    block = format_adverse_action_block([])
    assert "credit officer" in block.lower()


def test_format_adverse_action_block_with_reasons():
    block = format_adverse_action_block(["reason A", "reason B"])
    assert "reason A" in block
    assert "GDPR" in block
