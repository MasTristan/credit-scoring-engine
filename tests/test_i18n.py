import pytest

from src import i18n


@pytest.fixture(autouse=True)
def reset_language():
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def test_default_is_english():
    assert i18n.get_language() == "en"
    assert i18n.t("Score", "Scorer") == "Score"


def test_switch_to_french():
    i18n.set_language("fr")
    assert i18n.get_language() == "fr"
    assert i18n.t("Score", "Scorer") == "Scorer"


def test_unknown_language_rejected():
    with pytest.raises(ValueError):
        i18n.set_language("de")
