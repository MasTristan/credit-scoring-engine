"""Minimal bilingual (EN/FR) helper for the Streamlit app.

Usage in the app: read the language selector once per run and call
``set_language(code)``, then wrap every user-facing string as ``t("English",
"Francais")``. Kept free of any Streamlit import so it stays unit-testable.
"""

from __future__ import annotations

LANGUAGES = {"en": "English", "fr": "Francais"}
_DEFAULT = "en"

_current = _DEFAULT


def set_language(lang: str) -> None:
    """Set the active language code ('en' or 'fr')."""
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown language {lang!r}; expected one of {list(LANGUAGES)}.")
    global _current
    _current = lang


def get_language() -> str:
    return _current


def t(en: str, fr: str) -> str:
    """Return the string in the active language (falls back to English)."""
    return fr if _current == "fr" else en
