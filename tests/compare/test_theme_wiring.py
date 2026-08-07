import re
from pathlib import Path
SRC = Path(__file__).resolve().parents[2] / "scripts" / "_compare"


def test_no_literal_engine_hex_in_helpers():
    txt = (SRC / "plotly_helpers.py").read_text()
    assert "#4f46e5" not in txt and "#d97706" not in txt   # moved to theme
    assert "from scripts._compare.theme import" in txt or "from scripts._compare import theme" in txt


def test_no_literal_status_hex_in_report():
    txt = (SRC / "report.py").read_text()
    assert "#2e7d32" not in txt  # old --green literal
    assert "#ef6c00" not in txt  # old --amber literal
    assert "#c62828" not in txt  # old --red literal
    assert "from scripts._compare.theme import" in txt or "from scripts._compare import theme" in txt
    assert 'css_vars("light")' in txt
    assert 'css_vars("dark")' in txt
    assert "prefers-color-scheme: dark" in txt
