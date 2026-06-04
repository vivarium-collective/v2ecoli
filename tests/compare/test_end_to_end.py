import os
from pathlib import Path

import pytest

# Heavy: runs real ParCa (full) + a 2-gen sim on BOTH engines (hours).
# Opt in with COMPARE_E2E=1.
pytestmark = pytest.mark.skipif(
    os.environ.get("COMPARE_E2E") != "1",
    reason="set COMPARE_E2E=1 to run the full cross-engine harness",
)


def test_harness_produces_report(tmp_path):
    import scripts.compare_harness as h
    out = tmp_path / "report.html"
    h.main([
        "--config", "/Users/eranagmon/code/vEcoli/configs/two_generations.json",
        "-o", str(out),
        "--workdir", str(tmp_path / "work"),
    ])
    html = out.read_text()
    assert "ParCa / sim_data" in html
    assert "2-generation sim dynamics" in html
    assert "Config &amp; schema diff" in html or "Config & schema diff" in html
