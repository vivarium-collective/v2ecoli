"""Integration: the pbg-superpowers evaluator seam discovers this workspace's
report_card_axis evaluator.

Skipped unless the installed pbg-superpowers carries the pluggable-evaluator
seam (load_workspace_evaluators) — i.e. once
`pip install -e <pbg-superpowers feat/pluggable-workspace-evaluators>` (or the
merged version) is on the path. Run locally with
`PYTHONPATH=/path/to/pbg-superpowers pytest ...` against the feature branch.
"""
from pathlib import Path

import pytest

se = pytest.importorskip("pbg_superpowers.study_evaluator")

if not hasattr(se, "load_workspace_evaluators"):
    pytest.skip("pbg-superpowers lacks the pluggable-evaluator seam",
                allow_module_level=True)


def test_seam_discovers_report_card_axis():
    ws_root = Path(__file__).resolve().parents[1]  # worktree root
    se.clear_workspace_evaluator_cache()
    reg = se.load_workspace_evaluators(ws_root)
    assert "report_card_axis" in reg
