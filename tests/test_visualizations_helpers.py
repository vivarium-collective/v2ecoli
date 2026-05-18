"""Unit tests for v2ecoli.visualizations._helpers."""

import pytest


@pytest.mark.fast
def test_classify_returns_known_subsystem():
    from v2ecoli.visualizations._helpers import classify
    key, label = classify("ecoli-equilibrium")
    assert key in {"replication", "transcription", "rna", "translation",
                   "regulation", "signaling", "metabolism", "alloc",
                   "listen", "infra"}
    assert isinstance(label, str) and label


@pytest.mark.fast
def test_render_document_wraps_html():
    from v2ecoli.visualizations._helpers import render_document
    html = render_document(title="Test Page", body_html="<p>hello</p>",
                            include_banner=False)
    assert html.startswith("<!doctype html>") or html.startswith("<!DOCTYPE html>") \
        or html.startswith("<html")
    assert "<p>hello</p>" in html
    assert "Test Page" in html


@pytest.mark.fast
def test_render_repro_banner_includes_python_version():
    import sys
    from v2ecoli.visualizations._helpers import render_repro_banner
    banner = render_repro_banner()
    assert isinstance(banner, str)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert py_version in banner


@pytest.mark.fast
def test_history_to_arrays_extracts_paths():
    """history_to_arrays takes list-of-dict trajectory + list of dotted paths,
    returns {path: np.array} where each array is the column of values."""
    import numpy as np
    from v2ecoli.visualizations._helpers import history_to_arrays
    history = [
        {"a": 1.0, "b": {"c": 2.0}},
        {"a": 3.0, "b": {"c": 4.0}},
    ]
    arrs = history_to_arrays(history, ["a", "b.c"])
    assert "a" in arrs and "b.c" in arrs
    np.testing.assert_array_equal(arrs["a"], [1.0, 3.0])
    np.testing.assert_array_equal(arrs["b.c"], [2.0, 4.0])


@pytest.mark.fast
def test_group_by_generation_groups_rows():
    from v2ecoli.visualizations._helpers import group_by_generation
    history = [
        {"generation": 1, "v": 1},
        {"generation": 1, "v": 2},
        {"generation": 2, "v": 3},
    ]
    groups = group_by_generation(history)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 1
