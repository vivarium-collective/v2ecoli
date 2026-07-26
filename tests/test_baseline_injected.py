"""Tests for baseline() injected_processes hook (Task 4)."""
import os

import pytest

# These build a real baseline document, which needs a ParCa cache
# (out/cache/initial_state.json). Mark them `sim` so they run in the
# behavior-tests CI job (which builds out/cache) and are deselected from the
# cache-less fast-tests job.
pytestmark = pytest.mark.sim

FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")


def test_baseline_injects_fork_process():
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    core = build_core()
    inj = {"fork_repo": FORK, "add_processes": ["example-secretion"],
           "swap_processes": {},
           "process_configs": {"example-secretion": {"rate": 2.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    doc = baseline(core=core, seed=0, cache_dir="out/cache",
                   injected_processes=inj)
    cell = doc["state"]["agents"]["0"]
    assert "example-secretion" in cell
    assert "example-secretion" in doc["flow_order"]


def test_baseline_noop_without_injection_keeps_process_set():
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir="out/cache")
    cell = doc["state"]["agents"]["0"]
    assert "example-secretion" not in cell
