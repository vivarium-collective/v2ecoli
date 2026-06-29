"""Swap semantics for fork process injection: removing the swapped-out process
and any excluded processes from the composite (the converter/add path already
exists in inject.py; this is the missing 'remove' half of a true swap)."""
import os

import pytest

from scripts._compare import inject

FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")


def test_remove_processes_drops_from_state_and_flow():
    cell_state = {
        "ecoli-metabolism": {"_type": "process"},
        "keep-me": {"_type": "process"},
        "exchange_data": {"_type": "step"},
    }
    flow_order = ["ecoli-metabolism", "keep-me", "exchange_data"]

    removed = inject.remove_processes(
        cell_state, flow_order, ["ecoli-metabolism", "exchange_data"])

    assert "ecoli-metabolism" not in cell_state
    assert "exchange_data" not in cell_state
    assert "keep-me" in cell_state
    assert flow_order == ["keep-me"]
    assert set(removed) == {"ecoli-metabolism", "exchange_data"}


def test_remove_processes_ignores_absent_names():
    cell_state = {"keep-me": {"_type": "process"}}
    flow_order = ["keep-me"]

    removed = inject.remove_processes(cell_state, flow_order, ["not-here"])

    assert flow_order == ["keep-me"]
    assert removed == []


@pytest.mark.sim
def test_baseline_swaps_and_excludes_processes():
    """A swap-only injection (no add_processes) must still convert+add the swap
    target, remove the swapped-out process, and drop exclude_processes."""
    from v2ecoli.core import build_core
    from v2ecoli.composites.baseline import baseline
    core = build_core()
    inj = {
        "fork_repo": FORK,
        "add_processes": [],
        "swap_processes": {"ecoli-mass-listener": "example-secretion"},
        "exclude_processes": ["exchange_data"],
        "process_configs": {"example-secretion": {"rate": 2.0}},
        "topology": {"example-secretion": {"counts": ["bulk"]}},
        "time_step": 1.0,
    }
    doc = baseline(core=core, seed=0, cache_dir="out/cache",
                   injected_processes=inj)
    cell = doc["state"]["agents"]["0"]
    assert "example-secretion" in cell            # swap target converted + added
    assert "ecoli-mass-listener" not in cell      # swapped-out removed
    assert "exchange_data" not in cell            # excluded removed
    assert "ecoli-mass-listener" not in doc["flow_order"]
    assert "exchange_data" not in doc["flow_order"]
