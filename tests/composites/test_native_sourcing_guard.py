import pytest
from v2ecoli.composites.ecoli_baseline import assert_injection_sourcing


def test_native_forbids_fork_repo():
    with pytest.raises(ValueError, match=r"baseline\(native=False\)"):
        assert_injection_sourcing(
            native=True,
            injected_processes={"swap_processes": {"a": "b"}, "fork_repo": "/some/vEcoli"},
        )


def test_native_allows_empty_fork_repo():
    # native injection carries fork_repo="" (+ cache_dir); must not raise
    assert_injection_sourcing(
        native=True,
        injected_processes={"swap_processes": {"a": "b"}, "fork_repo": "", "cache_dir": "/c"},
    ) is None


def test_fork_sourcing_requires_fork_repo():
    with pytest.raises(ValueError, match=r"baseline\(native=False\) requires"):
        assert_injection_sourcing(
            native=False,
            injected_processes={"add_processes": ["p"], "fork_repo": ""},
        )


def test_fork_sourcing_with_fork_repo_ok():
    assert assert_injection_sourcing(
        native=False,
        injected_processes={"add_processes": ["p"], "fork_repo": "/some/vEcoli"},
    ) is None


def test_no_injection_is_noop():
    assert assert_injection_sourcing(native=True, injected_processes=None) is None
    assert assert_injection_sourcing(native=False, injected_processes={}) is None
