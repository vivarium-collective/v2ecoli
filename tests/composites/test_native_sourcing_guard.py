import pytest
from v2ecoli.composites.ecoli_baseline import assert_injection_sourcing


def test_fork_repo_now_raises():
    # Fork-sourcing removed: any non-empty fork_repo is a hard error, whether or
    # not there is an add/swap alongside it.
    with pytest.raises(ValueError, match="fork-sourcing has been removed"):
        assert_injection_sourcing(
            injected_processes={"swap_processes": {"a": "b"}, "fork_repo": "/some/vEcoli"},
        )


def test_bare_fork_repo_raises():
    with pytest.raises(ValueError, match="native-only"):
        assert_injection_sourcing(
            injected_processes={"add_processes": ["p"], "fork_repo": "/some/vEcoli"},
        )


def test_native_injection_ok():
    # Native (fork-free) injection carries fork_repo="" (+ cache_dir); must not raise.
    assert assert_injection_sourcing(
        injected_processes={"swap_processes": {"a": "b"}, "fork_repo": "", "cache_dir": "/c"},
    ) is None


def test_native_injection_without_fork_repo_key_ok():
    # A native injection that omits fork_repo entirely is fine.
    assert assert_injection_sourcing(
        injected_processes={"add_processes": ["p"]},
    ) is None


def test_no_injection_is_noop():
    assert assert_injection_sourcing(injected_processes=None) is None
    assert assert_injection_sourcing(injected_processes={}) is None
