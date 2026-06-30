import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2ecoli.workflow.post_sim import (
    KINDS, POST_SIM_REGISTRY, iter_post_sim, register_post_sim)


class _A: name = "a_demo"
class _V: name = "v_demo"


def test_register_and_iter_by_kind():
    register_post_sim(_A, "analysis")
    register_post_sim(_V, "visualization")
    assert POST_SIM_REGISTRY["a_demo"] == {"cls": _A, "kind": "analysis"}
    names = dict(iter_post_sim())
    assert "a_demo" in names and "v_demo" in names
    assert [n for n, _ in iter_post_sim("visualization")] == ["v_demo"]
    assert [n for n, _ in iter_post_sim("analysis")] == ["a_demo"]


def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        register_post_sim(_A, "bogus")


def test_blank_name_is_noop():
    class _N: name = ""
    before = len(POST_SIM_REGISTRY)
    register_post_sim(_N, "analysis")
    assert len(POST_SIM_REGISTRY) == before


def test_kinds_constant():
    assert KINDS == ("analysis", "visualization", "report_card")
