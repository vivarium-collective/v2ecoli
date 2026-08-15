import sys, types
import pytest
from v2ecoli.library import vivarium_ecoli_engine as ve


def _stub_parse_variants(monkeypatch):
    # Mirror the fork: a 6-point single-parameter grid, op=None.
    mod = types.ModuleType("runscripts.create_variants")
    mod.parse_variants = lambda cfg: [{"dose": v} for v in (0, 1, 2, 3, 4, 5)]
    pkg = types.ModuleType("runscripts")
    monkeypatch.setitem(sys.modules, "runscripts", pkg)
    monkeypatch.setitem(sys.modules, "runscripts.create_variants", mod)


def test_index_zero_is_baseline(monkeypatch):
    _stub_parse_variants(monkeypatch)
    name, params = ve._select_variant_params({"demo_variant": {}}, 0)
    assert (name, params) == (None, None)


def test_index_k_selects_k_minus_one(monkeypatch):
    _stub_parse_variants(monkeypatch)
    name, params = ve._select_variant_params({"demo_variant": {}}, 3)
    assert name == "demo_variant"
    assert params == {"dose": 2}      # param_dicts[3-1]


def test_index_out_of_range_raises(monkeypatch):
    _stub_parse_variants(monkeypatch)
    with pytest.raises(IndexError):
        ve._select_variant_params({"demo_variant": {}}, 7)


def test_apply_dispatches_to_variant_module(monkeypatch):
    _stub_parse_variants(monkeypatch)
    applied = {}
    vmod = types.ModuleType("ecoli.variants.demo_variant")
    def _apply(sim_data, params):
        applied["params"] = params
        sim_data["touched"] = True
        return sim_data
    vmod.apply_variant = _apply
    monkeypatch.setitem(sys.modules, "ecoli", types.ModuleType("ecoli"))
    monkeypatch.setitem(sys.modules, "ecoli.variants", types.ModuleType("ecoli.variants"))
    monkeypatch.setitem(sys.modules, "ecoli.variants.demo_variant", vmod)
    sd = {}
    out, meta = ve._apply_config_variant(sd, {"demo_variant": {}}, 2)
    assert out["touched"] is True
    assert applied["params"] == {"dose": 1}
    assert meta == {"variant_name": "demo_variant", "variant_index": 2, "params": {"dose": 1}}


def test_multiple_variants_raises_valueerror(monkeypatch):
    _stub_parse_variants(monkeypatch)
    with pytest.raises(ValueError):
        ve._select_variant_params({"a": {}, "b": {}}, 1)


def test_variant_requested_but_empty_config_raises(monkeypatch):
    # Finding #3 (spirit): variant>=1 with no 'variants' block must fail loud,
    # never silently run the unperturbed baseline.
    _stub_parse_variants(monkeypatch)
    with pytest.raises(ValueError):
        ve._select_variant_params({}, 2)


def test_variant_zero_with_empty_config_is_baseline_noop():
    # Back-compat: baseline is still a strict no-op (no fork import needed).
    assert ve._select_variant_params({}, 0) == (None, None)


def test_apply_variant_import_error_propagates(monkeypatch):
    # A missing ecoli.variants.<name> module must raise (ImportError), not be
    # swallowed into a silent baseline. This is the failure the build-level try
    # used to swallow (Finding #3).
    _stub_parse_variants(monkeypatch)
    import importlib
    def _boom(name):
        raise ImportError(f"no module {name}")
    monkeypatch.setattr(importlib, "import_module", _boom)
    with pytest.raises(ImportError):
        ve._apply_config_variant({}, {"demo_variant": {}}, 2)
