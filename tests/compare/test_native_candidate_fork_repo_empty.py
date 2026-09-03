"""Regression: the native candidate arm of run_comparison_ensemble.py must
never leak a truthy fork checkout dir into injected_processes.fork_repo.

``composite_kind == "v2ecoli"`` (``--composite ecoli_baseline``) is native-only
since fork-sourcing was removed (v2ecoli #651): assert_injection_sourcing in
ecoli_baseline.py hard-errors on any non-empty ``fork_repo``. make_run_one's
``--from-vecoli-config`` assembly resolves the fork's config STRUCTURE via
``resolve_vecoli_config_local(from_vecoli_config, fork_dir)`` (needs no fork
runtime — v2ecoli's own loader reads the file), but must pass ``fork_repo=""``
into ``_injected_from_resolved``, not the fork dir it used to read that file.
"""
import pytest

import scripts.run_comparison_ensemble as rce
from v2ecoli.composites.ecoli_baseline import assert_injection_sourcing


def _patch_resolve_vecoli_config_local(monkeypatch, resolved: dict):
    """make_run_one imports resolve_vecoli_config_local locally inside its try
    block (``from scripts._compare.config_adapter import
    resolve_vecoli_config_local``), so patching the source module's attribute
    is what a fresh local import picks up."""
    import scripts._compare.config_adapter as config_adapter
    monkeypatch.setattr(
        config_adapter, "resolve_vecoli_config_local",
        lambda config_path, fork_dir: resolved)


def test_native_candidate_gets_fork_repo_empty_not_the_fork_dir(monkeypatch, tmp_path):
    captured = {}
    orig = rce._injected_from_resolved

    def _spy(resolved, fork_repo, fork_sim_data, **kw):
        captured["fork_repo"] = fork_repo
        return orig(resolved, fork_repo, fork_sim_data, **kw)

    monkeypatch.setattr(rce, "_injected_from_resolved", _spy)
    _patch_resolve_vecoli_config_local(
        monkeypatch,
        {"add_processes": ["example-secretion"], "swap_processes": {}})

    fork_dir = "/some/checkout/vEcoli"  # truthy — the value that used to leak
    rce.make_run_one(
        composite_kind="v2ecoli", condition="basal", cache_dir=str(tmp_path),
        max_generations=1, max_steps=1, chunk=1, out_root=str(tmp_path),
        from_vecoli_config="dummy.json", vecoli_dir=fork_dir)

    assert captured["fork_repo"] == ""            # NOT fork_dir
    assert captured["fork_repo"] != fork_dir


def test_native_candidate_injection_clears_the_sourcing_guard(monkeypatch, tmp_path):
    """The exact injected_processes block make_run_one builds for the native
    candidate must not raise assert_injection_sourcing — the actual failure
    mode of the regression this pins. Pre-fix this raises ValueError because
    the captured fork_repo is the truthy fork checkout dir, not "".
    """
    captured = {}
    orig = rce._injected_from_resolved

    def _spy(resolved, fork_repo, fork_sim_data, **kw):
        inj = orig(resolved, fork_repo, fork_sim_data, **kw)
        captured["inj"] = inj
        return inj

    monkeypatch.setattr(rce, "_injected_from_resolved", _spy)
    _patch_resolve_vecoli_config_local(
        monkeypatch,
        {"add_processes": ["example-secretion"], "swap_processes": {}})

    fork_dir = "/some/checkout/vEcoli"
    rce.make_run_one(
        composite_kind="v2ecoli", condition="basal", cache_dir=str(tmp_path),
        max_generations=1, max_steps=1, chunk=1, out_root=str(tmp_path),
        from_vecoli_config="dummy.json", vecoli_dir=fork_dir)

    assert assert_injection_sourcing(captured["inj"]) is None   # must not raise


def test_vecoli_reference_arm_config_assembly_is_unchanged(monkeypatch, tmp_path):
    """The vecoli reference arm resolves the SAME fork config but must keep
    using the real fork dir for its own swap/flow application — this fix must
    not touch that branch at all."""
    captured = {}
    _patch_resolve_vecoli_config_local(
        monkeypatch,
        {"swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"},
         "flow": None})

    def _spy_resolve(config_path, fork_dir):
        captured["fork_dir_seen"] = fork_dir
        return {"swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"},
                "flow": None}

    import scripts._compare.config_adapter as config_adapter
    monkeypatch.setattr(config_adapter, "resolve_vecoli_config_local", _spy_resolve)

    fork_dir = "/some/checkout/vEcoli"
    rce.make_run_one(
        composite_kind="vecoli", condition="basal", cache_dir=str(tmp_path),
        max_generations=1, max_steps=1, chunk=1, out_root=str(tmp_path),
        from_vecoli_config="dummy.json", vecoli_dir=fork_dir)

    # The vecoli arm's resolve call still receives the real fork dir (it needs
    # it to run the fork's genuine EcoliSim) — untouched by this fix.
    assert captured["fork_dir_seen"] == fork_dir
