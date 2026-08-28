"""Regression coverage for scripts/build_upstream_parca.py's optional --config
override (item 87). Pure-logic tests only -- _resolve_parca_options has no
filesystem/import side effects beyond reading an explicit config_path, so
these run without the upstream vEcoli checkout this script otherwise needs.

Root cause / motivation: build_upstream_parca.py always built ONE pristine
baseline sim_data (configs/default.json's own parca_options, unconditionally)
-- there was no way for a caller to request a config-driven ParCa build (e.g.
a config declaring its own new_genes) at all. _resolve_parca_options is the
minimal, additive fix: config_path=None (every existing caller) must remain
byte-for-byte identical to the pre-existing behavior."""

import json

import pytest

from scripts.build_upstream_parca import _resolve_parca_options


def _default_cfg():
    return {
        "parca_options": {
            "cpus": 8,
            "operons": True,
            "new_genes": "off",
        }
    }


def test_no_config_path_returns_default_parca_options_unchanged():
    """The critical backward-compatibility case: every existing caller passes
    no config, and must get exactly today's behavior."""
    default_cfg = _default_cfg()
    result = _resolve_parca_options(default_cfg, None)
    assert result == default_cfg["parca_options"]
    # Not the SAME object (build_upstream_parca mutates outdir/cpus/cache_dir
    # on the returned dict afterwards) -- must not alias the input.
    assert result is not default_cfg["parca_options"]


def test_config_overrides_new_genes(tmp_path):
    override = tmp_path / "example_strain.json"
    override.write_text(json.dumps({"parca_options": {"new_genes": "example_new_gene_subdir"}}))

    result = _resolve_parca_options(_default_cfg(), str(override))

    assert result["new_genes"] == "example_new_gene_subdir"
    # Keys the override didn't declare keep the baseline's value.
    assert result["cpus"] == 8
    assert result["operons"] is True


def test_config_with_no_parca_options_key_is_a_no_op(tmp_path):
    """A config that declares no parca_options block at all (e.g. someone
    passes an unrelated config by mistake) must not crash or clobber
    anything -- falls back to the default entirely."""
    override = tmp_path / "no_parca_options.json"
    override.write_text(json.dumps({"n_init_sims": 2}))

    result = _resolve_parca_options(_default_cfg(), str(override))

    assert result == _default_cfg()["parca_options"]


def test_config_can_override_multiple_keys(tmp_path):
    override = tmp_path / "custom.json"
    override.write_text(json.dumps({"parca_options": {"cpus": 1, "operons": False}}))

    result = _resolve_parca_options(_default_cfg(), str(override))

    assert result["cpus"] == 1
    assert result["operons"] is False
    assert result["new_genes"] == "off"  # undeclared key keeps the baseline


def test_missing_config_path_raises_not_silently_ignored(tmp_path):
    """A typo'd --config path must fail loudly, not silently fall back to the
    baseline build (which would look like success while running the wrong
    ParCa config -- exactly the failure class this whole feature exists to
    avoid)."""
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        _resolve_parca_options(_default_cfg(), str(missing))
