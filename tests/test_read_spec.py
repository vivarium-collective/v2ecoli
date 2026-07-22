"""Unit-test the manifest parser scripts/_read_spec.py (the spec → bash bridge).

Locks in the row shapes comparison_harness.sh consumes, plus the spec fields
added for the spec-driven vEcoli fork + the upstream-wrapper one-command route:
  - engine rows (repo/commit/branch),
  - per-condition resolution (condition > defaults > CLI),
  - vecoli-fork: repo<TAB>ref with commit > branch > 'master' fallback + defaults,
  - vecoli-engine: default 'upstream-wrapper',
  - from-vecoli-config: the fork config path (or '' when unset).
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "_read_spec", os.path.join(_HERE, "..", "scripts", "_read_spec.py"))
rs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rs)


FULL = {
    "v2ecoli": {"repo": "https://x/v2", "commit": "abc", "branch": "main"},
    "vecoli": {"repo": "https://github.com/CovertLab/vEcoli",
               "commit": "deadbeef", "branch": "master"},
    "vecoli_engine": "upstream-wrapper",
    "from_vecoli_config": "configs/default.json",
    "defaults": {"seeds": 4, "gens": 2},
    "conditions": [
        {"name": "basal", "config": "cond_basal.json"},          # inherits defaults
        {"name": "with_aa", "config": "cond_with_aa.json", "seeds": 8, "gens": 6},
    ],
}


def test_engine_row():
    assert rs.engine_row(FULL, "v2ecoli") == ("https://x/v2", "abc", "main")


def test_condition_resolution_precedence():
    rows = list(rs.condition_rows(FULL, cli_seeds=1, cli_gens=1))
    # basal: no per-condition seeds/gens → defaults (4, 2)
    assert rows[0] == ("basal", "cond_basal.json", 4, 2)
    # with_aa: per-condition wins over defaults
    assert rows[1] == ("with_aa", "cond_with_aa.json", 8, 6)


def test_vecoli_fork_prefers_commit():
    assert rs.vecoli_fork(FULL) == ("https://github.com/CovertLab/vEcoli", "deadbeef")


def test_vecoli_fork_falls_back_to_branch_then_default():
    # no commit → branch
    spec = {"vecoli": {"repo": "https://r", "branch": "feat/x"}}
    assert rs.vecoli_fork(spec) == ("https://r", "feat/x")
    # nothing → CovertLab/master defaults
    assert rs.vecoli_fork({}) == (rs.DEFAULT_VECOLI_REPO, rs.DEFAULT_VECOLI_REF)
    assert rs.DEFAULT_VECOLI_REPO == "https://github.com/CovertLab/vEcoli"
    assert rs.DEFAULT_VECOLI_REF == "master"


def test_vecoli_engine_default_and_explicit():
    assert rs.vecoli_engine(FULL) == "upstream-wrapper"
    assert rs.vecoli_engine({}) == "upstream-wrapper"
    assert rs.vecoli_engine({"vecoli_engine": "nextflow"}) == "nextflow"


def test_from_vecoli_config():
    assert rs.from_vecoli_config(FULL) == "configs/default.json"
    assert rs.from_vecoli_config({}) == ""


SPEC = {
    "v2ecoli": {"repo": "r1", "commit": "c1"},
    "vecoli": {"repo": "r2", "commit": "c2"},
    "defaults": {"cards": ["standard"]},
    "configs": [
        {"config": "configs/cond_basal.json", "cards": ["standard", "statistical"]},
        {"config": "configs/cond_with_aa.json"},
    ],
}


def test_config_rows_uses_entry_cards_then_defaults():
    rows = list(rs.config_rows(SPEC))
    assert rows == [
        ("configs/cond_basal.json", "standard,statistical"),
        ("configs/cond_with_aa.json", "standard"),
    ]


def test_config_rows_defaults_to_standard_when_no_defaults():
    spec = {"configs": [{"config": "c.json"}]}
    assert list(rs.config_rows(spec)) == [("c.json", "standard")]

