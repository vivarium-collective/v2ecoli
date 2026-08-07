"""Flush Plan 3 hardening: report_cards allowlist in run_flush, worker flush
de-dup (sequential path only flushes on the driver), and a re-flush CLI."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _study(tmp_path, slug="demo", spec=None):
    sd = tmp_path / "workspace" / "studies" / slug
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump(spec or {"name": slug}))
    return sd


# --- Change 1: run_flush honors a study's report_cards: allowlist ---------------

def test_run_flush_honors_report_cards_allowlist(core, tmp_path):
    from v2ecoli.workflow.flush import run_flush
    # study has tests AND a vs_vecoli ref, but declares report_cards: [tests]
    ref = tmp_path / "rc" / "v.json"
    ref.parent.mkdir(parents=True)
    ref.write_text('{"schema":"report_card_verdict/v1","overall":"drift",'
                   '"groups":{"standard":{"verdict":"drift","axes":[]}}}')
    _study(tmp_path, "demo", {
        "name": "demo",
        "report_cards": ["tests"],                       # allowlist: tests only
        "tests": [{"name": "t1", "status": "passed",
                   "pass_if": {"op": "in_range", "low": 1, "high": 2}}],
        "report_card_refs": {"vs_vecoli": "rc/v.json"},   # would apply but excluded
    })
    res = run_flush(str(tmp_path / "out"), {"study": "demo"}, tmp_path,
                    core=core, kinds=("report_card",))
    names = {p["name"] for p in res["placed"]}
    assert "tests" in names
    assert "vs_vecoli" not in names   # excluded by the allowlist despite a valid ref


def test_run_flush_ignores_machine_generated_embed_paths(core, tmp_path):
    """#439, flush half: a path-shaped report_cards: must not exclude registry cards.

    The comparison studies get this key written as embed paths by
    scripts/_compare/materialize.py, so the name-vs-path comparison skipped every
    card. Both gates must agree, or eligibility depends on which one evaluates.
    """
    from v2ecoli.workflow.flush import run_flush
    _study(tmp_path, "demo", {
        "name": "demo",
        # verbatim shape from scripts/_compare/materialize.py
        "report_cards": ["viz/report_card/standard.html", "viz/report_card/parca.html"],
        "tests": [{"name": "t1", "status": "passed",
                   "pass_if": {"op": "in_range", "low": 1, "high": 2}}],
    })
    res = run_flush(str(tmp_path / "out"), {"study": "demo"}, tmp_path,
                    core=core, kinds=("report_card",))
    assert "tests" in {p["name"] for p in res["placed"]}


def test_run_flush_no_allowlist_runs_all_applicable(core, tmp_path):
    from v2ecoli.workflow.flush import run_flush
    _study(tmp_path, "demo", {
        "name": "demo",
        "tests": [{"name": "t1", "status": "passed",
                   "pass_if": {"op": "in_range", "low": 1, "high": 2}}],
    })
    res = run_flush(str(tmp_path / "out"), {"study": "demo"}, tmp_path,
                    core=core, kinds=("report_card",))
    assert "tests" in {p["name"] for p in res["placed"]}


# --- Change 2: workers (run_analysis=False) do NOT flush -------------------------

def test_sequential_worker_does_not_flush(monkeypatch):
    import v2ecoli.workflow.run as run_mod
    calls = {"n": 0}
    monkeypatch.setattr(run_mod, "_maybe_flush",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or (a[2] if len(a) > 2 else {}))
    # call the sequential path's flush gate directly via a tiny stand-in:
    # the production gate is `if run_analysis: result = _maybe_flush(...)`.
    # Verify the helper that decides it.
    assert run_mod._should_flush(run_analysis=False) is False
    assert run_mod._should_flush(run_analysis=True) is True


# --- Change 3: a standalone re-flush CLI ---------------------------------------

def test_reflush_cli_runs_flush(core, tmp_path, monkeypatch, capsys):
    import v2ecoli.workflow.flush as flush_mod
    seen = {}
    def _fake(out_dir, config, ws_root, **kw):
        seen.update(out_dir=out_dir, study=config.get("study"), kinds=kw.get("kinds"))
        return {"placed": [{"kind": "report_card", "name": "tests", "path": "p"}],
                "skipped": [], "study": config.get("study")}
    monkeypatch.setattr(flush_mod, "run_flush", _fake)
    rc = flush_mod.main(["out/x", "--study", "demo", "--ws-root", str(tmp_path),
                         "--kinds", "report_card"])
    assert rc == 0
    assert seen["out_dir"] == "out/x" and seen["study"] == "demo"
    assert seen["kinds"] == ("report_card",)
    assert "placed 1" in capsys.readouterr().out
