import scripts.compare_cli as cli


def test_run_invokes_run_investigation_with_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.runner, "run_investigation",
                        lambda inv, out, mode, render_only: captured.update(
                            inv=inv, out=out, mode=mode, render_only=render_only) or 0)
    rc = cli.main(["run"])
    assert rc == 0
    assert captured == {"inv": "whole-cell-model-comparison", "out": "out/report",
                        "mode": "serial", "render_only": False}


def test_run_ray_and_explicit_investigation(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.runner, "run_investigation",
                        lambda inv, out, mode, render_only: captured.update(
                            inv=inv, mode=mode, render_only=render_only) or 0)
    cli.main(["run", "my-investigation", "--ray", "--render-only"])
    assert captured == {"inv": "my-investigation", "mode": "ray", "render_only": True}


def test_study_loads_spec_then_runs_it(monkeypatch):
    captured = {}
    sentinel = object()
    monkeypatch.setattr(cli.runner, "load_study",
                        lambda name: captured.update(loaded=name) or sentinel)
    monkeypatch.setattr(cli.runner, "run_study",
                        lambda spec, out, mode, render_only: captured.update(
                            spec=spec, out=out, mode=mode) or 0)
    rc = cli.main(["study", "basal_4x4", "--ray", "--out", "out/x"])
    assert rc == 0
    assert captured["loaded"] == "basal_4x4"
    assert captured["spec"] is sentinel
    assert captured["mode"] == "ray" and captured["out"] == "out/x"


def test_study_render_only_propagates(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.runner, "load_study", lambda name: object())
    monkeypatch.setattr(cli.runner, "run_study",
                        lambda spec, out, mode, render_only: captured.update(
                            render_only=render_only) or 0)
    cli.main(["study", "basal", "--render-only"])
    assert captured["render_only"] is True


def test_scaffold_cli_builds_specs_from_configs_not_members(monkeypatch, tmp_path):
    """Regression test for the Task 6 fix-round finding: `scaffold` must build
    its specs via specs_from_configs(_context(...)) -- the same path `run`/`init`
    use -- not runner.load_investigation(), which resolves the legacy `members:`
    list and is empty for any configs[]-only (post-Task-6) investigation."""
    from scripts._compare.scaffold import scaffold_investigation
    inv_path = scaffold_investigation(
        name="scaffold-cli-test", reference_repo="/abs/vEcoli",
        configs=["basal", "with_aa", "acetate"], out_root=tmp_path)
    seen = []
    monkeypatch.setattr(cli, "_materialize", lambda spec: seen.append(spec.name))
    rc = cli.main(["scaffold", str(inv_path)])
    assert rc == 0
    assert seen == ["basal", "with_aa", "acetate"]
