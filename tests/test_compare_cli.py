import scripts.compare_cli as cli


def test_run_invokes_run_investigation_with_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.runner, "run_investigation",
                        lambda inv, out, mode, render_only: captured.update(
                            inv=inv, out=out, mode=mode, render_only=render_only) or 0)
    rc = cli.main(["run"])
    assert rc == 0
    assert captured == {"inv": "v2ecoli-vecoli-comparison", "out": "out/report",
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


def test_scaffold_materializes_all_studies(monkeypatch):
    seen = []
    fake_specs = [type("S", (), {"name": "basal"})(), type("S", (), {"name": "with_aa"})()]
    monkeypatch.setattr(cli.runner, "load_investigation",
                        lambda ref: ({}, fake_specs))
    monkeypatch.setattr(cli, "_materialize", lambda spec: seen.append(spec.name))
    rc = cli.main(["scaffold", "v2ecoli-vecoli-comparison"])
    assert rc == 0 and seen == ["basal", "with_aa"]
