import scripts.compare_cli as cli


def test_run_sequences_scaffold_then_run_then_validate(monkeypatch):
    seq = []
    monkeypatch.setattr(cli.scaffold_mod, "scaffold",
                        lambda m, r, force=False: seq.append("scaffold"))
    monkeypatch.setattr(cli.run_comparison, "main",
                        lambda argv: seq.append(("run", argv)) or 0)
    monkeypatch.setattr(cli.validate_mod, "validate", lambda m, r: [])
    rc = cli.main(["run", "comparison_spec.json"])
    assert rc == 0
    assert seq[0] == "scaffold"
    assert seq[1][0] == "run"


def test_run_returns_nonzero_on_drift(monkeypatch):
    monkeypatch.setattr(cli.scaffold_mod, "scaffold", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_comparison, "main", lambda argv: 0)
    monkeypatch.setattr(cli.validate_mod, "validate", lambda m, r: ["basal: drift"])
    assert cli.main(["run", "comparison_spec.json"]) == 1


def test_run_ray_selects_ray_mode(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.scaffold_mod, "scaffold", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_comparison, "main",
                        lambda argv: captured.update(argv=argv) or 0)
    monkeypatch.setattr(cli.validate_mod, "validate", lambda m, r: [])
    cli.main(["run", "spec.json", "--ray"])
    assert "ray" in captured["argv"]
    assert "serial" not in captured["argv"]


def test_render_only_skips_scaffold(monkeypatch):
    seq = []
    monkeypatch.setattr(cli.scaffold_mod, "scaffold",
                        lambda *a, **k: seq.append("scaffold"))
    monkeypatch.setattr(cli.run_comparison, "main", lambda argv: 0)
    monkeypatch.setattr(cli.validate_mod, "validate", lambda m, r: [])
    cli.main(["run", "spec.json", "--render-only"])
    assert "scaffold" not in seq


def test_study_resolves_manifest_and_condition(tmp_path, monkeypatch):
    sdir = tmp_path / "basal"
    sdir.mkdir()
    (sdir / "study.yaml").write_text(
        "comparison_manifest: comparison_spec.json\ncondition: basal\nname: basal\n",
        encoding="utf-8")
    captured = {}
    monkeypatch.setattr(cli.run_comparison, "main",
                        lambda argv: captured.update(argv=argv) or 0)
    rc = cli._run_study(str(sdir), None, "out/x", False, False)
    assert rc == 0
    argv = captured["argv"]
    assert "--condition" in argv and "basal" in argv
    assert any(a.endswith("comparison_spec.json") for a in argv)
