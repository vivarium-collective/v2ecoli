import tomllib, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent

def test_pyproject_has_no_parsimony():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert not any("parsimony" in d for d in deps)
    assert "pbg-parsimony" not in data.get("tool", {}).get("uv", {}).get("sources", {})

def test_structural_investigation_removed():
    assert not (ROOT / "workspace/investigations/structural-ecoli").exists()
    assert not (ROOT / "workspace/studies/s01-birth-and-division").exists()
