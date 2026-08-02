import os
import pytest
from scripts._compare.reference import ReferenceEngine


def test_from_spec_resolves_env_indirection(monkeypatch):
    monkeypatch.setenv("V2E_VECOLI_DIR", "/tmp/vEcoli")
    r = ReferenceEngine.from_spec({"repo": "env:V2E_VECOLI_DIR", "kind": "vecoli"})
    assert r.repo == "/tmp/vEcoli"
    assert r.python == "/tmp/vEcoli/.venv/bin/python"


def test_from_spec_literal_repo():
    r = ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"})
    assert r.repo == "/abs/vEcoli"


def test_vecoli_run_commands():
    r = ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"})
    parca = r.parca_cmd("/c.json", "/out", "/out")
    assert parca[0] == "/abs/vEcoli/.venv/bin/python"
    assert "runscripts/parca.py" in parca
    sim = r.sim_cmd("/c.json")
    assert sim[:3] == ["/abs/vEcoli/.venv/bin/python", "-m", "runscripts.workflow"]


def test_env_prepends_venv_to_path():
    r = ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"})
    env = r.env()
    assert env["PATH"].startswith("/abs/vEcoli/.venv/bin:")


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        ReferenceEngine.from_spec({"repo": "/abs/x", "kind": "martian"})
