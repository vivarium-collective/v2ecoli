"""Reference-engine descriptor: how to run the comparison's reference model.

The candidate is always v2ecoli; the reference engine is declared in the
investigation's `comparison.reference` block. `kind` selects a run-interface
convention. Today only `vecoli` (CovertLab vEcoli-family: ParCa via
`runscripts/parca.py`, sim via `-m runscripts.workflow`) is implemented; the
dispatch is isolated so a new kind is additive.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_KINDS = {"vecoli"}


@dataclass
class ReferenceEngine:
    repo: str
    kind: str

    @classmethod
    def from_spec(cls, block: dict) -> "ReferenceEngine":
        raw = (block or {}).get("repo", "")
        repo = os.environ.get(raw[4:], "") if isinstance(raw, str) and raw.startswith("env:") else raw
        kind = (block or {}).get("kind", "vecoli")
        if kind not in _KINDS:
            raise ValueError(f"unknown reference kind {kind!r}; known: {sorted(_KINDS)}")
        return cls(repo=repo, kind=kind)

    @property
    def python(self) -> str:
        return f"{self.repo}/.venv/bin/python"

    def env(self) -> dict:
        path = os.environ.get("PATH", "")
        return {**os.environ, "PATH": f"{self.repo}/.venv/bin:{path}"}

    def parca_cmd(self, config_path: str, out_dir: str, intermediates_dir: str) -> list:
        return [self.python, "runscripts/parca.py", "--config", config_path,
                "--outdir", out_dir, "--save-intermediates",
                "--intermediates-directory", intermediates_dir]

    def sim_cmd(self, config_path: str) -> list:
        return [self.python, "-m", "runscripts.workflow", "--config", config_path]
