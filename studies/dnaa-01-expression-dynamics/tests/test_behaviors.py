"""One pytest per `expected_behavior:` entry in study.yaml.

This file is intentionally thin: the assertion logic lives in
`_behaviors.evaluate()`. Each English sentence in study.yaml gets one
parametrized test, so adding a new behavior is purely a YAML edit — no
new test code.

Status semantics:
- `implemented` -> test runs normally. Skips cleanly when the required
  run isn't in runs.db yet.
- `stub`        -> test is marked xfail (expected to fail until the
  upstream gap closes).
- `gated`       -> test is xfail with a `requires:` reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from _behaviors import evaluate


STUDY_DIR = Path(__file__).resolve().parents[1]


def _load_expected_behavior() -> list[dict]:
    spec = yaml.safe_load((STUDY_DIR / "study.yaml").read_text())
    return spec.get("expected_behavior") or []


def _ids(entries: list[dict]) -> list[str]:
    return [e.get("name") or f"behavior-{i}" for i, e in enumerate(entries)]


_ENTRIES = _load_expected_behavior()


@pytest.mark.parametrize("entry", _ENTRIES, ids=_ids(_ENTRIES))
def test_expected_behavior(entry, request):
    status = entry.get("status", "implemented")
    if status == "stub":
        pytest.xfail(entry.get("notes") or f"{entry['name']} is a stub")

    # Resolve the right history fixture from `given.run`.
    given = entry.get("given") or {}
    run_name = given.get("run", "baseline")
    if run_name == "baseline":
        history = request.getfixturevalue("baseline_history")
    elif run_name == "variant":
        variant_name = given.get("variant")
        if variant_name == "stop-dnaA-synthesis":
            history = request.getfixturevalue("stop_synthesis_history")
        else:
            pytest.skip(f"no fixture for variant {variant_name!r}")
    else:
        pytest.skip(f"unknown run {run_name!r}")

    passed, message = evaluate(entry, history)
    assert passed, f"{entry['en']}\n  -> {message}"
