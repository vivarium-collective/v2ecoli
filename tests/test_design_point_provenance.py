"""A run must be able to say which design point it ran.

The comparison harness hardcoded ``variant=0`` on the genuine-vEcoli arm and
offered no way to change it, so the reference side could only ever run the
UNPERTURBED BASELINE. On a heterologous-pathway configuration that baseline is
typically the declared no-expression control — the variant such a config's own
``skip_baseline: true`` exists to exclude — so an equivalence study would have
compared a producing candidate against a control reference and reported the
difference as an engine disagreement. Measured 2026-08-25 on one such pair: the
reference product read 0 at all 9 timepoints while its three positive controls
were live and correctly signed, so nothing in the output could have revealed it.

Neither arm recorded a design point either, and the v2ecoli cache did not record
what perturbation it carried, so the mistake was not reconstructable after the
fact. These tests cover the flag and the provenance together, because a flag
without a record is a second way to be wrong quietly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# --- the cache records what it encodes -------------------------------------

def test_design_point_filename_is_stable():
    """Consumers read this by name; renaming it silently breaks provenance."""
    from v2ecoli.perturbations.variant_cache import DESIGN_POINT_FILE
    assert DESIGN_POINT_FILE == "design_point.json"


def test_a_cache_without_a_design_point_reads_as_missing_not_as_baseline(tmp_path):
    """★ The load-bearing case. A cache built by any other route — build_cache.py,
    an upstream fixture — has no design point, and that must read as UNRECORDED
    rather than as 'the baseline was used'. Absence and zero are different
    claims, and this whole defect came from treating one as the other."""
    import scripts.run_comparison_ensemble as rce
    assert rce.read_design_point(str(tmp_path)) is None


def test_an_unreadable_design_point_does_not_raise(tmp_path):
    """Provenance that cannot be parsed must not stop a run — it must be
    reported as absent. A harness that dies on a malformed sidecar gets the
    sidecar deleted."""
    import scripts.run_comparison_ensemble as rce
    (tmp_path / "design_point.json").write_text("{not json")
    assert rce.read_design_point(str(tmp_path)) is None


def test_a_written_design_point_round_trips(tmp_path):
    """The positive control for the two tests above: if reading always returned
    None they would both pass while proving nothing."""
    import scripts.run_comparison_ensemble as rce
    payload = {"label": "induced", "condition": "basal",
               "new_gene": {"expression": 1.17e6}}
    (tmp_path / "design_point.json").write_text(json.dumps(payload))
    got = rce.read_design_point(str(tmp_path))
    assert got is not None
    assert got["label"] == "induced"
    assert got["new_gene"]["expression"] == pytest.approx(1.17e6)


# --- the flag exists and defaults to the old behaviour ----------------------

def test_the_variant_flag_reaches_the_real_parser(capsys):
    """Exercises the ACTUAL parser rather than a re-declaration of it: --help
    builds the same argparse object the CLI uses. A source grep would pass
    against a flag defined in a dead branch."""
    import scripts.run_comparison_ensemble as rce
    with pytest.raises(SystemExit) as exc:
        rce.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--variant" in out, "the flag never reaches the CLI surface"
    # Positive control: a flag known to exist must also show, so a help text
    # that rendered empty could not make the assertion above pass vacuously.
    assert "--composite" in out


def test_the_variant_flag_is_documented_as_arm_asymmetric():
    """⚠ The flag means different things on the two arms — APPLIED on vecoli,
    a LABEL on v2ecoli where the cache carries the design point. A caller who
    does not know that will mislabel a run, so the help text has to say it."""
    src = Path("scripts/run_comparison_ensemble.py").read_text()
    i = src.index('"--variant"')
    helptext = src[i:i + 800]
    assert "1-BASED" in helptext
    assert "vecoli arm" in helptext
    assert "cache" in helptext


def test_the_hardcoded_variant_zero_is_gone():
    """The defect itself: the vecoli arm passed a literal 0 and no flag reached
    it. Asserted on the source because the alternative is a full engine run."""
    src = Path("scripts/run_comparison_ensemble.py").read_text()
    assert "variant=0, lineage_seed=seed" not in src
    assert "variant=variant, lineage_seed=seed" in src
