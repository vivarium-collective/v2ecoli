"""The v1 equivalence pin is condition-parameterized, not condition-specific.

The measurement must not branch on the nutrient condition — that is the whole
reason five conditions can share one reviewed script. These tests pin that
property: the same synthetic sweep pinned under two different --condition values
produces byte-identical axes and differs only in the stimulus label, title and
output location.

The sweep is synthesized (a few cells, the columns the reader names) so this
runs in a second without the ~21 GB of real parquet.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_PIN = pathlib.Path(__file__).resolve().parents[1] / "scripts" / \
    "pin_vecoli_equivalence_reference.py"


def _load_pin_module():
    spec = importlib.util.spec_from_file_location("_pin_vecoli_eq", _PIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pin():
    return _load_pin_module()


def test_default_out_is_the_card_named_for_the_condition(pin):
    for cond in ["basal", "acetate", "succinate", "no_oxygen", "with_aa"]:
        out = pin._default_out(cond)
        assert out == (f"docs/report_cards/population_phenotype_{cond}"
                       f"/vs_vecoli/vecoli_reference.json")
    # basal's default must stay where the existing card already lives, so this
    # change cannot silently relocate a committed reference.
    assert pin._default_out("basal").startswith(
        "docs/report_cards/population_phenotype_basal/vs_vecoli")


def _write_v1_sweep(root: pathlib.Path, n_seeds=2, n_gens=5, n_steps=6):
    """A vEcoli-shaped hive-partitioned history sweep: `time` (cumulative, not
    `global_time`) and positional `bulk`, plus the three vector columns."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    t0 = 0.0
    for seed in range(n_seeds):
        for gen in range(n_gens):
            rows = {
                "variant": [], "lineage_seed": [], "generation": [],
                "agent_id": [], "time": [],
                "listeners__mass__dry_mass": [],
                "listeners__mass__protein_mass": [],
                "listeners__mass__rna_mass": [],
                "listeners__mass__dna_mass": [],
                "listeners__mass__cell_mass": [],
                "listeners__mass__volume": [],
                "listeners__replication_data__number_of_oric": [],
                "listeners__replication_data__fork_coordinates": [],
                "listeners__rna_counts__mRNA_cistron_counts": [],
                "listeners__monomer_counts": [],
                "listeners__fba_results__external_exchange_fluxes": [],
            }
            for s in range(n_steps):
                # deterministic, seed/gen-dependent so the ensemble has spread
                scale = 1.0 + 0.01 * seed + 0.002 * gen + 0.05 * s
                rows["variant"].append(0)
                rows["lineage_seed"].append(seed)
                rows["generation"].append(gen)
                rows["agent_id"].append("0" * (gen + 1))
                rows["time"].append(t0 + gen * 2000.0 + s * 100.0)
                rows["listeners__mass__dry_mass"].append(300.0 * scale)
                rows["listeners__mass__protein_mass"].append(130.0 * scale)
                rows["listeners__mass__rna_mass"].append(39.0 * scale)
                rows["listeners__mass__dna_mass"].append(5.4 * scale)
                rows["listeners__mass__cell_mass"].append(1000.0 * scale)
                rows["listeners__mass__volume"].append(1.0 * scale)
                rows["listeners__replication_data__number_of_oric"].append(
                    1 if s < n_steps // 2 else 2)
                rows["listeners__replication_data__fork_coordinates"].append(
                    [] if s < n_steps // 2 else [100, -100])
                rows["listeners__rna_counts__mRNA_cistron_counts"].append(
                    [int(10 * scale), int(20 * scale), int(30 * scale)])
                rows["listeners__monomer_counts"].append(
                    [int(5 * scale), int(6 * scale)])
                rows["listeners__fba_results__external_exchange_fluxes"].append(
                    [-1.0 * scale, 2.0 * scale, -0.5 * scale])
            d = (root / "history" / "experiment_id=e" / "variant=0"
                 / f"lineage_seed={seed}" / f"generation={gen}"
                 / f"agent_id={'0' * (gen + 1)}")
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(rows), d / "0.pq")
    return root


@pytest.fixture(scope="module")
def synthetic_sweep(tmp_path_factory):
    return _write_v1_sweep(tmp_path_factory.mktemp("v1sweep"))


def _run_pin(pin, sweep, out, condition, monkeypatch):
    argv = ["pin", "--sweep-dir", str(sweep), "--condition", condition,
            "--model-ref", "deadbeef", "--gen-lb", "2", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    pin.main()
    return json.loads(pathlib.Path(out).read_text(encoding="utf-8"))


def test_condition_labels_the_reference_without_changing_the_measurement(
        pin, synthetic_sweep, tmp_path, monkeypatch):
    a = _run_pin(pin, synthetic_sweep, tmp_path / "a.json", "acetate", monkeypatch)
    b = _run_pin(pin, synthetic_sweep, tmp_path / "b.json", "no_oxygen", monkeypatch)

    assert a["stimulus"]["condition"] == "acetate"
    assert b["stimulus"]["condition"] == "no_oxygen"
    assert "acetate" in a["title"] and "no_oxygen" in b["title"]

    # The measurement is condition-blind: identical sweep -> identical axes.
    assert a["axes"] == b["axes"], "condition must not change what is measured"
    assert a["axes"], "expected at least one graded axis from the synthetic sweep"


def test_pin_respects_the_burn_in_bound(pin, synthetic_sweep, tmp_path, monkeypatch):
    """gen_lb drops early generations, so a higher bound yields fewer cells."""
    def n_cells(gen_lb):
        argv = ["pin", "--sweep-dir", str(synthetic_sweep), "--model-ref", "x",
                "--gen-lb", str(gen_lb), "--out", str(tmp_path / f"g{gen_lb}.json")]
        monkeypatch.setattr(sys, "argv", argv)
        pin.main()
        d = json.loads((tmp_path / f"g{gen_lb}.json").read_text(encoding="utf-8"))
        ax = d["axes"]["physiology.cell_mass"]
        return len(ax["criterion"]["ref_values"])

    assert n_cells(0) > n_cells(3) > 0


def test_mismatched_flux_width_skips_kpi_axes_instead_of_misaligning(
        pin, synthetic_sweep, tmp_path, monkeypatch, capsys):
    """A medium can change the external-exchange molecule set, and the KPI axes
    are sliced by the TEMPLATE's flux_ids index. When the widths disagree the
    positional assumption is broken, so the KPIs must be dropped with a warning
    rather than slicing the wrong molecule (or raising IndexError)."""
    d = _run_pin(pin, synthetic_sweep, tmp_path / "w.json", "acetate", monkeypatch)
    warned = capsys.readouterr().out

    # the synthetic sweep is 3 fluxes wide; the real template names 87
    assert "template flux_ids" in warned and "not positionally safe" in warned
    assert not [p for p in d["axes"] if p.startswith("fluxes.")
                and p != "fluxes.exchange"], "named flux KPIs must be skipped"
    # the whole-vector scatter is still pinned — it compares v1 to v2, not to the
    # template, so it does not depend on the template's molecule ordering
    assert "fluxes.exchange" in d["axes"]


def test_pin_writes_the_default_location_when_out_is_omitted(
        pin, synthetic_sweep, tmp_path, monkeypatch):
    """--out is optional; the condition decides where the reference lands."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "pin", "--sweep-dir", str(synthetic_sweep), "--condition", "succinate",
        "--model-ref", "x", "--gen-lb", "2",
        "--template", str(_PIN.parents[1] / pin._DEFAULT_TEMPLATE)])
    pin.main()
    assert (tmp_path / pin._default_out("succinate")).is_file()
