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


# --- Measurement symmetry with the v2 side -------------------------------------
# Both sides of an equivalence card must select cells the same way, or the verdict
# confounds a reader difference with an engine difference. These pin the v1 reader
# against PopulationPhenotypeBasalCard.analyze's filters
# (v2ecoli/workflow/analysis.py) rather than against a remembered description.

def test_axis_filters_match_the_v2_card_exactly(pin):
    """The v1 filter map must name the same filter per axis as the v2 card.

    Read out of the v2 source rather than restated, so drift on either side fails
    here instead of silently skewing a reference.
    """
    import re
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "v2ecoli" / "workflow" / "analysis.py").read_text(encoding="utf-8")
    # e.g.  "doubling_time": _stat(_lab("division_time", _divided)),
    found = dict(re.findall(r'"(\w+)":\s*_stat\(_lab\("(?:\w+)",\s*_(\w+)\)\)', src))
    assert found, "could not read the v2 card's per-axis filters — did it move?"
    for axis, kind in pin._AXIS_FILTER.items():
        leaf = axis.split(".", 1)[1]
        assert leaf in found, f"{axis} has no counterpart in the v2 card"
        assert found[leaf] == kind, (
            f"{axis}: v1 filters '{kind}', v2 filters '{found[leaf]}' — the two "
            f"sides would select different cells")


def test_doubling_time_excludes_cells_that_never_divided(pin):
    """A non-divided cell's division_time is the duration cap, not a doubling time.

    v2 drops those; before this, the v1 reader had no `divided` field at all, so a
    capped cell entered the reference as though it were a fast divider. Latent on
    basal (every cell divides) and live on a condition where division stalls.
    """
    capped = {"divided": False}
    assert pin._keep_cell("divided", capped, 10800.0) is False
    assert pin._keep_cell("divided", {"divided": True}, 3000.0) is True
    # a zero/negative duration is not a doubling time either
    assert pin._keep_cell("divided", {"divided": True}, 0.0) is False


def test_missing_divided_signal_declines_to_filter_rather_than_dropping_all(pin):
    """`divided is None` means "cannot know", not "did not divide".

    A remote sweep has no daughter_states to read. Treating that as False would
    empty the axis; the reader must fall back to v2's other condition instead.
    """
    assert pin._keep_cell("divided", {"divided": None}, 3000.0) is True
    assert pin._keep_cell("divided", {"divided": None}, 0.0) is False


def test_pos_filter_drops_zero_valued_cells(pin):
    """v2 skips zero/absent levels; this reader's mean() yields 0.0, not None, for
    a cell with no valid timepoints — so `is not None` would have kept it."""
    assert pin._keep_cell("pos", {}, 0.0) is False
    assert pin._keep_cell("pos", {}, 0.42) is True
    # event-time axes keep a legitimate zero, and drop only None
    assert pin._keep_cell("any", {}, 0.0) is True
    assert pin._keep_cell("any", {}, None) is False


def test_divided_signal_is_absent_for_a_sweep_without_daughter_states(pin, tmp_path):
    """No daughter_states -> None (unknown), never an empty map (all-false)."""
    assert pin._divided_by_cell(str(tmp_path)) is None
    assert pin._divided_by_cell("s3://bucket/sweep") is None


def test_divided_signal_reads_daughter_state_directories(pin, tmp_path):
    """A cell divided iff it wrote daughter states; an empty agent dir does not."""
    root = tmp_path / "daughter_states" / "variant=0" / "seed=1" / "generation=4"
    (root / "agent_id=01").mkdir(parents=True)
    (root / "agent_id=01" / "daughter_state_0.json").write_text("{}")
    (root / "agent_id=01" / "daughter_state_1.json").write_text("{}")
    (root / "agent_id=02").mkdir(parents=True)          # ran, never divided
    got = pin._divided_by_cell(str(tmp_path))
    assert got == {(0, 1, 4, "01"): True}
    assert (0, 1, 4, "02") not in got


# --- 30S/50S subunit indices ---------------------------------------------------

def test_wrong_subunit_index_is_rejected_rather_than_silently_measured(
        pin, tmp_path, monkeypatch):
    """A wrong bulk index does not raise in DuckDB — list_extract returns NULL,
    which reads downstream as a zero count and collapses the active fraction to
    1.0. Wrong quietly, on all four ribosome axes. Resolve by id and refuse a
    disagreeing index."""
    class _FakeSD:
        class internal_state:
            class bulk_molecules:
                bulk_data = {"id": ["A[c]", "S30[c]", "B[c]", "S50[c]"]}

        class molecule_ids:
            s30_full_complex = "S30[c]"
            s50_full_complex = "S50[c]"

    kb = tmp_path / "parca" / "kb"
    kb.mkdir(parents=True)
    (kb / "simData.cPickle").write_bytes(b"placeholder")
    monkeypatch.setattr(pin, "open", lambda *a, **k: __import__("io").BytesIO(b""),
                        raising=False)
    import pickle as _p
    monkeypatch.setattr(_p, "load", lambda _f: _FakeSD)

    # resolves by id when no index is supplied
    assert pin._resolve_subunit_indices(str(tmp_path), None, None) == (1, 3)
    # accepts an index that agrees
    assert pin._resolve_subunit_indices(str(tmp_path), 1, 3) == (1, 3)
    # refuses one that does not
    with pytest.raises(SystemExit, match="does not name"):
        pin._resolve_subunit_indices(str(tmp_path), 1, 2)


def test_unresolvable_sim_data_warns_but_still_pins(pin, tmp_path, capsys):
    """No sim_data is a warning, not a failure — an s3 sweep still needs to pin."""
    got = pin._resolve_subunit_indices(str(tmp_path), 5456, 5464)
    assert got == (5456, 5464)
    assert "UNVALIDATED" in capsys.readouterr().out


def test_end_to_end_a_non_divided_cell_is_excluded_from_doubling_time_only(
        pin, tmp_path, monkeypatch):
    """The discriminating test: run the real pin over a sweep holding one cell
    that never divided, and assert it leaves the doubling-time axis but stays in
    the level axes.

    This is the test that fails against the pre-fix reader, which had no `divided`
    concept and so pinned the capped cell's duration as a doubling time.
    """
    sweep = _write_v1_sweep(tmp_path / "sweep", n_seeds=2, n_gens=5)
    # gen_lb=2 keeps gens 2,3,4 over 2 seeds = 6 cells; agent_id is "0"*(gen+1).
    kept = [(s, g) for s in range(2) for g in (2, 3, 4)]
    stalled = (1, 4)                     # this one wrote no daughter states
    for s, g in kept:
        if (s, g) == stalled:
            continue
        d = (tmp_path / "sweep" / "daughter_states" / "variant=0" /
             f"seed={s}" / f"generation={g}" / f"agent_id={'0' * (g + 1)}")
        d.mkdir(parents=True)
        (d / "daughter_state_0.json").write_text("{}")
        (d / "daughter_state_1.json").write_text("{}")

    ref = _run_pin(pin, sweep, tmp_path / "e2e.json", "basal", monkeypatch)
    axes = ref["axes"]
    n_doubling = len(axes["physiology.doubling_time"]["criterion"]["ref_values"])
    n_mass = len(axes["physiology.cell_mass"]["criterion"]["ref_values"])

    assert n_mass == len(kept), "a stalled cell still has a valid mass"
    assert n_doubling == len(kept) - 1, (
        "the non-divided cell must not contribute a doubling time "
        f"(got {n_doubling}, expected {len(kept) - 1})")
