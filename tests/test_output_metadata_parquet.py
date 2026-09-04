"""Tests: output_metadata walker wired into parquet + sqlite run paths.

Fast, config-level proof that self-describing labels flow through the two
default emitter backends without requiring the ParCa cache or a real
composite run.

Test level achieved: end-to-end parquet write + DuckDB config-partition
read-back (the actual config.pq is read and the names are verified via
``field_metadata()`` and by direct column query). SQLite is tested at the
``save_simulation_metadata`` + ``load_simulation_metadata`` level (the JSON
blob path).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("polars")
pytest.importorskip("viva_emitters")

import duckdb  # noqa: E402

from v2ecoli.library.parquet_run import run_multigen_parquet  # noqa: E402
from v2ecoli.library.sqlite_run import run_multigen_sqlite  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — stub composite with an annotated step instance
# ---------------------------------------------------------------------------

def _build_annotated_composite(monomer_ids: list[str]) -> object:
    """Build a minimal composite-like object whose state includes an annotated
    step instance (simulating CountsDeriver after initialize()).

    The composite:
      * registers a ``monomer_counts_vec`` LabeledArray type in its core
      * exposes a step instance in state whose ``outputs()`` returns the
        type-string ``"monomer_counts_vec"`` for the ``listeners.monomer_counts``
        port — exactly the pattern used by CountsDeriver
      * advances a per-agent ``count`` scalar on each ``run(n)``
    """
    from v2ecoli.core import build_core
    from v2ecoli.types.labeled_array import LabeledArray

    core = build_core()
    core.register_type(
        "monomer_counts_vec",
        LabeledArray(
            _shape=(len(monomer_ids),),
            _data=np.dtype("int64"),
            _labels=tuple(monomer_ids),
        ),
    )

    class _AnnotatedStep:
        """Minimal step that reports monomer_counts as a LabeledArray type."""

        def __init__(self, the_core):
            self.core = the_core

        def outputs(self):
            return {"listeners": {"monomer_counts": "monomer_counts_vec"}}

    step_instance = _AnnotatedStep(core)

    class _AnnotatedComposite:
        def __init__(self):
            self.core = core
            self.state = {
                "agents": {
                    "0": {
                        "counts_deriver": {
                            "instance": step_instance,
                            "outputs": {"listeners": ["listeners"]},
                        },
                        "listeners": {
                            "monomer_counts": np.zeros(len(monomer_ids), dtype=np.int64),
                            "mass": {"cell_mass": 1.0},
                        },
                    }
                }
            }
            self._tick = 0

        def run(self, n: int) -> None:
            self._tick += n
            for aid in self.state["agents"]:
                self.state["agents"][aid]["listeners"]["mass"]["cell_mass"] += 0.1 * n

        def find_instance_paths(self, *_a, **_kw) -> None:
            pass

    return _AnnotatedComposite()


# ---------------------------------------------------------------------------
# Test 1: output_metadata ends up in config.pq (parquet path, fast)
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_output_metadata_in_parquet_config(tmp_path):
    """run_multigen_parquet writes monomer names into the config parquet.

    This is an end-to-end parquet read-back test at the config-partition
    level. No ParCa cache needed — uses a stub composite with an annotated
    step instance.

    Verification path:
      1. Run 10 steps (no division) against the annotated stub composite.
      2. Open the resulting config.pq with DuckDB.
      3. Assert the ``output_metadata__listeners__monomer_counts`` column
         exists and contains the expected monomer IDs.
      4. Also verify via ``field_metadata()`` (the canonical reader API).
    """
    monomer_ids = ["MONA[c]", "MONB[c]", "MONC[c]"]
    comp = _build_annotated_composite(monomer_ids)

    result = run_multigen_parquet(
        comp,
        experiment_id="test-meta",
        out_dir=tmp_path / "out",
        emit_paths=["listeners/mass/cell_mass"],
        max_steps=10,
        max_generations=1,
        chunk=5,
        initial_agent_id="0",
        batch_size=2,
        threaded=False,
        core=comp.core,
    )
    assert result["steps"] == 10

    # Locate the config.pq
    config_pq = (
        tmp_path / "out" / "test-meta" / "configuration"
        / "experiment_id=test-meta" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0" / "config.pq"
    )
    assert config_pq.is_file(), f"config.pq not found at {config_pq}"

    # Read via DuckDB and verify the column exists + contains the names.
    conn = duckdb.connect(":memory:")
    config_sql = f"SELECT * FROM read_parquet('{config_pq}')"

    # Check column presence (the flatten_dict key for nested output_metadata)
    cols = [col[0] for col in conn.sql(f"DESCRIBE ({config_sql})").fetchall()]
    expected_col = "output_metadata__listeners__monomer_counts"
    assert expected_col in cols, (
        f"Column '{expected_col}' missing from config.pq. "
        f"Columns present: {[c for c in cols if 'output' in c.lower() or 'metadata' in c.lower()]}"
    )

    # Verify the stored names match via direct query.
    raw_names = conn.sql(
        f'SELECT "{expected_col}" FROM ({config_sql})'
    ).fetchone()[0]
    assert list(raw_names) == monomer_ids, (
        f"Names mismatch: expected {monomer_ids}, got {list(raw_names)}"
    )

    # Also verify via the canonical field_metadata() reader API.
    from viva_emitters import field_metadata
    recovered = field_metadata(conn, config_sql, "listeners__monomer_counts")
    assert recovered == monomer_ids, (
        f"field_metadata recovered {recovered!r}, expected {monomer_ids!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: unannotated composite → no output_metadata key (backward-compat)
# ---------------------------------------------------------------------------

class _PlainStubComposite:
    """Minimal composite with NO annotated step instances — backward-compat check."""

    def __init__(self, core, initial_agent_id: str = "0"):
        self.core = core
        self.state = {
            "agents": {
                initial_agent_id: {
                    "listeners": {"mass": {"cell_mass": 1.0}, "count": 0},
                },
            },
        }
        self._tick = 0

    def run(self, n: int) -> None:
        self._tick += n
        for aid in self.state["agents"]:
            self.state["agents"][aid]["listeners"]["count"] += n
            self.state["agents"][aid]["listeners"]["mass"]["cell_mass"] += 0.1 * n

    def find_instance_paths(self, *_a, **_kw) -> None:
        pass


@pytest.mark.fast
def test_output_metadata_absent_for_unannotated_composite(tmp_path):
    """Stub composite with no annotated steps → config.pq has no output_metadata__ cols."""
    from bigraph_schema import allocate_core

    core = allocate_core()
    comp = _PlainStubComposite(core, initial_agent_id="0")

    run_multigen_parquet(
        comp,
        experiment_id="no-meta",
        out_dir=tmp_path / "out",
        emit_paths=["listeners/mass/cell_mass", "listeners/count"],
        max_steps=10,
        max_generations=1,
        chunk=5,
        initial_agent_id="0",
        batch_size=2,
        threaded=False,
        core=core,
    )

    config_pq = (
        tmp_path / "out" / "no-meta" / "configuration"
        / "experiment_id=no-meta" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0" / "config.pq"
    )
    assert config_pq.is_file()

    conn = duckdb.connect(":memory:")
    cols = [
        col[0]
        for col in conn.sql(
            f"DESCRIBE (SELECT * FROM read_parquet('{config_pq}'))"
        ).fetchall()
    ]
    output_meta_cols = [c for c in cols if c.startswith("output_metadata__")]
    assert output_meta_cols == [], (
        f"Expected no output_metadata__ columns for unannotated composite; got: {output_meta_cols}"
    )


# ---------------------------------------------------------------------------
# Test 3: sqlite path — output_metadata stored in simulations.metadata JSON
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_output_metadata_in_sqlite_simulations(tmp_path):
    """run_multigen_sqlite writes output_metadata into simulations.metadata JSON.

    SQLite stores the catalog as a JSON blob (not individual queryable
    columns like the parquet path). Recoverable via load_simulation_metadata.
    """
    monomer_ids = ["MONA[c]", "MONB[c]", "MONC[c]"]
    comp = _build_annotated_composite(monomer_ids)

    db_file = tmp_path / "sim.db"

    run_multigen_sqlite(
        comp,
        run_id="run-meta-test",
        db_file=db_file,
        emit_paths=["listeners/mass/cell_mass"],
        max_steps=10,
        chunk=5,
        initial_agent_id="0",
        core=comp.core,
    )

    from viva_emitters.sqlite_emitter import load_simulation_metadata
    row = load_simulation_metadata(db_file, "run-meta-test")
    assert row is not None, "No simulations row found"
    meta = row.get("metadata")
    assert meta is not None, f"metadata column is null; row={row}"
    om = meta.get("output_metadata")
    assert om is not None, f"output_metadata key missing from metadata JSON; meta={meta}"
    recovered = om.get("listeners", {}).get("monomer_counts")
    assert recovered == monomer_ids, (
        f"SQLite output_metadata mismatch: expected {monomer_ids}, got {recovered}"
    )


# ---------------------------------------------------------------------------
# Test 4: config-level golden — real composite carries names (requires cache)
# ---------------------------------------------------------------------------

@pytest.mark.sim
def test_golden_output_metadata_in_parquet_config_real_composite(tmp_path, sim_data_cache):
    """End-to-end with a real baseline composite: monomer names flow into config.pq.

    Skipped when out/cache is absent. Uses the symlinked cache read-only
    (tmp_path for writes). Runs 2 steps — just enough to trigger emitter
    construction and config write, not a full-gen run.
    """
    from v2ecoli import build_composite
    from viva_emitters import field_metadata

    comp = build_composite("ecoli_baseline", seed=0, cache_dir=sim_data_cache)

    result = run_multigen_parquet(
        comp,
        experiment_id="golden-meta",
        out_dir=tmp_path / "out",
        emit_paths=["listeners/mass/cell_mass"],
        max_steps=2,
        max_generations=1,
        chunk=2,
        initial_agent_id="0",
        batch_size=10,
        threaded=False,
        core=comp.core,
    )
    assert result["steps"] == 2

    config_pq = (
        tmp_path / "out" / "golden-meta" / "configuration"
        / "experiment_id=golden-meta" / "variant=0" / "lineage_seed=0"
        / "generation=1" / "agent_id=0" / "config.pq"
    )
    assert config_pq.is_file(), f"config.pq not written: {config_pq}"

    conn = duckdb.connect(":memory:")
    config_sql = f"SELECT * FROM read_parquet('{config_pq}')"
    recovered = field_metadata(conn, config_sql, "listeners__monomer_counts")
    assert isinstance(recovered, list) and len(recovered) > 0, (
        f"Expected non-empty monomer name list; got: {recovered!r}"
    )
    assert any("[" in n for n in recovered), (
        f"Expected bracket-containing IDs (e.g. 'MONA[c]'); sample: {recovered[:3]!r}"
    )
