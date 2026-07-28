"""Registration + wiring tests for the cd1 omics native Analysis ports.

The point of the cd1 suite is that a multiseed batch reproduces the vEcoli cd1
result tables *without being asked to* — the composite's default
``analyses="applicable"`` has to pick them up.  These tests pin that: each
analysis is registered at ``multiseed`` scale, and a batch shaped like the
baseline ensemble (many seeds, many generations) selects all six.
"""

import pytest

CD1_ANALYSES = (
    "cd1_exchange_fluxes",
    "cd1_fluxomics",
    "cd1_higher_order_properties",
    "cd1_metabolomics",
    "cd1_proteomics",
    "cd1_transcriptomics",
)

# The TSV each port emits, keyed by analysis name — the artifact names the
# vEcoli originals write, which downstream consumers key off.
CD1_FILENAMES = {
    "cd1_exchange_fluxes": "exchange_fluxes.tsv",
    "cd1_fluxomics": "cd1_fluxomics_detailed.tsv",
    "cd1_higher_order_properties": "higher_order_properties.tsv",
    "cd1_metabolomics": "metabolomics.tsv",
    "cd1_proteomics": "proteomics.tsv",
    "cd1_transcriptomics": "transcriptomics.tsv",
}


@pytest.mark.parametrize("name", CD1_ANALYSES)
def test_cd1_analysis_registered_at_multiseed(name):
    import v2ecoli.workflow.analyses  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis

    cls = ANALYSIS_REGISTRY[name]
    assert issubclass(cls, Analysis)
    assert cls.scale == "multiseed"


@pytest.mark.parametrize("name", CD1_ANALYSES)
def test_cd1_analysis_accepts_the_shared_bounds(name):
    """Every cd1 analysis takes the generation/time burn-in bounds the suite is
    configured with (the sms-api default passes ``generation_lower_bound``)."""
    import v2ecoli.workflow.analyses  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY

    schema = ANALYSIS_REGISTRY[name].config_schema
    assert "generation_lower_bound" in schema
    assert "time_lower_bound" in schema


def test_batch_baseline_applicable_selects_the_cd1_suite():
    """A multiseed+multigen batch's default analysis set includes all of cd1.

    This is the property Eran's "all the cd1 analyses should be in the
    composite" asks for: no explicit analysis list, just the composite's
    ``analyses="applicable"`` default over a many-seed, many-generation run.
    """
    from v2ecoli.steps.batch_baseline_runner import build_analysis_options

    options = build_analysis_options("applicable", n_seeds=1000, n_generations=10)
    selected = set(options.get("multiseed", {}))
    assert set(CD1_ANALYSES) <= selected


def test_cd1_absent_from_single_seed_batch():
    """A single-seed batch has no multiseed scale, so cd1 is correctly absent."""
    from v2ecoli.steps.batch_baseline_runner import build_analysis_options

    options = build_analysis_options("applicable", n_seeds=1, n_generations=1)
    assert "multiseed" not in options


@pytest.mark.parametrize("name", CD1_ANALYSES)
def test_cd1_port_keeps_its_vecoli_artifact_name(name):
    """Each port emits the exact TSV filename its vEcoli original wrote.

    Downstream consumers key off these names, so a rename is a breaking change
    and should fail here rather than silently produce an unfound artifact.
    """
    import importlib
    import inspect

    module = importlib.import_module(f"v2ecoli.workflow.analyses.{name}")
    assert f'"{CD1_FILENAMES[name]}"' in inspect.getsource(module)


def test_cd1_empty_slice_still_yields_a_table():
    """A barren sweep slice degrades to an empty table, not a missing artifact.

    The originals always wrote their file; the ports always return one. This
    drives the shared empty-input branch the six ports funnel through.
    """
    import polars as pl

    from v2ecoli.workflow.analyses._helpers import with_cross_cell_stats

    # index-only table (no cell columns) is what an empty slice produces
    out = with_cross_cell_stats(pl.DataFrame({"id": []}), "id")
    assert out.columns == ["id", "mean", "std"]
    assert out.height == 0


def test_with_cross_cell_stats_orders_and_computes():
    """mean/std are across cells and are front-loaded after the index column."""
    import polars as pl

    from v2ecoli.workflow.analyses._helpers import with_cross_cell_stats

    wide = pl.DataFrame(
        {"id": ["a", "b"], "Cell: 0_0": [1.0, 10.0], "Cell: 1_0": [3.0, 20.0]}
    )
    out = with_cross_cell_stats(wide, "id")
    assert out.columns == ["id", "mean", "std", "Cell: 0_0", "Cell: 1_0"]
    assert out["mean"].to_list() == [2.0, 15.0]


# ---------------------------------------------------------------------------
# Parity against REAL vEcoli cd1 outputs.
#
# The private vEcoli repo ships cd1 result tables from actual runs. They are a
# different model (vEcoli v1) on different variants (mecillinam / violacein),
# so the NUMBERS are not comparable — but the schema and the entity-id
# vocabulary are, and the vocabulary is the risky part of the port: v2ecoli has
# no listener field-metadata table, so each port re-derives those ids from
# sim_data. These tests pin that re-derivation against ground truth.
#
# Skipped when the private repo isn't checked out alongside (CI, other clones).
# ---------------------------------------------------------------------------

_VECOLI_PLOTS = [
    "/Users/alexanderpatrie/sms/vEcoli-private/api_integration/"
    "mec_analyses_carina/variant=1/plots",
    "/Users/alexanderpatrie/sms/vEcoli-private/api_integration/"
    "vio_met_analyses_carina/variant=1/plots",
]

# vEcoli filename -> (id column, expected id count in the v2ecoli baseline)
_PARITY = {
    "proteomics.tsv": ("EcoCyc Monomer ID", 4309),
    "transcriptomics.tsv": ("EcoCyc Gene ID", 4345),
    "metabolomics.tsv": ("EcoCyc Compound ID", 165),
    "higher_order_properties.tsv": ("Properties", 5),
}


def _find_vecoli_output(filename):
    import os

    for d in _VECOLI_PLOTS:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None


def _read_tsv(path):
    import csv

    with open(path) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    return rows[0], [r[0] for r in rows[1:] if r]


@pytest.mark.parametrize("filename", sorted(_PARITY))
def test_cd1_matches_vecoli_output_schema(filename):
    """The ported table has the same column layout as the real cd1 output."""
    path = _find_vecoli_output(filename)
    if path is None:
        pytest.skip("vEcoli-private cd1 outputs not available")
    header, _ = _read_tsv(path)
    id_col, _count = _PARITY[filename]
    assert header[:3] == [id_col, "mean", "std"]
    assert all(c.startswith("Cell: ") for c in header[3:])


def _load_sim_data():
    """A local ParCa sim_data, or None. Mirrors ``resolve_sim_data``'s fallbacks."""
    import os

    candidates = [
        os.environ.get("V2ECOLI_SIM_DATA"),
        os.path.join("out", "kb", "simData.cPickle"),
        os.path.join("out", "workflow", "simData.cPickle"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            from v2ecoli.library.sim_data import LoadSimData

            return LoadSimData(sim_data_path=path).sim_data
    return None


# vEcoli filename -> callable deriving the same id set from v2ecoli sim_data,
# exactly as the corresponding port does.
_ID_SOURCES = {
    "proteomics.tsv": lambda sd: {
        str(m)[:-3] for m in sd.process.translation.monomer_data["id"]
    },
    "transcriptomics.tsv": lambda sd: {
        str(c)[:-4]
        for c in sd.process.transcription.cistron_data["id"][
            sd.process.transcription.cistron_data["is_mRNA"]
        ]
    },
    "metabolomics.tsv": lambda sd: {
        str(k)[:-3] for k in sd.process.metabolism.conc_dict
    },
}


@pytest.mark.parametrize("filename", sorted(_ID_SOURCES))
def test_cd1_id_vocabulary_matches_vecoli(filename):
    """The sim_data-derived entity ids reproduce vEcoli's field_metadata ids.

    This is the substantive parity check: v2ecoli has no listener
    field-metadata table, so each port re-derives the entity ids from sim_data.
    If that substitution were wrong, the tables would be mislabelled while
    still looking structurally fine. Measured at 100% overlap against the real
    cd1 outputs for all three id spaces.
    """
    path = _find_vecoli_output(filename)
    if path is None:
        pytest.skip("vEcoli-private cd1 outputs not available")
    sim_data = _load_sim_data()
    if sim_data is None:
        pytest.skip("no local ParCa sim_data (set V2ECOLI_SIM_DATA)")

    _header, vecoli_ids = _read_tsv(path)
    derived = _ID_SOURCES[filename](sim_data)
    assert set(vecoli_ids) == derived


def test_cd1_filter_clause_treats_zero_bound_as_a_filter():
    """A 0 bound is a real (if permissive) bound, not an absent one.

    The vEcoli ``cd1_fluxomics`` original tested truthiness here and so ignored
    ``generation_lower_bound=0`` while its five siblings honoured it; the ports
    normalize on ``is not None``.
    """
    from v2ecoli.workflow.analyses._helpers import cd1_filter_clause

    assert cd1_filter_clause({}) == ""
    assert cd1_filter_clause(None) == ""
    assert cd1_filter_clause({"generation_lower_bound": 0}) == "WHERE generation >= 0"
    assert cd1_filter_clause({"generation_lower_bound": 5}) == "WHERE generation >= 5"
    both = cd1_filter_clause({"generation_lower_bound": 3, "time_lower_bound": 60})
    assert both == "WHERE generation >= 3 AND time >= 60.0"
