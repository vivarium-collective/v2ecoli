"""Tests for v2ecoli.library.output_metadata (self-describing emitters).

TDD for sub-project #1 (readout-coordination): make v2ecoli runs self-describing
by annotating listener outputs() with element names and harvesting them via a
walker that mirrors vEcoli's EcoliSim.output_metadata() pattern.
"""
import pytest


# ---------------------------------------------------------------------------
# Task 1: output_metadata() walker baseline (before any annotations)
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_extract_metadata_no_annotation():
    """Schema without _properties.metadata → extract_metadata returns None."""
    from v2ecoli.library.output_metadata import extract_metadata

    schema = {
        "monomer_counts": {"_type": "array[integer]", "_default": []}
    }
    result = extract_metadata(schema)
    assert result is None, (
        "Un-annotated schema should return None, got: %r" % result
    )


@pytest.mark.fast
def test_extract_metadata_with_annotation():
    """Nested schema with _properties.metadata → extract_metadata returns the names."""
    from v2ecoli.library.output_metadata import extract_metadata

    names = ["MONOMER_A", "MONOMER_B"]
    schema = {
        "listeners": {
            "monomer_counts": {
                "_type": "array[integer]",
                "_default": [],
                "_properties": {"metadata": names},
            }
        }
    }
    result = extract_metadata(schema)
    assert result == {"listeners": {"monomer_counts": names}}, (
        "Expected listener-nested metadata dict, got: %r" % result
    )


@pytest.mark.fast
def test_extract_metadata_top_level_properties():
    """Schema where _properties.metadata is at the leaf → returns metadata directly."""
    from v2ecoli.library.output_metadata import extract_metadata

    names = ["X", "Y"]
    schema = {
        "_type": "array[integer]",
        "_default": [],
        "_properties": {"metadata": names},
    }
    result = extract_metadata(schema)
    assert result == names


@pytest.mark.fast
def test_output_metadata_empty_state_returns_empty():
    """output_metadata({}) returns {} without crashing."""
    from v2ecoli.library.output_metadata import output_metadata

    assert output_metadata({}) == {}


@pytest.mark.fast
def test_output_metadata_no_instances_returns_empty():
    """State with data but no step instances → returns {}."""
    from v2ecoli.library.output_metadata import output_metadata

    state = {
        "listeners": {"monomer_counts": []},
        "global_time": 0.0,
        "bulk": [],
    }
    assert output_metadata(state) == {}


@pytest.mark.fast
def test_output_metadata_walker_unannotated_step():
    """Walker with a step that has no _properties.metadata → returns {}."""
    from v2ecoli.library.output_metadata import output_metadata

    class UnannotatedStep:
        def outputs(self):
            return {
                "listeners": {
                    "monomer_counts": {"_type": "array[integer]", "_default": []}
                }
            }

    state = {
        "some_step": {
            "instance": UnannotatedStep(),
            "outputs": {"listeners": ["listeners"]},
        }
    }
    assert output_metadata(state) == {}


@pytest.mark.fast
def test_output_metadata_walker_annotated_step():
    """Walker finds _properties.metadata in a step outputs() and remaps to store path."""
    from v2ecoli.library.output_metadata import output_metadata

    monomer_ids = ["MONOMER_A", "MONOMER_B", "MONOMER_C"]

    class AnnotatedStep:
        def outputs(self):
            return {
                "listeners": {
                    "monomer_counts": {
                        "_type": "array[integer]",
                        "_default": [],
                        "_properties": {"metadata": monomer_ids},
                    }
                }
            }

    # Wiring: port 'listeners' → store path ['listeners']
    state = {
        "some_step": {
            "instance": AnnotatedStep(),
            "outputs": {"listeners": ["listeners"]},
        }
    }
    result = output_metadata(state)
    assert result == {"listeners": {"monomer_counts": monomer_ids}}, (
        "Expected monomer names at listeners.monomer_counts, got: %r" % result
    )


@pytest.mark.fast
def test_output_metadata_nested_state():
    """Walker recurses into nested state (e.g. agents.0.step_name)."""
    from v2ecoli.library.output_metadata import output_metadata

    names = ["A", "B"]

    class AnnotatedStep:
        def outputs(self):
            return {
                "listeners": {
                    "monomer_counts": {
                        "_type": "array[integer]",
                        "_default": [],
                        "_properties": {"metadata": names},
                    }
                }
            }

    state = {
        "agents": {
            "0": {
                "some_step": {
                    "instance": AnnotatedStep(),
                    "outputs": {"listeners": ["listeners"]},
                }
            }
        }
    }
    result = output_metadata(state)
    assert result == {"listeners": {"monomer_counts": names}}


# ---------------------------------------------------------------------------
# Task 2: CountsDeriver.outputs() carries _properties.metadata for monomer_counts
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_counts_deriver_outputs_carries_monomer_ids_metadata():
    """After annotation, CountsDeriver.outputs() has _properties.metadata for monomer_counts."""
    from v2ecoli.steps.derivers.counts_deriver import CountsDeriver

    monomer_ids = ["MONA_[c]", "MONB_[c]", "MONC_[c]"]

    # Build a minimal instance with just the attrs outputs() needs.
    instance = CountsDeriver.__new__(CountsDeriver)
    instance.n_monomers = len(monomer_ids)
    instance.monomer_ids = monomer_ids
    instance.n_mRNA_TU = 2
    instance.n_mRNA_cistron = 2
    instance.n_rRNA_TU = 1
    instance.n_rRNA_cistron = 1

    schema = instance.outputs()
    monomer_schema = schema["listeners"]["monomer_counts"]
    props = monomer_schema.get("_properties", {})
    assert "metadata" in props, (
        "monomer_counts schema missing _properties.metadata; schema=%r" % monomer_schema
    )
    assert props["metadata"] == monomer_ids


@pytest.mark.fast
def test_output_metadata_walker_with_counts_deriver():
    """Walker returns monomer_ids at listeners.monomer_counts for a CountsDeriver instance."""
    from v2ecoli.steps.derivers.counts_deriver import CountsDeriver
    from v2ecoli.library.output_metadata import output_metadata

    monomer_ids = ["MONA_[c]", "MONB_[c]"]

    instance = CountsDeriver.__new__(CountsDeriver)
    instance.n_monomers = len(monomer_ids)
    instance.monomer_ids = monomer_ids
    instance.n_mRNA_TU = 1
    instance.n_mRNA_cistron = 1
    instance.n_rRNA_TU = 1
    instance.n_rRNA_cistron = 1

    state = {
        "counts_deriver": {
            "instance": instance,
            "outputs": {"listeners": ["listeners"]},
        }
    }
    result = output_metadata(state)
    assert result.get("listeners", {}).get("monomer_counts") == monomer_ids


# ---------------------------------------------------------------------------
# Task 3: extract_output_metadata_from_state uses names when available
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_extract_output_metadata_prefers_names_over_range():
    """extract_output_metadata_from_state uses named_metadata instead of range(N)."""
    from v2ecoli.library.xarray_run import extract_output_metadata_from_state
    import numpy as np

    monomer_ids = ["MONA", "MONB", "MONC"]

    # Minimal state: listeners.monomer_counts = array of length 3
    state = {
        "listeners": {
            "monomer_counts": np.zeros(3, dtype=int),
        }
    }
    view = [{"root": ("listeners",), "variables": {
        "monomer_counts": [{"path": "monomer_counts", "dtype": "<f8"}]
    }}]

    named_metadata = {"listeners": {"monomer_counts": monomer_ids}}
    result = extract_output_metadata_from_state(state, view, named_metadata=named_metadata)
    assert result.get("monomer_counts") == monomer_ids, (
        "Expected named coord for monomer_counts, got: %r" % result
    )


@pytest.mark.fast
def test_extract_output_metadata_fallback_to_range():
    """extract_output_metadata_from_state falls back to range(N) for un-annotated vectors."""
    from v2ecoli.library.xarray_run import extract_output_metadata_from_state
    import numpy as np

    state = {
        "listeners": {
            "monomer_counts": np.zeros(3, dtype=int),
        }
    }
    view = [{"root": ("listeners",), "variables": {
        "monomer_counts": [{"path": "monomer_counts", "dtype": "<f8"}]
    }}]

    # No named_metadata for monomer_counts → falls back to [0, 1, 2]
    result = extract_output_metadata_from_state(state, view)
    assert result.get("monomer_counts") == [0, 1, 2]


# ---------------------------------------------------------------------------
# Task 4: Config-level golden — names reach the built emitter config
# ---------------------------------------------------------------------------

@pytest.mark.sim
def test_golden_monomer_names_in_output_metadata():
    """Config-level golden: output_metadata(composite_state) carries monomer_ids at
    listeners.monomer_counts after CountsDeriver annotation.

    Requires out/cache (the simulation cache). Uses the existing cache; never
    rebuilds ParCa.
    """
    import os
    if not os.path.isdir("out/cache"):
        pytest.skip("out/cache absent; build via scripts/build_cache.py")

    from v2ecoli import build_composite
    from v2ecoli.library.output_metadata import output_metadata

    comp = build_composite("baseline", seed=0, cache_dir="out/cache")
    result = output_metadata(comp.state)

    # The names live at listeners.monomer_counts (store path from CountsDeriver.topology)
    assert "listeners" in result, "output_metadata missing 'listeners' key"
    assert "monomer_counts" in result["listeners"], (
        "output_metadata['listeners'] missing 'monomer_counts'"
    )
    names = result["listeners"]["monomer_counts"]
    assert isinstance(names, list) and len(names) > 0, (
        "monomer_counts names should be a non-empty list, got: %r" % names
    )
    # Spot-check: monomer IDs are molecular IDs like 'MONA_CPLX_RNA_[c]'
    assert any("[" in n for n in names), (
        "Expected bracket-containing monomer IDs (e.g. 'MONA[c]'), got sample: %r" % names[:3]
    )
