"""Tests for v2ecoli.library.output_metadata (self-describing emitters).

TDD for sub-project #1 (readout-coordination): make v2ecoli runs self-describing
by annotating listener outputs() with element names and harvesting them via a
registry-based walker.

Approach: labels are carried as type-level metadata on LabeledArray (a thin
Array subtype). Each listener registers a named vec-type (e.g.
'monomer_counts_vec') in initialize() so output_metadata() can recover element
labels via core.access(type_name)._labels, without touching per-tick logic.
"""
import pytest


# ---------------------------------------------------------------------------
# Step 2a: LabeledArray type — structural checks
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_labeled_array_is_array_subtype():
    """LabeledArray is a subclass of bigraph_schema Array."""
    from bigraph_schema.schema import Array
    from v2ecoli.types.labeled_array import LabeledArray

    assert issubclass(LabeledArray, Array)


@pytest.mark.fast
def test_labeled_array_labels_not_in_schema_keys():
    """_labels is absent from _schema_keys so the engine treats it as static metadata."""
    from v2ecoli.types.labeled_array import LabeledArray

    assert '_labels' not in LabeledArray._schema_keys, (
        "_labels must NOT be in _schema_keys; it should be invisible to per-tick ops"
    )


@pytest.mark.fast
def test_labeled_array_default_labels_empty():
    """LabeledArray() with no args has empty _labels tuple."""
    from v2ecoli.types.labeled_array import LabeledArray

    la = LabeledArray()
    assert la._labels == ()


@pytest.mark.fast
def test_labeled_array_stores_labels():
    """LabeledArray(_labels=(...)) stores the provided tuple."""
    import numpy as np
    from v2ecoli.types.labeled_array import LabeledArray

    labels = ('A', 'B', 'C')
    la = LabeledArray(_shape=(3,), _data=np.dtype('int64'), _labels=labels)
    assert la._labels == labels


# ---------------------------------------------------------------------------
# Step 2b: core registration — core.access('labeled_array') and named vec-types
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_core_access_labeled_array():
    """build_core() registers 'labeled_array'; core.access returns a LabeledArray."""
    from v2ecoli.core import build_core
    from v2ecoli.types.labeled_array import LabeledArray

    core = build_core()
    node = core.access('labeled_array')
    assert isinstance(node, LabeledArray), (
        "core.access('labeled_array') should return a LabeledArray, got: %r" % type(node)
    )


@pytest.mark.fast
def test_named_vec_type_roundtrips_labels():
    """Register a named LabeledArray instance; core.access preserves _labels."""
    import numpy as np
    from v2ecoli.core import build_core
    from v2ecoli.types.labeled_array import LabeledArray

    core = build_core()
    labels = ('MON-0', 'MON-1', 'MON-2')
    instance = LabeledArray(_shape=(3,), _data=np.dtype('int64'), _labels=labels)
    core.register_type('test_labeled_vec', instance)

    recovered = core.access('test_labeled_vec')
    assert isinstance(recovered, LabeledArray)
    assert recovered._labels == labels, (
        "Labels should survive round-trip through registry, got: %r" % (recovered._labels,)
    )


@pytest.mark.fast
def test_labeled_array_port_resolves_in_core():
    """core.access on a registered LabeledArray type name works without error."""
    import numpy as np
    from v2ecoli.core import build_core
    from v2ecoli.types.labeled_array import LabeledArray

    labels = tuple(f'MOL-{i}' for i in range(5))
    core = build_core()
    core.register_type('test_vec5', LabeledArray(
        _shape=(5,), _data=np.dtype('int64'), _labels=labels
    ))
    node = core.access('test_vec5')
    assert isinstance(node, LabeledArray)
    assert node._labels == labels


# ---------------------------------------------------------------------------
# Task 1: output_metadata() walker baseline (unchanged — legacy _properties path)
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
    """Walker with a step that has no labels → returns {}."""
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
def test_output_metadata_walker_legacy_properties_path():
    """Walker still finds _properties.metadata (legacy backward-compat path)."""
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

    state = {
        "some_step": {
            "instance": AnnotatedStep(),
            "outputs": {"listeners": ["listeners"]},
        }
    }
    result = output_metadata(state)
    assert result == {"listeners": {"monomer_counts": monomer_ids}}, (
        "Legacy _properties.metadata path broken; got: %r" % result
    )


@pytest.mark.fast
def test_output_metadata_walker_registry_string_port():
    """Walker extracts labels when port type is a registered LabeledArray name."""
    import numpy as np
    from v2ecoli.core import build_core
    from v2ecoli.types.labeled_array import LabeledArray
    from v2ecoli.library.output_metadata import output_metadata

    monomer_ids = ("MONA_[c]", "MONB_[c]", "MONC_[c]")
    core = build_core()
    core.register_type('test_mon_vec', LabeledArray(
        _shape=(3,), _data=np.dtype('int64'), _labels=monomer_ids
    ))

    class RegistryStep:
        def __init__(self, core):
            self.core = core

        def outputs(self):
            return {"listeners": {"monomer_counts": "test_mon_vec"}}

    state = {
        "some_step": {
            "instance": RegistryStep(core),
            "outputs": {"listeners": ["listeners"]},
        }
    }
    result = output_metadata(state)
    assert result.get("listeners", {}).get("monomer_counts") == list(monomer_ids), (
        "Walker should recover labels from registry; got: %r" % result
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
# Step 3: CountsDeriver annotates monomer_counts via registry approach
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_counts_deriver_outputs_monomer_counts_is_string_type():
    """CountsDeriver.outputs() declares monomer_counts as 'monomer_counts_vec' (a string type name)."""
    from v2ecoli.steps.derivers.counts_deriver import CountsDeriver

    monomer_ids = ["MONA_[c]", "MONB_[c]", "MONC_[c]"]

    instance = CountsDeriver.__new__(CountsDeriver)
    instance.n_monomers = len(monomer_ids)
    instance.monomer_ids = monomer_ids
    instance.n_mRNA_TU = 2
    instance.n_mRNA_cistron = 2
    instance.n_rRNA_TU = 1
    instance.n_rRNA_cistron = 1

    schema = instance.outputs()
    monomer_schema = schema["listeners"]["monomer_counts"]
    assert monomer_schema == 'monomer_counts_vec', (
        "monomer_counts should be the string 'monomer_counts_vec'; got: %r" % monomer_schema
    )


@pytest.mark.fast
def test_output_metadata_walker_with_counts_deriver():
    """Walker returns monomer_ids at listeners.monomer_counts for a CountsDeriver instance."""
    import numpy as np
    from v2ecoli.core import build_core
    from v2ecoli.types.labeled_array import LabeledArray
    from v2ecoli.steps.derivers.counts_deriver import CountsDeriver
    from v2ecoli.library.output_metadata import output_metadata

    monomer_ids = ("MONA_[c]", "MONB_[c]")

    # Build a real core and register monomer_counts_vec (simulating initialize()).
    core = build_core()
    core.register_type('monomer_counts_vec', LabeledArray(
        _shape=(len(monomer_ids),), _data=np.dtype('int64'), _labels=tuple(monomer_ids)
    ))

    # Create a minimal CountsDeriver instance (bypassing full __init__).
    instance = CountsDeriver.__new__(CountsDeriver)
    instance.core = core  # walker uses instance.core
    instance.n_monomers = len(monomer_ids)
    instance.monomer_ids = list(monomer_ids)
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
    assert result.get("listeners", {}).get("monomer_counts") == list(monomer_ids), (
        "Walker should recover monomer_ids at listeners.monomer_counts; got: %r" % result
    )


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
