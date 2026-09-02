"""Regression tests for the drug-agnostic injection seam (PR #629 fix).

Two silent dose-drops the pbg_native injection branch used to have, plus the
list-shaped-timeline mangling on the multigeneration path:

1. A config-declared ``initial_state`` for a native process was never applied
   (the ``vivarium_1`` branch overlays it; the ``pbg_native`` branch did not).
2. A freshly-created native store (e.g. ``fields``) was made as a bare ``{}``,
   which bigraph-schema infers as a plain ADDITIVE map that then clobbers a
   Step's ``overwrite[...]`` output — so a wrapped Step's write never composes
   forward to a downstream reader (the field-delivery chain
   ``field_timeline`` -> ``fields`` -> ``well_mixed_field`` -> ``boundary``).
3. ``LineageProcess.config_schema['injected_processes']`` carried no ``_type``,
   so a list-shaped fork ``timeline`` (``[[100, {"drug": 1.0}]]``) was mangled
   to ``[[100, 100]]`` when the generation-1 composite re-realized the config.

These mirror the ``sulfadiazine_native_run.json`` ``_timeline_note`` diagnosis.
"""
import numpy as np

from scripts._compare import inject
from v2ecoli.library.ecoli_step import EcoliStep


# --------------------------------------------------------------------------- #
# Minimal pbg-native steps standing in for field_timeline / well_mixed_field.
# They define inputs()/outputs()/update() directly, so classify_process() ->
# "pbg_native", exactly like the real sms_modules antibiotic-chain steps.
# --------------------------------------------------------------------------- #
class _FieldWriter(EcoliStep):
    """field_timeline analog: writes a dose onto the shared ``fields`` store via
    a ``map[overwrite[array[float]]]`` output (SET semantics)."""

    name = "field_writer_test"
    config_schema = {"dose": {"_type": "float", "_default": 0.0}}

    def inputs(self):
        return {"global_time": {"_type": "float", "_default": 0.0}}

    def outputs(self):
        return {"fields": "map[overwrite[array[float]]]"}

    def update(self, states, interval=None):
        return {"fields": {"drug": np.full((1, 1), self.parameters["dose"],
                                           dtype=np.float64)}}


class _BoundaryWriter(EcoliStep):
    """A native step whose nested ``boundary`` output wires to the shared
    (baseline-owned) ``boundary`` store — used to check protected-root skip."""

    name = "boundary_writer_test"

    def inputs(self):
        return {"fields": "map[overwrite[array[float]]]"}

    def outputs(self):
        return {"boundary": {"external": "map[overwrite[float[mM]]]"}}

    def update(self, states, interval=None):
        return {}


class _BoundaryReader(EcoliStep):
    """antibiotic-transport analog: reads the per-drug dose from the SHARED
    ``boundary.external`` store (exactly as AntibioticTransportOdeint reads
    ``boundary.external[<drug>]`` to set its ``external`` species boundary
    condition)."""

    name = "boundary_reader_test"

    def inputs(self):
        return {"boundary": {"external": "map[overwrite[float[mM]]]"}}

    def outputs(self):
        return {"witness": "map[overwrite[float]]"}

    def update(self, states, interval=None):
        ext = states.get("boundary", {}).get("external", {}) or {}
        got = ext.get("drug")
        if got is None:
            return {}
        val = float(got.magnitude) if hasattr(got, "magnitude") else float(got)
        return {"witness": {"drug": val}}


def _spec(cls, name, topology, initial_state=None, config=None, as_step=True):
    """Build a resolved pbg_native InjectionSpec by hand (resolve_injections
    needs a real fork registry; apply_injected_processes takes specs directly)."""
    inject._fork_class_cache[(cls.__module__, cls.__qualname__)] = cls
    return {
        "name": name,
        "module": cls.__module__,
        "qualname": cls.__qualname__,
        "kind": "pbg_native",
        "as_step": as_step,
        "config": config,
        "topology": topology,
        "interval": 1.0,
        "initial_state": initial_state or {},
    }


# --------------------------------------------------------------------------- #
# Defect 2 — the store type. A Step's overwrite output must compose forward.
# --------------------------------------------------------------------------- #
def test_native_overwrite_store_survives_realization():
    """The freshly-created ``fields`` store must be typed as the native step's
    declared ``map[overwrite[array[float]]]`` output, so a value placed on it
    SURVIVES composite realization (ready to compose forward to a downstream
    reader). Under the pre-fix bare-``{}`` create, bigraph-schema infers a plain
    additive map and the value is dropped at build — the ``fields`` chain is
    silently inert. Seeding via a config ``initial_state`` and reading the store
    back off the built composite distinguishes the two: fixed -> the dose is
    present; buggy -> the store realizes empty."""
    from process_bigraph import Composite
    from v2ecoli.core import build_core

    core = build_core()
    spec = _spec(_FieldWriter, "field_writer_test",
                 {"global_time": ["global_time"], "fields": ["fields"]},
                 initial_state={"fields": {"drug": [[5.0]]}}, config={"dose": 0.0})
    cell_state = {"global_time": 0.0}
    inject.apply_injected_processes(cell_state, [], core, [spec])

    # Seam-level: the fresh `fields` store is typed as the writer's overwrite map
    # (NOT a bare {} that would infer additive and clobber the write forward).
    assert cell_state["fields"].get("_type") == "map[overwrite[array[float]]]"

    # Realization-level: the seeded dose survives the composite build intact,
    # i.e. the overwrite store is real (a bare-{} additive store drops it).
    comp = Composite({"state": cell_state}, core=core)
    fields = comp.state.get("fields", {})
    assert "drug" in fields, f"fields chain inert — value dropped at build: {fields!r}"
    assert float(np.asarray(fields["drug"]).flat[0]) == 5.0


# --------------------------------------------------------------------------- #
# STATIC dose — shape_seed_literal -> boundary.external.<drug> read as FINITE.
#
# shape_seed_literal resolves to a pint Quantity (magnitude * units) applied to
# boundary.external.<drug> at the END of apply_injected_processes — AFTER the
# baseline's own _normalize_boundary_units pass. boundary.external is a plain-mM-
# float store (`map[overwrite[float[mM]]]`); a pint Quantity value there silently
# realizes to None at composite build, so the transport reads None -> NaN ->
# solve_ivp "y0 must be finite". apply_injected_processes must re-normalize the
# seeded dose to a plain float so it survives realization.
# --------------------------------------------------------------------------- #
def test_static_shape_seed_boundary_dose_is_finite_after_realize():
    import math

    from process_bigraph import Composite
    from vivarium.library.units import units

    from v2ecoli.core import build_core

    core = build_core()
    spec = _spec(_BoundaryReader, "boundary_reader_test", {"boundary": ["boundary"]})
    # shape_seed_literal resolves to a pint Quantity keyed by store-path tuple.
    spec["shape_seed"] = {("boundary", "external", "drug"): 0.001 * units.mM}
    cell_state = {"boundary": {"external": {"GLC[p]": 20.0}}}  # baseline plain-mM media
    inject.apply_injected_processes(cell_state, [], core, [spec])

    # Seam-level: the seeded dose is a PLAIN float (not a pint Quantity), so it
    # can live in the plain-mM-float boundary.external store.
    seeded = cell_state["boundary"]["external"]["drug"]
    assert not hasattr(seeded, "magnitude"), f"still a pint Quantity: {seeded!r}"
    assert seeded == 0.001

    # Realization-level: the value SURVIVES composite build as a finite float
    # (a pint Quantity here realized to None — the y0-must-be-finite crash).
    comp = Composite({"state": cell_state}, core=core)
    realized = comp.state["boundary"]["external"]["drug"]
    assert realized is not None, "boundary.external.drug realized to None (NaN crash)"
    assert math.isfinite(realized) and realized == 0.001
    # And the media nutrient the baseline owns is untouched.
    assert comp.state["boundary"]["external"]["GLC[p]"] == 20.0


# --------------------------------------------------------------------------- #
# Defect 1 — the initial_state overlay onto a fresh native store.
# --------------------------------------------------------------------------- #
def test_native_initial_state_seeds_and_types_fresh_store():
    """A config-declared ``initial_state`` for a native process seeds its NEW
    store AND that store carries the correct overwrite type (so a later Step
    output composes forward instead of being clobbered)."""
    from v2ecoli.core import build_core

    core = build_core()
    spec = _spec(_FieldWriter, "field_writer_test",
                 {"global_time": ["global_time"], "fields": ["fields"]},
                 initial_state={"fields": {"drug": [[3.0]]}}, config={"dose": 0.0})
    cell_state = {"global_time": 0.0}
    inject.apply_injected_processes(cell_state, [], core, [spec])

    assert cell_state["fields"]["_type"] == "map[overwrite[array[float]]]"
    assert cell_state["fields"]["drug"] == [[3.0]]          # config initial_state applied


def test_native_initial_state_skips_baseline_root():
    """A config ``initial_state`` targeting a PROTECTED baseline root (``boundary``)
    is skipped, never clobbering v2's own store — same contract as vivarium_1."""
    from v2ecoli.core import build_core

    core = build_core()
    spec = _spec(_BoundaryWriter, "boundary_writer_test",
                 {"fields": ["fields"], "boundary": ["boundary"]},
                 initial_state={"boundary": {"external": {"drug": 9.0}}})
    cell_state = {"boundary": {"_real_array": True}}        # baseline store v2 owns
    inject.apply_injected_processes(cell_state, [], core, [spec])

    assert cell_state["boundary"] == {"_real_array": True}  # untouched
    assert "_type" not in cell_state["boundary"]            # baseline type unchanged


# --------------------------------------------------------------------------- #
# No-regression — the common path (no initial_state) still builds.
# --------------------------------------------------------------------------- #
def test_native_without_initial_state_still_builds():
    from v2ecoli.core import build_core

    core = build_core()
    spec = _spec(_FieldWriter, "field_writer_test",
                 {"global_time": ["global_time"], "fields": ["fields"]},
                 config={"dose": 1.0})
    cell_state = {"global_time": 0.0}
    added = inject.apply_injected_processes(cell_state, [], core, [spec])

    assert added == ["field_writer_test"]
    assert "field_writer_test" in cell_state                # edge added
    assert "fields" in cell_state                           # store introduced
    assert cell_state["fields"]["_type"] == "map[overwrite[array[float]]]"


# --------------------------------------------------------------------------- #
# Defect 3 — list-shaped timeline survives config-realize through LineageProcess.
# --------------------------------------------------------------------------- #
def test_lineage_injected_processes_list_timeline_survives_config_realize():
    """A list-shaped fork ``timeline`` inside ``injected_processes`` must survive
    the bigraph-schema config-realize a Composite performs on a LineageProcess
    edge. Without the ``quote`` type on the ``injected_processes`` config key,
    ``[[100, {"drug": 1.0}]]`` is mangled to ``[[100, 100]]`` — crashing the
    generation-1 rebuild of a dynamic-dose run."""
    from process_bigraph import Composite

    from v2ecoli.core import build_core
    from v2ecoli.workflow.meta_composite import register_workflow_processes

    core = build_core()
    register_workflow_processes(core)

    timeline = [[100, {"drug": 1.0}]]
    doc = {
        "lin": {
            "_type": "process",
            "address": "local:LineageProcess",
            "config": {
                "experiment_id": "x",
                "injected_processes": {
                    "fork_repo": "/tmp/fork",
                    "add_processes": ["field_timeline"],
                    "process_configs": {
                        "field_timeline": {"timeline": timeline, "bins": [1, 1]},
                    },
                },
            },
            "inputs": {},
            "outputs": {},
            "interval": 1.0,
        }
    }
    comp = Composite({"state": doc}, core=core)
    realized = (comp.state["lin"]["config"]["injected_processes"]
                ["process_configs"]["field_timeline"]["timeline"])
    assert realized == timeline, f"timeline mangled by config-realize: {realized!r}"
