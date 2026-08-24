"""Unit tests for the native ``CellGeometry`` step (v2ecoli mecillinam
candidate arm).

``CellGeometry`` ports vEcoli's ``ecoli/processes/shape.py`` geometry math:
periplasm/cytoplasm volume split (0.2 / 0.8 of whole-cell volume) plus the
outer surface area from 3D capsule geometry. It exists to populate
``volumes.periplasm`` / ``volumes.cytoplasm`` / ``boundary.outer_surface_area``
for the injected ``antibiotic_transport_odeint`` process, which otherwise
finds nothing writing those stores in the single-cell candidate.
"""

import os

import pytest

from v2ecoli.steps.derivers.cell_geometry import CellGeometry, PERIPLASM_FRACTION


CACHE_DIR = "out/cache"
_no_cache = not os.path.isdir(CACHE_DIR) and not os.environ.get("CI")


@pytest.mark.fast
def test_cell_geometry_splits_periplasm_and_cytoplasm():
    step = CellGeometry({}, core=None)
    cell_volume_L = 1.0e-15  # 1 fL
    out = step.compute(cell_volume_L)
    assert abs(out["periplasm"] - cell_volume_L * PERIPLASM_FRACTION) < 1e-30
    assert abs(out["cytoplasm"] - cell_volume_L * (1 - PERIPLASM_FRACTION)) < 1e-30
    assert out["periplasm"] > 0 and out["cytoplasm"] > 0


@pytest.mark.fast
def test_cell_geometry_periplasm_and_cytoplasm_sum_to_whole_cell():
    step = CellGeometry({}, core=None)
    cell_volume_L = 1.2e-15
    out = step.compute(cell_volume_L)
    assert abs(
        (out["periplasm"] + out["cytoplasm"]) - cell_volume_L
    ) < 1e-30


@pytest.mark.fast
def test_cell_geometry_outer_surface_area_positive_and_scales_with_volume():
    step = CellGeometry({}, core=None)
    small = step.compute(1.0e-15)["outer_surface_area"]
    large = step.compute(2.0e-15)["outer_surface_area"]
    assert small > 0
    assert large > small


@pytest.mark.fast
def test_cell_geometry_default_width_is_one_micron():
    # Matches vEcoli ecoli/processes/shape.py Shape.defaults["width"] (1.0 um).
    step = CellGeometry({}, core=None)
    assert step.width_um == pytest.approx(1.0)


@pytest.mark.fast
def test_cell_geometry_update_reads_mass_listener_volume_only():
    """The step's `update` reads cell volume from `listeners.mass.volume`
    (this candidate's own mass listener) and never touches `boundary` as an
    input — it must not choke on a `boundary` store containing the media
    `external` dict of plain floats (the bridge quirk this step exists to
    avoid). `inputs()` must not declare a `boundary` port at all."""
    step = CellGeometry({}, core=None)
    assert "boundary" not in step.inputs()

    from v2ecoli.types.quantity import ureg as units

    states = {"listeners": {"mass": {"volume": 1.2 * units.fL}}}
    update = step.update(states)

    periplasm = update["volumes"]["periplasm"]
    cytoplasm = update["volumes"]["cytoplasm"]
    outer_surface_area = update["boundary"]["outer_surface_area"]

    assert periplasm.to(units.L).magnitude > 0
    assert cytoplasm.to(units.L).magnitude > 0
    assert outer_surface_area.to(units.um**2).magnitude > 0

    total = periplasm.to(units.L) + cytoplasm.to(units.L)
    expected = (1.2 * units.fL).to(units.L)
    assert abs(total.magnitude - expected.magnitude) < 1e-30


@pytest.mark.fast
def test_antibiotic_mode_off_path_unaffected():
    """Non-antibiotic (mecillinam=False) baseline builds do not enable the
    `cell_geometry` feature module, so `volumes`/geometry wiring is untouched.
    Guards the wiring gate itself, independent of the (heavier) build-level
    document test in test_ecoli_baseline_cell_geometry_wiring.py."""
    from v2ecoli.composites.ecoli_baseline import (
        FEATURE_MODULES, build_execution_layers,
    )

    assert "cell_geometry" in FEATURE_MODULES

    layers_off = build_execution_layers([])
    steps_off = {s for layer in layers_off for s in layer}
    assert "cell_geometry_step" not in steps_off

    layers_on = build_execution_layers(["cell_geometry"])
    steps_on = {s for layer in layers_on for s in layer}
    assert "cell_geometry_step" in steps_on


# --- Build-level: the actual composite document (mirrors the skip-gated
# `@pytest.mark.sim` pattern in tests/test_build_composite.py) -------------

@pytest.mark.sim
@pytest.mark.skipif(
    _no_cache,
    reason=f"cache dir {CACHE_DIR!r} not present; "
           f"rebuild with `python scripts/build_cache.py`",
)
def test_mecillinam_candidate_populates_volumes_and_outer_surface_area():
    """A mecillinam=True candidate build carries a `volumes` store with
    positive periplasm/cytoplasm, and a positive `boundary.outer_surface_area`
    — populated by the native CellGeometry step, not present at all in the
    non-antibiotic baseline."""
    from v2ecoli import build_composite
    from v2ecoli.library.quantity_helpers import as_quantity
    from v2ecoli.types.quantity import ureg as units

    comp = build_composite(
        "ecoli_baseline", seed=0, cache_dir=CACHE_DIR, mecillinam=True)
    comp.run(2)
    state = comp.state["agents"]["0"] if "agents" in comp.state else comp.state

    periplasm = as_quantity(state["volumes"]["periplasm"], units.L)
    cytoplasm = as_quantity(state["volumes"]["cytoplasm"], units.L)
    outer_surface_area = as_quantity(
        state["boundary"]["outer_surface_area"], units.um**2)

    assert periplasm.magnitude > 0
    assert cytoplasm.magnitude > 0
    assert outer_surface_area.magnitude > 0


@pytest.mark.sim
@pytest.mark.skipif(
    _no_cache,
    reason=f"cache dir {CACHE_DIR!r} not present; "
           f"rebuild with `python scripts/build_cache.py`",
)
def test_non_antibiotic_candidate_has_no_volumes_store():
    """mecillinam=False (the default) never gains a `volumes` store — the
    cell_geometry feature is only auto-enabled by mecillinam=True."""
    from v2ecoli import build_composite

    comp = build_composite(
        "ecoli_baseline", seed=0, cache_dir=CACHE_DIR, mecillinam=False)
    comp.run(2)
    state = comp.state["agents"]["0"] if "agents" in comp.state else comp.state
    assert "volumes" not in state
