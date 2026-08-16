import os
import warnings; warnings.filterwarnings("ignore")
import pytest

from v2ecoli.core import build_core
from v2ecoli.steps.millard_pdmp_metabolism import MillardPDMPMetabolism


# build_core() is expensive; share one across the module.
@pytest.fixture(scope="module")
def core():
    return build_core()


@pytest.mark.sim
def test_millard_step_accepts_external_concentrations(core):
    """The metabolism step exposes an external_concentrations input port so the
    bioreactor environment can drive the Millard kinetics each tick."""
    s = MillardPDMPMetabolism(config={}, core=core)
    assert "external_concentrations" in s.inputs()


def _run_with_external_glucose(core, glcx_mM):
    """Advance one Millard tick driven by an external glucose level (mM);
    return the PTS glucose-uptake flux (PTS_4: GLCp + eiicbP -> G6P + eiicb)."""
    s = MillardPDMPMetabolism(config={}, core=core)
    out = s.update(
        {
            "lqr_control": {},
            "bulk": None,
            "listeners_mass": {"cell_mass": 1000.0, "dry_mass": 300.0},
            "external_concentrations": {"GLCx": glcx_mM},
        },
        1.0,
    )
    return out["central_fluxes"]["PTS_4"]


@pytest.mark.sim
def test_millard_uptake_responds_to_external_glucose(core):
    """Glucose uptake (PTS_4) must fall when external glucose (GLCx) is low.

    GLCx is the SBML external-glucose species; PTS_4 is the phosphotransferase
    uptake reaction that pulls glucose into the cell as G6P. Driving the model
    with low external glucose must starve the uptake flux relative to a
    glucose-rich environment."""
    flux_high = _run_with_external_glucose(core, 1.0)
    flux_low = _run_with_external_glucose(core, 1.0e-4)
    assert abs(flux_low) < abs(flux_high), (
        f"uptake flux did not fall at low glucose: "
        f"high={flux_high:.5g}, low={flux_low:.5g}"
    )


@pytest.mark.sim
def test_baseline_millard_builds():
    from v2ecoli import build_composite
    from process_bigraph.composite import Composite
    comp = build_composite("ecoli_millard", seed=0, cache_dir="out/cache")
    assert isinstance(comp, Composite)


@pytest.mark.sim
def test_baseline_millard_runs_and_grows():
    from v2ecoli import build_composite
    c = build_composite("ecoli_millard", seed=0, cache_dir="out/cache")
    c.run(5)
    ag = (c.state.get("agents") or {}).get("0") or {}
    assert (ag.get("listeners", {}).get("mass", {}).get("cell_mass", 0.0)) > 0.0
    assert ag.get("central_fluxes")


@pytest.mark.sim
def test_baseline_millard_has_no_unbound_core_instances():
    """Regression guard for item 57: ``_millard_helpers.py``'s own copy of
    ``_get_step_config``'s Requester/Evolver wrapping had the identical
    missing-``core=`` defect as ``ecoli_baseline.py``'s (see
    ``test_composites_baseline.py::test_baseline_composite_has_no_unbound_core_instances``
    for the full mechanism) -- same disease, copy-pasted into a second
    composite. Walks the entire real composite state so any other
    construction site with the same disease is caught here too."""
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    from v2ecoli import build_composite
    from _core_binding_check import unbound_core_instances

    c = build_composite("ecoli_millard", seed=0, cache_dir="out/cache")

    agent = c.state["agents"]["0"]
    partitioned_steps = [k for k in agent if k.endswith("_requester") or k.endswith("_evolver")]
    assert partitioned_steps, (
        "expected at least one Requester/Evolver step under agents.0 -- "
        "PARTITIONED_PROCESSES wiring may have changed; update this test's "
        "assumptions rather than silently passing on zero coverage")

    unbound = unbound_core_instances(c.state)
    assert not unbound, (
        f"instance(s) with core=None found in the real composite state: {unbound} -- "
        "Composite.serialize_state() will crash on these with "
        "AttributeError: 'NoneType' object has no attribute 'access' "
        "(bigraph_schema/methods/serialize.py, Link.serialize) the moment "
        "a real run reaches its final state write.")


@pytest.mark.sim
def test_baseline_millard_serializes_final_state():
    """Regression guard for item 58: ``_millard_helpers.py``'s own copy of
    the ``counts_deriver``/``rnap_data_listener`` wiring has the identical
    ``register_labeled_array()`` string-``_data`` defect as
    ``ecoli_baseline.py``'s (see
    ``test_composites_baseline.py::test_baseline_composite_serializes_final_state``
    for the full mechanism) -- same disease, same shared
    ``v2ecoli/types/labeled_array.py`` helper, copy-pasted deriver wiring
    into a second composite. Calls the exact real ``serialize_state()``
    entry point ``run_pbg.py`` calls at the end of every real run."""
    if not os.path.isdir("out/cache") and not os.environ.get("CI"):
        pytest.skip("cache dir 'out/cache' not present; "
                    "build via `python scripts/build_cache.py` (CI builds it automatically)")
    from v2ecoli import build_composite
    c = build_composite("ecoli_millard", seed=0, cache_dir="out/cache")
    serialized = c.serialize_state()
    assert isinstance(serialized, dict)
