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
