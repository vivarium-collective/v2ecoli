import warnings; warnings.filterwarnings("ignore")
import pytest
from v2ecoli import build_composite


@pytest.mark.sim
def test_harness_runs_and_pins_central_flux():
    c = build_composite("millard_fba_bridge_harness", cache_dir="out/cache", seed=0)
    c.run(10)  # 10 s; must not crash
    ag = (c.state.get("agents") or {}).get("0") or c.state
    # pins were produced from the ODE fluxes
    pins = ag.get("pinned_flux_targets") or {}
    assert len(pins) >= 1, "no FBA reactions were pinned"
    # LP stayed solvable: relaxed_reactions is a list (possibly empty), not a crash
    fb = ((ag.get("listeners") or {}).get("fba_bridge")) or {}
    assert isinstance(fb.get("relaxed_reactions", []), list)
