import warnings; warnings.filterwarnings("ignore")
import pytest
from v2ecoli import build_composite


@pytest.mark.sim
def test_millard_emits_central_fluxes():
    c = build_composite("millard_pdmp_baseline", with_ref_growth=True,
                        ref_growth_flux_source="consumption_matched", seed=0)
    c.run(5)
    ag = (c.state.get("agents") or {}).get("0") or {}
    fluxes = ag.get("central_fluxes") or {}
    assert fluxes, "central_fluxes store is empty"
    # fluxes are finite floats keyed by Millard reaction name
    assert all(isinstance(v, float) for v in fluxes.values())
    # a known glycolytic reaction is present (PGI/PFK/PYK/ENO ... at least one)
    assert any(k in fluxes for k in ("PFK", "PGI", "PYK", "ENO", "PGK"))
