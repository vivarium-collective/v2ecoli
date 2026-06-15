from v2ecoli.steps.fba_flux_coupler import FBAFluxCoupler
from v2ecoli.core import build_core


def test_coupler_maps_and_converts():
    core = build_core()
    c = FBAFluxCoupler(config={
        "reaction_map_file": "v2ecoli/data/millard_v2ecoli_reaction_map.yaml",
        "coefficient": 2.0,
    }, core=core)
    out = c.update({"central_fluxes": {"PFK": 3.0, "AdK": 9.9}}, 1.0)
    pins = out["pinned_flux_targets"]
    fba_id = c.reaction_map["PFK"][0][0]      # PFK is mapped
    assert pins[fba_id] == 6.0                # 3.0 * coefficient 2.0 * scale 1.0
    assert all("AdK" not in k for k in pins)  # AdK is millard_only -> not pinned


def test_coupler_empty_fluxes_ok():
    core = build_core()
    c = FBAFluxCoupler(config={
        "reaction_map_file": "v2ecoli/data/millard_v2ecoli_reaction_map.yaml"}, core=core)
    out = c.update({"central_fluxes": {}}, 1.0)
    assert out["pinned_flux_targets"] == {}
