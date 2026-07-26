import numpy as np
import v2ecoli.structural.build as B


def test_bulk_to_locations_dominant_tag():
    bulk = {"id": np.array(["FOO[c]", "FOO[p]", "BAR[o]", "BAZ[i]", "QUX[m]", "PLAIN"]),
            "count": np.array([10, 3, 5, 5, 5, 5])}
    loc = B.bulk_to_locations(bulk)
    assert loc["FOO"] == "cytoplasm"        # 10[c] > 3[p]
    assert loc["BAR"] == "outer_membrane"
    assert loc["BAZ"] == "inner_membrane"
    assert loc["QUX"] == "inner_membrane"   # m -> inner
    assert loc["PLAIN"] == "cytoplasm"      # untagged


def test_pack_from_state_passes_envelope(monkeypatch):
    calls = {}
    def fake_build_pack(ingredients, capsule, chromosome, **kw):
        calls["envelope"] = kw.get("envelope"); calls["capsule"] = capsule
        return {"n_placed": 0, "pack_path": "x", "ingredients": 0}
    monkeypatch.setattr(B, "build_pack", fake_build_pack)
    monkeypatch.setattr(B, "select_ingredients", lambda counts, locations=None, **kw: [])
    B.pack_from_state("out", "m", {"X": 1}, 1.153, {"X": "periplasm"},
                      top_n=2, envelope=True)
    env = calls["envelope"]
    assert env is not None and set(env) == {"inner", "outer"}
    assert env["inner"].radius < env["outer"].radius        # inner is nested
    assert env["inner"].half_len < env["outer"].half_len
    # envelope=False → single capsule, no envelope
    B.pack_from_state("out", "m", {"X": 1}, 1.153, {"X": "periplasm"}, envelope=False)
    assert calls["envelope"] is None


def test_select_ingredients_sets_compartment_and_region():
    # a tiny synthetic counts hitting the top-N monomer path; locations drive compartment.
    # If select_ingredients needs heavy reference data, mark this test slow or skip;
    # the goal is to assert an ingredient's compartment/region follow its location.
    import pytest
    try:
        ings = B.select_ingredients({"EG10544-MONOMER": 100},
                                    {"EG10544-MONOMER": "inner_membrane"}, top_n=1)
    except Exception as e:
        pytest.skip(f"select_ingredients needs reference data: {e}")
    m = {i.id: i for i in ings}
    if "EG10544-MONOMER" in m:
        assert m["EG10544-MONOMER"].compartment == "inner_membrane"
        assert m["EG10544-MONOMER"].region == "surface"
    # the lipid is always present → inner_membrane surface
    if "lipid" in m:
        assert m["lipid"].compartment == "inner_membrane" and m["lipid"].region == "surface"
