from dataclasses import dataclass
import v2ecoli.structural.build as B
from pbg_parsimony import StructureRef


def test_relax_ingredients_rewrites_and_skips(monkeypatch, tmp_path):
    calls = []

    def fake(ref, cache_dir, relax_cfg, obj_id):
        calls.append(obj_id)
        if obj_id == "boom":
            raise RuntimeError("fetch failed")
        return tmp_path / f"{obj_id}.pdb"

    monkeypatch.setattr("pbg_parsimony.relax_cache.get_or_relax", fake)
    ings = [
        B.Ingredient(id="af", count=1, structure=StructureRef("alphafold", "P0A9B2")),
        B.Ingredient(id="pdbx", count=1, structure=StructureRef("pdb", "1CRN")),
        B.Ingredient(id="boom", count=1, structure=StructureRef("alphafold", "PXXXX")),
        B.Ingredient(id="lipid", count=9, sphere_radius=12.0),  # no structure
    ]
    out = B.relax_ingredients(ings, cache_dir=str(tmp_path), relax_cfg={"equil_ps": 5.0})
    by = {i.id: i for i in out}
    assert by["af"].structure == StructureRef("file", str(tmp_path / "af.pdb"))
    assert by["pdbx"].structure == StructureRef("file", str(tmp_path / "pdbx.pdb"))
    assert by["boom"].structure == StructureRef("alphafold", "PXXXX")  # failure keeps raw
    assert by["lipid"].structure is None                              # no-structure passthrough
    assert set(calls) == {"af", "pdbx", "boom"}                        # lipid never attempted


def test_pack_step_config_accepts_relax():
    from v2ecoli.core import build_core
    from v2ecoli.structural.pack_step import EcoliPackStep
    core = build_core()
    core.register_link("EcoliPackStep", EcoliPackStep)
    step = EcoliPackStep(config={"snapshots": {"t": 1.0}, "study": "x", "out_dir": str(""),
                                 "relax": True, "cache_dir": "out/cache",
                                 "relax_params": {"equil_ps": 100.0}}, core=core)
    assert step.config["relax"] is True and step.config["relax_params"]["equil_ps"] == 100.0
