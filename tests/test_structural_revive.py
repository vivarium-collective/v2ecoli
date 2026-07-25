def test_structural_imports_and_selects():
    from v2ecoli.structural.build import select_ingredients, DATA
    assert DATA.is_dir()
    # a couple of known abundant species → non-empty ingredient list
    ings = select_ingredients({"EG10893-MONOMER": 5000, "CPLX0-3964": 500}, top_n=2)
    assert isinstance(ings, list) and len(ings) >= 1
