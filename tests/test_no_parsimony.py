import importlib, sys, pytest

def test_import_v2ecoli_does_not_import_pbg_parsimony():
    for m in list(sys.modules):
        if m == "pbg_parsimony" or m.startswith("pbg_parsimony."):
            del sys.modules[m]
    importlib.import_module("v2ecoli")
    import v2ecoli.composites.ecoli_baseline
    assert not any(m == "pbg_parsimony" or m.startswith("pbg_parsimony.")
                   for m in sys.modules), "v2ecoli must not import pbg_parsimony"

def test_structural_modules_gone():
    for name in ("v2ecoli.structural", "v2ecoli.structural.build",
                 "v2ecoli.composites.ecoli_structural"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
