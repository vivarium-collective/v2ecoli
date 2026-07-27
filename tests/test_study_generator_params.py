import glob, yaml, pathlib
from viva_superpowers.composite_generator import discover_generators, _REGISTRY
WS = pathlib.Path(__file__).resolve().parents[1] / "workspace"
discover_generators()

def _gen_params(composite_id):
    e = _REGISTRY.get(composite_id)
    return set((getattr(e, "parameters", {}) or {}).keys()) if e else None

def test_baseline_params_are_generator_accepted():
    bad = []
    for p in glob.glob(str(WS / "studies/*/study.yaml")):
        spec = yaml.safe_load(open(p, encoding="utf-8")) or {}
        base = ((spec.get("conditions") or {}).get("baseline") or {})
        comp, params = base.get("composite"), (base.get("params") or {})
        if not comp:
            continue
        gp = _gen_params(comp)
        if gp is None:      # unresolved composite -> Task 2's concern; skip here
            continue
        unknown = set(params) - gp
        if unknown:
            bad.append(f"{p.split('/')[-2]}: {sorted(unknown)} not in {comp} params")
    assert not bad, "studies with non-generator params:\n" + "\n".join(bad)
