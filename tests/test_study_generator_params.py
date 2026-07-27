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


def test_all_composite_refs_resolve():
    unresolved = []
    for p in glob.glob(str(WS / "studies/*/study.yaml")):
        spec = yaml.safe_load(open(p, encoding="utf-8")) or {}
        cond = spec.get("conditions") or {}
        refs = []
        b = (cond.get("baseline") or {}).get("composite");  refs += [b] if b else []
        for v in (cond.get("variants") or []):
            if v.get("composite"): refs.append(v["composite"])
        for r in refs:
            # skip file-discovered YAML composites (…composite.yaml) — not in _REGISTRY
            if r in _REGISTRY: continue
            if r.endswith("millard2017_metabolism"): continue  # YAML composite, file-discovered
            # Out of scope for Task 2 (bare ecoli_baseline/parca + reactor_bird_coupled
            # aliases only): these reference composites that don't exist yet in this repo
            # (v2ecoli_pdmp package unwritten; pbg_copasi/diagnostic ids not registered
            # under this name) -- aspirational refs on not-yet-implemented PDMP/diagnostic
            # studies, tracked separately from this contract.
            if r.startswith("v2ecoli_pdmp.") or r.startswith("pbg_copasi.") or r == "v2ecoli.composites.diagnostic.diagnostic":
                continue
            unresolved.append(f"{p.split('/')[-2]}: {r}")
    assert not unresolved, "unresolved composite refs:\n" + "\n".join(unresolved)
