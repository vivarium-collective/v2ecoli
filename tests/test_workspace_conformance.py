import locale, pathlib, yaml, graphlib
from viva_superpowers.composite_generator import discover_generators, _REGISTRY
WS = pathlib.Path(__file__).resolve().parents[1] / "workspace"

# discover_generators() (via a transitive import, e.g. polars) resets LC_CTYPE
# to "C", which silently flips Path.read_text()'s default encoding to ASCII
# for the rest of THIS PROCESS -- including other tests running later in the
# same pytest session that read non-ASCII study.yaml/reference files (see the
# matching fix in scripts/lint-workspace.py's check_generator_contract()).
# Save + restore around the module-level call so importing this test module
# has no side effect on the process locale.
_saved_lc_ctype = locale.setlocale(locale.LC_CTYPE)
try:
    discover_generators()
finally:
    locale.setlocale(locale.LC_CTYPE, _saved_lc_ctype)

# Documented exceptions to the canonical conditions-form rule:
NO_MODEL = {"parca"}                                  # upstream artifact producer, no model
MULTI_BASELINE_PENDING = {"mbp-07-millard-kinetic-metabolism"}  # multi_baseline_needs_human (user decision)

# Study-config <-> generator contract (Tasks 1-3 of the
# study-config-generator-contract SDD): baseline.params must be a subset of
# the generator's declared parameters, modulo:
#   - _RUN_CONTROL_KEYS: keys the workbench engine strips from generator
#     overrides before calling it (they control run mechanics, e.g. how many
#     steps to run, not composite structure) -- see workbench #611.
#   - _PARAM_CONTRACT_EXCEPTIONS: studies with a baseline param that is a
#     genuine mismatch pending a study-owner decision on where the value
#     belongs. Documented here (not silently dropped) so the underlying study
#     data is untouched until that decision is made.
_RUN_CONTROL_KEYS = {"n_steps"}
_PARAM_CONTRACT_EXCEPTIONS = {
    # swap: v2ecoli.composites.ecoli_baseline.ecoli_baseline has no `swap`
    # param; may belong under config_overrides/injected_processes -- a
    # domain call for the study owner, not auto-mapped here.
    "metabolism_redux_basal",
    "metabolism_redux_with_aa",
    "metabolism_redux_succinate",
    "metabolism_redux_no_oxygen",
    "metabolism_redux_acetate",
    # mode: full: v2ecoli.composites.parca.parca declares debug/cpus/
    # cache_dir, not mode -- `mode` is a v2ecoli-parca CLI flag
    # (debug=(mode=='fast')), not a composite-generator parameter.
    "showcase-1-parca",
    # features: [mass_conservation]: v2ecoli.composites.reactor_bird_coupled
    # has no `features` param (that key exists on ecoli_baseline /
    # reactor_bird_coupled_millard) -- looks copied from a sibling composite;
    # pending a study-owner fix.
    "mbp-04-multigeneration-runs",
}

# Composite refs the registry can't resolve for reasons outside this
# contract's scope (mirrors Task 2's skip-list):
_COMPOSITE_UNRESOLVED_PREFIXES = ("v2ecoli_pdmp.", "pbg_copasi.")  # optional/external packages, not in _REGISTRY
_COMPOSITE_UNRESOLVED_SUFFIXES = ("millard2017_metabolism",)  # file-discovered YAML composite, not in _REGISTRY
_COMPOSITE_UNRESOLVED_EXACT = {"v2ecoli.composites.diagnostic.diagnostic"}  # known-nonexistent, aspirational ref

def _studies():
    return {p.parent.name: (yaml.safe_load(p.read_text(encoding="utf-8")) or {})
            for p in (WS / "studies").glob("*/study.yaml")}

def test_no_nested_studies():
    nested = list(WS.glob("investigations/*/studies/*/study.yaml"))
    assert not nested, f"nested study.yaml must not exist: {nested}"

def test_investigations_use_members_not_studies():
    for inv in (WS / "investigations").glob("*/investigation.yaml"):
        spec = yaml.safe_load(inv.read_text(encoding="utf-8")) or {}
        assert "studies" not in spec, f"{inv.parent.name} still uses studies: (must be members:)"

def test_studies_canonical_conditions_form():
    for slug, spec in _studies().items():
        if slug in MULTI_BASELINE_PENDING:
            # exception: still top-level multi-baseline; must at least be a valid non-empty list
            bl = spec.get("baseline")
            assert isinstance(bl, list) and bl and all(b.get("composite") for b in bl), \
                f"{slug}: multi-baseline exception must be a valid top-level baseline list"
            continue
        assert "baseline" not in spec, f"{slug} has a stray top-level baseline: (should be conditions.baseline)"
        assert "parent_studies" not in spec, f"{slug} retains parent_studies (ordering must be inputs.from)"
        pg = spec.get("pipeline_gate") or {}
        assert "prerequisites" not in pg, f"{slug} retains pipeline_gate.prerequisites (must be inputs.from)"
        if slug not in NO_MODEL:
            comp = ((spec.get("conditions") or {}).get("baseline") or {}).get("composite")
            assert comp, f"{slug} missing conditions.baseline.composite"

def test_no_dict_shaped_tests_in_conditions_form():
    for slug, spec in _studies().items():
        if isinstance(spec.get("conditions"), dict):
            assert not isinstance(spec.get("tests"), dict), \
                f"{slug} conditions-form study has dict-shaped tests (must be a list)"

def test_inputs_dag_acyclic_and_resolvable():
    studies = _studies()
    slugs = set(studies) | {"parca"}
    ts = graphlib.TopologicalSorter()
    for slug, spec in studies.items():
        deps = []
        for e in (spec.get("inputs") or []):
            frm = e.get("from")
            assert frm in slugs, f"{slug} inputs.from '{frm}' is not a real study"
            deps.append(frm)
        ts.add(slug, *deps)
    ts.prepare()   # raises graphlib.CycleError if cyclic


def _gen_params(composite_id):
    e = _REGISTRY.get(composite_id)
    return set((getattr(e, "parameters", {}) or {}).keys()) if e else None


def test_baseline_params_are_generator_accepted():
    bad = []
    for slug, spec in _studies().items():
        if slug in _PARAM_CONTRACT_EXCEPTIONS:
            continue
        base = ((spec.get("conditions") or {}).get("baseline") or {})
        comp, params = base.get("composite"), (base.get("params") or {})
        if not comp:
            continue
        gp = _gen_params(comp)
        if gp is None:      # unresolved composite -> test_all_composite_refs_resolve's concern
            continue
        unknown = set(params) - gp - _RUN_CONTROL_KEYS
        if unknown:
            bad.append(f"{slug}: {sorted(unknown)} not in {comp} params")
    assert not bad, "studies with non-generator params:\n" + "\n".join(bad)


def test_all_composite_refs_resolve():
    unresolved = []
    for slug, spec in _studies().items():
        cond = spec.get("conditions") or {}
        refs = []
        b = (cond.get("baseline") or {}).get("composite")
        if b:
            refs.append(b)
        for v in (cond.get("variants") or []):
            if v.get("composite"):
                refs.append(v["composite"])
        for r in refs:
            if r in _REGISTRY or r in _COMPOSITE_UNRESOLVED_EXACT:
                continue
            if r.startswith(_COMPOSITE_UNRESOLVED_PREFIXES) or r.endswith(_COMPOSITE_UNRESOLVED_SUFFIXES):
                continue
            unresolved.append(f"{slug}: {r}")
    assert not unresolved, "unresolved composite refs:\n" + "\n".join(unresolved)
