import pathlib, yaml, graphlib
WS = pathlib.Path(__file__).resolve().parents[1] / "workspace"

# Documented exceptions to the canonical conditions-form rule:
NO_MODEL = {"parca"}                                  # upstream artifact producer, no model
MULTI_BASELINE_PENDING = {"mbp-07-millard-kinetic-metabolism"}  # multi_baseline_needs_human (user decision)

def _studies():
    return {p.parent.name: (yaml.safe_load(p.read_text()) or {})
            for p in (WS / "studies").glob("*/study.yaml")}

def test_no_nested_studies():
    nested = list(WS.glob("investigations/*/studies/*/study.yaml"))
    assert not nested, f"nested study.yaml must not exist: {nested}"

def test_investigations_use_members_not_studies():
    for inv in (WS / "investigations").glob("*/investigation.yaml"):
        spec = yaml.safe_load(inv.read_text()) or {}
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
