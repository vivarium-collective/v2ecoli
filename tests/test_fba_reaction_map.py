from v2ecoli.library.fba_reaction_map import load_reaction_map

MAP_PATH = "v2ecoli/data/millard_v2ecoli_reaction_map.yaml"


def test_load_reaction_map_shape():
    m = load_reaction_map(MAP_PATH)
    assert "PFK" in m  # glycolysis phosphofructokinase is mapped
    entry = m["PFK"]
    assert isinstance(entry, list) and len(entry) >= 1
    fba_id, scale = entry[0]
    assert isinstance(fba_id, str) and isinstance(scale, float)


def test_map_excludes_millard_only():
    m = load_reaction_map(MAP_PATH)
    assert all(v for v in m.values())  # no empty target lists
    # at least ~10 central reactions mapped (glycolysis core minimum)
    assert len(m) >= 10


def test_mapped_ids_exist_in_wcm():
    import warnings

    warnings.filterwarnings("ignore")
    try:
        from v2ecoli import build_composite

        c = build_composite("baseline", cache_dir="out/cache", seed=0)
    except Exception as e:  # pragma: no cover - environment dependent
        import pytest

        pytest.skip(f"WCM build unavailable: {e}")

    base_ids = _find_base_reaction_ids(c)
    if base_ids is None:  # pragma: no cover - environment dependent
        import pytest

        pytest.skip("could not locate base_reaction_ids on the composite")

    base_ids = set(base_ids)
    m = load_reaction_map(MAP_PATH)
    missing = {
        millard: fba
        for millard, targets in m.items()
        for fba, _scale in targets
        if fba not in base_ids
    }
    assert not missing, f"mapped fba ids not in WCM base_reaction_ids: {missing}"


def _find_base_reaction_ids(composite):
    """Walk a built composite looking for a metabolism instance carrying
    base_reaction_ids."""
    seen = set()

    def walk(obj, depth=0):
        if depth > 8 or id(obj) in seen:
            return None
        seen.add(id(obj))
        base = getattr(obj, "base_reaction_ids", None)
        if base is not None:
            return base
        # process-bigraph composites keep their tree on .state
        for attr in ("state", "instance", "processes"):
            child = getattr(obj, attr, None)
            if child is not None:
                found = walk(child, depth + 1)
                if found is not None:
                    return found
        if isinstance(obj, dict):
            for v in obj.values():
                found = walk(v, depth + 1)
                if found is not None:
                    return found
        return None

    return walk(composite)
