from v2ecoli.library.fba_reaction_map import load_reaction_map

MAP_PATH = "v2ecoli/data/millard_v2ecoli_reaction_map.yaml"


def test_load_reaction_map_shape():
    m = load_reaction_map(MAP_PATH)
    assert "PGI" in m  # glucose-6-P isomerase (single-isozyme, actively pinned)
    entry = m["PGI"]
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


def _find_metabolism_fba(composite):
    """Walk a built composite for a metabolism instance exposing model.fba
    (the live FBA object whose getReactionIDs() are the ids pinning uses)."""
    seen = set()

    def walk(obj, depth=0):
        if depth > 8 or id(obj) in seen:
            return None
        seen.add(id(obj))
        model = getattr(obj, "model", None)
        fba = getattr(model, "fba", None)
        if fba is not None and hasattr(fba, "getReactionIDs"):
            return fba
        try:
            children = obj.values() if isinstance(obj, dict) else (
                vars(obj).values() if hasattr(obj, "__dict__") else [])
        except Exception:
            children = []
        for c in children:
            r = walk(c, depth + 1)
            if r is not None:
                return r
        return None

    for root in (getattr(composite, "state", None), composite):
        r = walk(root)
        if r is not None:
            return r
    return None


def test_pins_are_real_fba_reaction_ids():
    """Every ACTIVE pin must be a valid fba.getReactionIDs() id, not just a
    base_reaction_id. A base id that has no single fba reaction (isozyme-split)
    is silently IGNORED by setReactionFluxBounds ('unknown reaction') -- this
    test catches that class (it found PFK/FBA/PYK/PPC/PTA/SDH, now parked in
    needs_variant_pinning)."""
    import warnings

    warnings.filterwarnings("ignore")
    try:
        from v2ecoli import build_composite

        c = build_composite("baseline", cache_dir="out/cache", seed=0)
    except Exception as e:  # pragma: no cover - environment dependent
        import pytest

        pytest.skip(f"WCM build unavailable: {e}")

    fba = _find_metabolism_fba(c)
    if fba is None:  # pragma: no cover - environment dependent
        import pytest

        pytest.skip("could not locate metabolism fba on the composite")

    fba_ids = set(fba.getReactionIDs())
    m = load_reaction_map(MAP_PATH)
    ignored = {
        millard: fba_id
        for millard, targets in m.items()
        for fba_id, _scale in targets
        if fba_id not in fba_ids
    }
    assert not ignored, (
        "active pins whose id is NOT a real fba reaction (silently ignored): "
        f"{ignored} -- move to needs_variant_pinning or fix the id"
    )
