from viva_superpowers.composite_generator import _REGISTRY, discover_generators


def _entry():
    if not _REGISTRY:
        discover_generators()
    import v2ecoli.composites  # force registration
    return _REGISTRY["v2ecoli.composites.vecoli.vecoli"]


def test_whole_config_and_variant_are_declared_params():
    params = _entry().parameters
    assert "whole_config" in params
    assert "variant" in params
    assert "observable_bulk_ids" in params


def test_unknown_param_still_rejected():
    # sanity: declared set is a subset guard the workbench relies on
    params = set(_entry().parameters)
    assert {"whole_config", "variant", "observable_bulk_ids"} <= params
