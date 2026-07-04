"""Static/empirical audit of vEcoli-cf's translate_ports converter.

Feeds representative vivarium ``ports_schema()`` fragments to the REAL
converter (``ecoli.library.bigraph_types.translate_ports`` in the
read-only vEcoli-cf checkout) and asserts what it produces, so the
"handled? yes/partial/no" column of docs/translate_ports_audit.md is
backed by executable evidence rather than source reading alone.

Run:  .venv/bin/python -m pytest tests/test_translate_ports_audit.py -q

Importing the converter out of vEcoli-cf requires the same shimming the
comparison harness uses (pin the installed/compiled ``wholecell`` tree,
make vivarium's registry idempotent, then purge+reimport ``ecoli`` from
the vEcoli-cf path). If that environment can't be built the module
skips rather than failing.
"""

import importlib
import pkgutil
import sys

import numpy as np
import pytest

VECOLI_DIR = "/Users/eranagmon/code/vEcoli-cf"


def _load_converter():
    # 1. Pin the compiled wholecell tree before vEcoli-cf source shadows it.
    import wholecell as _wc

    for mi in pkgutil.walk_packages(_wc.__path__, "wholecell."):
        try:
            importlib.import_module(mi.name)
        except Exception:
            pass
    # 2. Make vivarium's process-global registry idempotent (ecoli/__init__
    #    re-registers parquet/updaters/dividers on reimport).
    import vivarium.core.registry as _vreg

    if not getattr(_vreg.Registry, "_audit_patched", False):
        _orig = _vreg.Registry.register

        def _idem(self, key, item, alternate_keys=tuple()):
            if key in self.registry:
                return None
            return _orig(self, key, item, alternate_keys)

        _vreg.Registry.register = _idem
        _vreg.Registry._audit_patched = True
    # 3. Put vEcoli-cf first and force a fresh ecoli import from it.
    if VECOLI_DIR not in sys.path:
        sys.path.insert(0, VECOLI_DIR)
    for m in [k for k in list(sys.modules) if k == "ecoli" or k.startswith("ecoli.")]:
        del sys.modules[m]
    from ecoli.library.bigraph_types import translate_ports, ECOLI_TYPES
    from process_bigraph import allocate_core

    assert sys.modules["ecoli"].__file__.startswith(VECOLI_DIR)
    core = allocate_core()
    try:
        core.register_types(ECOLI_TYPES)
    except Exception:
        pass
    return core, translate_ports


@pytest.fixture(scope="module")
def converter():
    try:
        return _load_converter()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load vEcoli-cf translate_ports: {e!r}")


def _render(core, schema):
    """Best-effort type-string for a translate_ports result."""
    from bigraph_schema.methods import render

    if isinstance(schema, dict) and "_type" in schema:
        return schema["_type"]
    if isinstance(schema, dict):
        return {k: _render(core, v) for k, v in schema.items()}
    try:
        return render(schema)
    except Exception:
        return type(schema).__name__


# --- HANDLED constructs (regression guards) --------------------------------


def test_empty_dict_is_node(converter):
    core, tp = converter
    assert _render(core, tp(core, {})) == "node"


def test_empty_collection_defaults_are_node(converter):
    core, tp = converter
    for d in ([], {}, set(), None):
        assert _render(core, tp(core, {"_default": d})) == "node", d


def test_set_updater_wraps_overwrite(converter):
    core, tp = converter
    r = tp(core, {"_default": "", "_updater": "set"})
    assert _render(core, r).startswith("overwrite")


def test_accumulate_is_plain_numeric(converter):
    core, tp = converter
    assert _render(core, tp(core, {"_default": 0.0, "_updater": "accumulate"})) == "float"


def test_glob_sole_child_is_map(converter):
    core, tp = converter
    assert _render(core, tp(core, {"*": {"_default": 0}})) == "map"


def test_glob_with_underscore_sibling_still_map(converter):
    # flagella_motor: {"_divider": "split_dict", "*": {...}} — underscore keys
    # are filtered out so the glob is still recognised.
    core, tp = converter
    r = tp(core, {"_divider": "split_dict", "*": {"_default": 1, "_updater": "set"}})
    assert _render(core, r) == "map"


def test_nested_glob(converter):
    core, tp = converter
    r = tp(core, {"external": {"*": {"_default": 0}}})
    assert _render(core, r) == {"external": "map"}


def test_numpy_array_default(converter):
    core, tp = converter
    assert _render(core, tp(core, {"_default": np.ones(3)})) == "array[3,float]"


# --- GAPS (documents current wrong/lossy behavior) -------------------------


def test_divider_is_ignored(converter):
    """GAP: _divider (split/binomial/zero/set_value/custom) is dropped.

    A count port declared to split binomially at division is rendered as a
    plain Integer with no divide semantics.
    """
    core, tp = converter
    r = tp(core, {"_default": 0, "_divider": "binomial_ecoli"})
    assert _render(core, r) == "integer"  # divider lost


def test_glob_with_named_sibling_breaks(converter):
    """GAP (latent): a '*' glob mixed with a NAMED sibling at the same level
    is NOT turned into a map — the literal '*' becomes a store child.

    No currently-un-migrated process triggers this, but it is a sharp edge.
    """
    core, tp = converter
    r = tp(core, {"flagella": {"_default": 1}, "*": {"_default": 1}})
    assert isinstance(r, dict) and "*" in r  # literal star child, not a map


def test_set_updater_overwrite_applies_to_reads_too(converter):
    """BUG CLASS: translate_ports has no read/write distinction, so a port a
    process only READS but declares with `_updater: set` (e.g. media_id) is
    rendered overwrite[...] — and the bridge declares that on BOTH inputs and
    outputs, injecting Overwrite structure where a scalar string is expected.
    """
    core, tp = converter
    r = tp(core, {"_default": "", "_updater": "set"})
    assert _render(core, r) == "overwrite[string]"
