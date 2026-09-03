"""``baseline()`` must thread its own ``cache_dir`` into the injection spec.

An injected process whose config has to be built from the ParCa bundle needs a
path to it, and the resolver looks for that path **on the injection spec** --
not in ``baseline()``'s arguments, which it cannot see.

A caller that declares only ``swap_processes`` therefore gets a process built
with NO config: every ``config_schema`` default fires. The failure then surfaces
far from the cause -- an empty stoichiometry gives empty ``float64`` index
arrays, and the resulting ``IndexError: arrays used as indices must be of
integer (or boolean) type`` is raised inside the solver, hundreds of lines and a
call layer away from the missing key.

This is not hypothetical for batch dispatch: ``batch_baseline_runner`` puts
``cache_dir`` and ``injected_processes`` on the SAME config dict
(``:212``/``:252``) and threads both to every per-seed ``baseline()`` build --
so the runner has the path, and only the spec lacks it.
"""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "v2ecoli" / "composites" / "ecoli_baseline.py"


def _injection_block() -> str:
    """The source of the branch that assembles and applies injected_processes."""
    src = _SRC.read_text()
    i = src.index("assert_injection_sourcing(injected_processes)")
    return src[i:src.index("remove_processes(cell_state", i)]


def test_cache_dir_is_threaded_into_the_spec_before_resolution():
    block = _injection_block()
    assert '"cache_dir": injected_processes.get("cache_dir") or cache_dir' in block, (
        "baseline() must seed the spec's cache_dir from its own, or an injected "
        "process needing a bundle-built config is constructed config-less")
    assert block.index('"cache_dir"') < block.index("resolve_injections("), (
        "the seed must happen BEFORE resolve_injections, or it has no effect")


def test_an_explicit_cache_dir_on_the_spec_still_wins():
    """`or`, not an unconditional assignment: a caller deliberately pointing an
    injected process at a different bundle from the document's must keep it."""
    block = _injection_block()
    assert 'injected_processes.get("cache_dir") or cache_dir' in block
    assert '"cache_dir": cache_dir,' not in block, (
        "unconditional assignment would silently override a deliberate override")


def test_the_spec_is_copied_not_mutated():
    """The caller's dict must not be mutated -- callers reuse it across builds
    (e.g. one spec threaded to every per-seed build in a batch dispatch)."""
    block = _injection_block()
    assert "**injected_processes," in block, "must build a new dict, not mutate"
    assert "injected_processes[" not in block.split("resolve_injections(")[0], (
        "no item assignment on the caller's dict before resolution")


def test_module_still_parses():
    ast.parse(_SRC.read_text())
