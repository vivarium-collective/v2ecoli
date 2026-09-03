"""Tests for the product-agnostic ``seed_exchange_species`` injection seam.

``environment.exchange`` is a ``map[float]`` store initialised from the ParCa
cache bundle with only the media's external molecules. A bare-float map leaf
ACCUMULATES -- it updates keys that already exist and never ADDS one. So an
injected subsystem that secretes a species the bundle never registered writes
into a key nobody created: the process runs, the composite completes clean, and
every downstream reader sees bit-exact zero with nothing raising.

``injected_processes["seed_exchange_species"]`` lets the CALLER declare the
exchange ids it needs present, so the engine holds no pathway knowledge. The
sibling of ``seed_bulk_species`` for the exchange store.

⚠ These assert on the ACHIEVED store in the built document, not on the request.
A "seeded N stores" log line prints from the requested dict and would pass while
the key never reached the build.
"""
import os

import pytest

# These build a real baseline document, which needs a ParCa cache
# (out/cache/initial_state.json). Mark them `sim` so they run in the
# behavior-tests CI job and are deselected from the cache-less fast-tests job.
pytestmark = pytest.mark.sim

CACHE = "out/cache"

# Invented ids: realistic enough that the assertions stay meaningful, and
# deliberately not any real pathway's exchange species.
SEED_A = "EXAMPLE-PRODUCT-A"
SEED_B = "EXAMPLE-PRODUCT-B"


def _exchange(doc):
    """The ACHIEVED environment.exchange store of a built baseline document.

    ⚠ Note the path: baseline() mounts the cache bundle's initial_state under
    ``agents/<id>``, so the seeded store is at
    ``state.agents.0.environment.exchange`` -- NOT at the top level where the
    pre-build initial_state dict carries it. Asserting on the top-level path
    would fail even when the seed worked."""
    return doc["state"]["agents"]["0"]["environment"]["exchange"]


def test_declared_species_reach_the_built_exchange_store():
    """The declared ids exist in the BUILT document, seeded at 0.0 -- the
    property the accumulate-only map needs and the one a request-side log line
    cannot establish."""
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    core = build_core()
    doc = baseline(core=core, seed=0, cache_dir=CACHE,
                   injected_processes={"fork_repo": "",
                                       "seed_exchange_species": [SEED_A, SEED_B]})
    ex = _exchange(doc)
    assert SEED_A in ex and SEED_B in ex
    assert ex[SEED_A] == 0.0 and ex[SEED_B] == 0.0


def test_absent_declaration_leaves_the_exchange_store_unchanged():
    """Opt-in: no declaration must be byte-identical to a plain build. This is
    what keeps every existing config's behaviour untouched."""
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    core = build_core()
    plain = _exchange(baseline(core=core, seed=0, cache_dir=CACHE))
    empty = _exchange(baseline(core=core, seed=0, cache_dir=CACHE,
                               injected_processes={"fork_repo": "",
                                                   "seed_exchange_species": []}))
    assert plain == empty
    assert SEED_A not in plain


def test_seeding_never_clobbers_a_species_the_bundle_already_carries(monkeypatch):
    """setdefault, not assignment: re-declaring a species the bundle already
    carries must keep the bundle's own value, or this seam would silently reset
    a real initial condition.

    ⚠ The bundle is patched to give that species a NON-ZERO value on purpose.
    Every key in the shipped cache's environment.exchange is 0, so asserting
    against an unmodified bundle compares 0 == 0.0 and passes even if the code
    assigned instead of setdefault-ing — the test could not fail for the reason
    it exists.
    """
    import copy
    from v2ecoli.core import build_core
    from v2ecoli.composites import ecoli_baseline as eb

    real_loader = eb.load_cache_bundle
    marked_value = 7.5
    _store = real_loader(CACHE)["initial_state"]["environment"]["exchange"]
    if not _store:
        pytest.skip("cache bundle ships an empty environment.exchange; nothing "
                    "already-present to test the no-clobber property against")
    existing = next(iter(_store))

    def _patched(cache_dir, *a, **kw):
        bundle = copy.deepcopy(real_loader(cache_dir, *a, **kw))
        bundle["initial_state"]["environment"]["exchange"][existing] = marked_value
        return bundle

    monkeypatch.setattr(eb, "load_cache_bundle", _patched)

    core = build_core()
    reseeded = _exchange(eb.baseline(
        core=core, seed=0, cache_dir=CACHE,
        injected_processes={"fork_repo": "",
                            "seed_exchange_species": [existing, SEED_A]}))
    assert reseeded[existing] == marked_value, (
        "an already-present species was overwritten — this seam must "
        "setdefault, never assign")
    assert reseeded[SEED_A] == 0.0


def test_a_bare_string_declaration_raises_instead_of_seeding_characters():
    """⛔ The highest-value guard here. A bare string is iterable, so
    ``seed_exchange_species: "MY-PRODUCT"`` — the natural single-item form in a
    config — would seed one key PER CHARACTER ('M', 'Y', '-', ...) and never the
    declared species. Every character is a non-empty str, so a per-element check
    alone lets it through. The build then completes clean with a zero product:
    exactly the failure class this seam exists to close.
    """
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    core = build_core()
    with pytest.raises(ValueError, match="takes a LIST"):
        baseline(core=core, seed=0, cache_dir=CACHE,
                 injected_processes={"fork_repo": "",
                                     "seed_exchange_species": "MY-PRODUCT"})


def test_a_compartment_tagged_id_raises_because_no_writer_would_match_it():
    """Writers of environment.exchange strip the compartment
    (``metabolism.py`` emits ``str(molecule[:-3])``), so a tagged id seeds a key
    nothing ever writes to — a clean build with a zero product.

    ⚠ This is the one place the `seed_bulk_species` sibling MISLEADS: its ids
    are compartment-tagged ("X[c]"). A caller declaring both blocks would
    reasonably use the same form in each and get silence from this one.
    """
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    core = build_core()
    with pytest.raises(ValueError, match="BARE species ids"):
        baseline(core=core, seed=0, cache_dir=CACHE,
                 injected_processes={"fork_repo": "",
                                     "seed_exchange_species": ["MY-PRODUCT[c]"]})


@pytest.mark.parametrize("bad", [["", "X"], [None], [123], [{"id": "X"}],
                                 True, {"MY-PRODUCT": 5.0}, "MY-PRODUCT"])
def test_a_malformed_declaration_raises_rather_than_seeding_nothing(bad):
    """A bad entry must fail loudly. Skipping it silently would reproduce the
    exact class this seam exists to close -- a clean run with no product."""
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline
    core = build_core()
    with pytest.raises(ValueError, match="seed_exchange_species"):
        baseline(core=core, seed=0, cache_dir=CACHE,
                 injected_processes={"fork_repo": "",
                                     "seed_exchange_species": bad})


def test_the_premise_an_unseeded_exchange_key_silently_discards_its_write():
    """⭐ The reason this seam exists, asserted rather than assumed.

    ``environment.exchange`` is a bare-float map. Applying an update for a key
    the store does NOT already carry drops it — no error, no warning, no key.
    That is what turns a working secretion into a bit-exact zero downstream, and
    it is why seeding at 0.0 (rather than relying on the writer) is necessary.

    This does not go through baseline(); it pins the underlying type behaviour
    the seeding depends on, so if that behaviour ever changes this test says so
    instead of the seam quietly becoming redundant.
    """
    from v2ecoli.core import build_core
    core = build_core()

    def apply(state, update):
        out = core.apply("map[float]", dict(state), update)
        return out[0] if isinstance(out, tuple) else out

    update = {SEED_A: 2.5}
    seeded = apply({"GLC": 1.0, SEED_A: 0.0}, update)
    unseeded = apply({"GLC": 1.0}, update)

    assert seeded[SEED_A] == 2.5, "a seeded key must accept the write"
    assert SEED_A not in unseeded, (
        "premise broken: the map now ADDS unknown keys, so seeding may be "
        "redundant — re-derive whether this seam is still needed")
    assert unseeded["GLC"] == 1.0
