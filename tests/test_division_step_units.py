"""Regression tests for the Division / MarkDPeriod steps.

Two concerns are covered:

1. Quantity[fg] handling — under units-on-ports, listeners.mass.dry_mass is a
   pint Quantity[fg]. Division.next_update must do its arithmetic/comparison in
   plain fg floats; process-bigraph swallows exceptions inside next_update, so a
   raise would silently leave division_threshold stuck and the cell never split.

2. d_period gating — vEcoli divides D_period after replication completes (the
   flag MarkDPeriod raises at the chromosome's division_time) and IGNORES the
   dry-mass threshold (ecoli/processes/cell_division.py). v2's Division step
   defaults to that behavior (``d_period=True``); the legacy mass-distribution
   threshold is used only when ``d_period=False``. Before the fix the mass
   threshold always won, dividing slow-growth cells ~D_period too early.
"""
import numpy as np

from v2ecoli.steps.division import Division, MarkDPeriod
from v2ecoli.types.quantity import ureg as units


def _make_division(d_period=False):
    d = Division.__new__(Division)
    d.initialize({"dry_mass_inc_dict": {"minimal": units.Quantity(150.0, "fg")},
                  "d_period": d_period})
    return d


def _chroms(n):
    # _entryState is int8 in real MetadataArrays (attrs() views it as bool).
    arr = np.zeros(n, dtype=[("division_time", "f8"),
                             ("has_triggered_division", "?"),
                             ("_entryState", "i1")])
    arr["_entryState"] = 1
    return arr


# --- legacy mass-distribution path (d_period=False) --------------------------

def test_threshold_init_handles_quantity_dry_mass():
    """The threshold-init branch returns a numeric threshold (not raise) when
    dry_mass is a pint Quantity[fg]."""
    d = _make_division(d_period=False)
    states = {
        "division_threshold": "mass_distribution",
        "media_id": "minimal",
        "listeners": {"mass": {"dry_mass": units.Quantity(400.0, "fg")}},
    }
    update = d.next_update(1.0, states)
    thr = update["division_threshold"]
    assert isinstance(thr, float)
    assert thr == 550.0          # 400 + 150 * multiplier(=1)


def test_division_check_handles_quantity_dry_mass():
    """dry_mass < threshold must compare without raising (Quantity vs float)."""
    d = _make_division(d_period=False)
    states = {
        "division_threshold": 760.0,
        "listeners": {"mass": {"dry_mass": units.Quantity(400.0, "fg")}},
        "unique": {"full_chromosome": None},
    }
    assert d.next_update(1.0, states) == {}


# --- D-period path (d_period=True, the vEcoli default) ------------------------

def test_d_period_ignores_mass_threshold():
    """Under d_period, a cell over its mass threshold with 2 chromosomes must
    NOT divide unless the D-period `divide` flag is set — the dry-mass threshold
    is ignored. (This is the fix: previously the mass threshold force-divided
    slow-growth cells ~D_period early.)"""
    d = _make_division(d_period=True)
    states = {
        "division_threshold": 100.0,                 # far below dry_mass
        "listeners": {"mass": {"dry_mass": units.Quantity(99999.0, "fg")}},
        "unique": {"full_chromosome": _chroms(2)},   # replication done
        "divide": False,                             # D-period not elapsed
        "global_time": 5000.0,
    }
    assert d.next_update(1.0, states) == {}          # mass ignored -> no division


def test_mark_d_period_raises_divide_at_division_time():
    """MarkDPeriod sets the `divide` flag once global_time reaches the
    chromosome's division_time (with >= 2 full chromosomes)."""
    m = MarkDPeriod.__new__(MarkDPeriod)
    m.initialize({})
    chrom = _chroms(2)
    chrom["division_time"] = [3000.0, 5064.0]
    # before division_time -> no divide
    before = m.next_update(1.0, {"full_chromosome": chrom, "global_time": 2000.0})
    assert not before.get("divide")
    # at/after the earliest untriggered division_time -> divide
    after = m.next_update(1.0, {"full_chromosome": chrom, "global_time": 3000.0})
    assert after.get("divide") is True


# --- injected_processes survive division -------------------------------------

def test_division_threads_injected_processes_to_daughter_baseline(monkeypatch):
    """The DivisionStep must re-supply its ``injected_processes`` spec to each
    daughter's ``baseline()`` rebuild, so a swapped/injected process (e.g.
    metabolism-redux) survives cell division instead of reverting to the plain
    FBA baseline. Before the fix ``_build_daughter_doc`` called ``baseline(...)``
    without ``injected_processes`` and the daughter re-realization crashed with
    ``KeyError: 'current_timeline'`` at the first division.

    Fast proxy for the 40-min multi-gen run: monkeypatch ``baseline`` (and the
    heavy biology helpers) so a division trigger drives ``_build_daughter_doc``
    without any real ParCa/sim, then assert the captured kwargs carry the spec.
    """
    injected = {
        "fork_repo": "/Users/x/vEcoli",
        "fork_sim_data": "/Users/x/simData.cPickle",
        "swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"},
    }

    # 1. initialize() captures the spec off the config.
    d = Division.__new__(Division)
    d.core = None  # normally set by the Edge constructor; unused here (baseline patched)
    d.initialize({"agent_id": "0", "d_period": True,
                  "injected_processes": injected})
    assert d._injected_processes == injected

    # Capture every baseline() call the daughter rebuild makes.
    calls = []

    def _fake_baseline(**kwargs):
        calls.append(kwargs)
        # Minimal doc shape _build_daughter_doc consumes.
        agent = {"listeners": {}, "division": None}
        return {"state": {"agents": {"0": agent}}}

    # next_update imports these lazily from their source modules at call time,
    # so patch the SOURCE symbols (patching the division module would miss them).
    import v2ecoli.composites.ecoli_baseline as _eb
    import v2ecoli.library.division as _libdiv
    import v2ecoli.composites._helpers as _h
    monkeypatch.setattr(_eb, "baseline", _fake_baseline)
    monkeypatch.setattr(_eb, "seed_mass_listener", lambda *a, **k: None)
    _proxy_bulk = {"count": np.array([1, 2, 3])}
    monkeypatch.setattr(_libdiv, "divide_cell",
                        lambda cell_data: ({"bulk": dict(_proxy_bulk)},
                                           {"bulk": dict(_proxy_bulk)}))
    monkeypatch.setattr(_h, "finalize_emitter_for_agent", lambda *a, **k: None)

    # Fabricate a state that fires the d_period division trigger.
    states = {
        "bulk": {}, "unique": {"full_chromosome": _chroms(2)},
        "listeners": {"mass": {"dry_mass": units.Quantity(500.0, "fg")}},
        "environment": {}, "boundary": {},
        "global_time": 3600.0, "divide": True,
    }
    update = d.next_update(1.0, states)

    # Division actually fired (two daughters added), and BOTH daughter baseline()
    # calls received the injected_processes spec unchanged (incl. fork_sim_data,
    # the path build_fork_config needs to re-supply current_timeline).
    assert "_add" in update["agents"]
    assert len(calls) == 2
    for kwargs in calls:
        assert kwargs.get("injected_processes") == injected
        assert kwargs["injected_processes"]["fork_sim_data"] == injected["fork_sim_data"]
