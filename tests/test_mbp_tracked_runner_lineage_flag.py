"""The tracked runner's ``single_daughters`` must reach the composite builder.

``scripts/run_mbp_tracked.py`` records ``single_daughters`` in the
``run_identity.json`` sidecar. v2ecoli#591 makes the division bookkeeping an
in-composite Step gated on the *composite's* own ``single_daughters`` flag. If
the runner records the flag but does not forward it, a run takes the old
chunk-dependent path while its provenance claims otherwise -- a lying artifact,
which is worse than a failing one.

These tests pin the seam in both directions, because the failure is silent:
nothing raises, and the sweep reports a clean pass on the unfixed path.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest


def _runner_module():
    p = Path(__file__).resolve().parents[1] / "scripts" / "run_mbp_tracked.py"
    spec = importlib.util.spec_from_file_location("_mbp_tracked_for_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Recorder:
    """Stands in for the composite; records the kwargs it was called with."""

    def __init__(self, accepts_flag: bool):
        self.kwargs = None
        if accepts_flag:
            def fn(core=None, *, single_daughters=False, **kw):
                self.kwargs = {"single_daughters": single_daughters, **kw}
                return {"state": {}}
        else:
            def fn(core=None, **kw):
                self.kwargs = dict(kw)
                return {"state": {}}
        self.fn = fn


class TestBuilderForwardsTheFlag:
    def test_flag_reaches_the_composite_when_the_composite_accepts_it(
        self, monkeypatch
    ):
        mod = _runner_module()
        rec = _Recorder(accepts_flag=True)
        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled", rec.fn
        )

        mod._build_reactor_bird_coupled(None, "out/cache", single_daughters=True)

        assert rec.kwargs is not None, "composite was never called"
        assert rec.kwargs["single_daughters"] is True, (
            "the runner recorded single_daughters but the composite did not "
            "receive it -- the in-composite bookkeeping would be absent while "
            "run_identity.json claimed it was on"
        )

    def test_flag_is_suppressed_when_the_composite_predates_it(self, monkeypatch):
        """A tree predating #591 has no Step to install and the runners own the
        pruning. Forwarding unconditionally would TypeError there, which would
        make this fix impose a landing order on #591."""
        mod = _runner_module()
        rec = _Recorder(accepts_flag=False)
        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled", rec.fn
        )

        mod._build_reactor_bird_coupled(None, "out/cache", single_daughters=True)

        assert rec.kwargs is not None
        assert "single_daughters" not in rec.kwargs


class TestArrestFlagForwarding:
    """v2ecoli#592's `carbon_exhaustion_arrest` is the SECOND instance of the same
    defect: recorded in run_identity, not forwarded. Measured 2026-08-25 -- with it
    unthreaded, an mbp-04 run reports "the arrest does not hold at population scale"
    from a run where the arrest was never enabled."""

    def test_arrest_flag_reaches_the_composite(self, monkeypatch):
        mod = _runner_module()
        seen = {}

        def fn(core=None, *, carbon_exhaustion_arrest=False, **kw):
            seen["arrest"] = carbon_exhaustion_arrest
            return {"state": {}}

        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled", fn
        )
        mod._build_reactor_bird_coupled(
            None, "out/cache", carbon_exhaustion_arrest=True
        )
        assert seen.get("arrest") is True

    def test_arrest_flag_suppressed_when_composite_predates_it(self, monkeypatch):
        mod = _runner_module()
        seen = {}

        def fn(core=None, **kw):
            seen.update(kw)
            return {"state": {}}

        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled", fn
        )
        mod._build_reactor_bird_coupled(
            None, "out/cache", carbon_exhaustion_arrest=True
        )
        assert "carbon_exhaustion_arrest" not in seen


class _Stop(Exception):
    pass


class TestVariantDispatchInjectsTheFlag:
    """``_run_one_variant`` owns the other half: the 20+ variant tuples carry no
    ``single_daughters`` in their ``builder_kwargs``, so the dispatcher injects
    it -- but only for builders that accept it."""

    @staticmethod
    def _call(mod, builder_fn):
        seen = {}

        def spy(core, cache_dir, **kw):
            seen.update(kw)
            raise _Stop

        spy.__signature__ = inspect.signature(builder_fn)
        with pytest.raises(_Stop):
            mod._run_one_variant(
                sim_name="x", study_slug="s", builder_fn=spy, builder_kwargs={},
                extra_root_paths=[], duration_sec=1, max_generations=1, chunk=1,
                cache_dir="out/cache", core=None, emitter="parquet",
                single_daughters=True,
            )
        return seen

    def test_injected_for_a_builder_that_accepts_it(self):
        mod = _runner_module()
        seen = self._call(mod, mod._build_reactor_bird_coupled)
        assert seen.get("single_daughters") is True

    def test_not_injected_for_a_builder_that_does_not(self):
        mod = _runner_module()
        seen = self._call(mod, mod._build_baseline)
        assert "single_daughters" not in seen
