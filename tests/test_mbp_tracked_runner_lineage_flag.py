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
import sys
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
    """v2ecoli#592's `carbon_exhaustion_arrest` is NEW here, not a pre-existing
    unforwarded flag -- `main`'s runner has no such flag at all. It is added with
    forwarding wired from the start, and with the refusal below, precisely so it
    never becomes a second instance of the `single_daughters` defect.

    The failure it forecloses was measured 2026-08-25 on a #592-merged tree: an
    mbp-04 coupled run reported "the arrest does not hold at population scale"
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

    def test_arrest_refused_when_composite_predates_it(self, monkeypatch):
        """An explicit arrest that cannot be honoured must RAISE, not be dropped.

        This test previously asserted the opposite -- that the flag was silently
        suppressed on a pre-#592 composite. That is the same lying-artifact defect
        this module exists to prevent: unlike `single_daughters`, nothing applies
        the arrest runner-side, so a dropped flag means the run simply has no
        arrest while the operator asked for one.

        Note which guard bites: `_build_reactor_bird_coupled` ALWAYS accepts the
        kwarg, so the dispatcher's signature check passes and only the composite's
        fails. A caller-side guard alone does not cover this.
        """
        mod = _runner_module()

        def fn(core=None, **kw):  # no carbon_exhaustion_arrest parameter
            return {"state": {}}

        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled", fn
        )
        with pytest.raises(ValueError, match="carbon_exhaustion_arrest"):
            mod._build_reactor_bird_coupled(
                None, "out/cache", carbon_exhaustion_arrest=True
            )


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


class TestProvenanceMatchesWhatTheCompositeGot:
    """The point of the change: the sidecar must never claim a flag the composite
    did not receive. These pin the RELATIONSHIP between the two, which the
    forwarding tests above do not -- they stop at the builder boundary.
    """

    @staticmethod
    def _run_and_capture(mod, monkeypatch, tmp_path, builder_fn, **flags):
        """Drive the real dispatcher to the point of writing provenance.

        Returns the ``design`` dict handed to ``write_run_identity`` alongside the
        kwargs the builder actually received, so a test can compare the two.
        """
        got = {}

        def spy_builder(core, cache_dir, **kw):
            got.update(kw)
            return {"state": {}}

        spy_builder.__signature__ = inspect.signature(builder_fn)
        design = {}
        monkeypatch.setattr(
            mod, "write_run_identity",
            lambda *a, **kw: design.update(kw.get("design", {})),
        )
        # Composite is imported inside the function, so patch it at the source.
        monkeypatch.setattr(
            "process_bigraph.Composite", lambda doc, core=None: object()
        )
        runner_got = {}

        def spy_runner(*a, **kw):
            runner_got.update(kw)
            return {"generations": 0, "final_time": 0}

        monkeypatch.setattr(mod, "run_multigen_parquet", spy_runner)
        monkeypatch.setattr(mod, "_count_parquet_rows", lambda *a, **kw: 0)
        # Keep the real dispatcher from mkdir-ing into the repo. It creates only
        # empty dirs, which git never shows -- invisible pollution. REPO_ROOT
        # moves with it because the dispatcher renders the artifact path
        # relative to it.
        monkeypatch.setattr(mod, "STUDIES_ROOT", tmp_path)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        got["__runner__"] = runner_got
        mod._run_one_variant(
            sim_name="x", study_slug="s", builder_fn=spy_builder,
            builder_kwargs={}, extra_root_paths=[], duration_sec=1,
            max_generations=1, chunk=1, cache_dir="out/cache", core=None,
            emitter="parquet", **flags,
        )
        return design, got

    def test_sidecar_never_claims_an_arrest_the_composite_did_not_get(
        self, monkeypatch, tmp_path
    ):
        """The regression, stated behaviourally.

        A builder that cannot take the arrest (14 of 15 variants) must produce a
        sidecar recording ``false`` -- not the operator's request. Pre-fix this
        recorded ``true`` while the composite got nothing.
        """
        mod = _runner_module()
        design, got = self._run_and_capture(
            mod, monkeypatch, tmp_path, mod._build_baseline,
            single_daughters=True, carbon_exhaustion_arrest=True,
        )
        assert "carbon_exhaustion_arrest" not in got
        assert design["carbon_exhaustion_arrest"] is False, (
            "run_identity recorded an arrest the composite never received"
        )

    def test_sidecar_records_the_arrest_when_the_composite_does_get_it(
        self, monkeypatch, tmp_path
    ):
        """The other direction, so the assertion above cannot pass vacuously by
        the sidecar always saying False."""
        mod = _runner_module()
        design, got = self._run_and_capture(
            mod, monkeypatch, tmp_path, mod._build_reactor_bird_coupled,
            single_daughters=True, carbon_exhaustion_arrest=True,
        )
        assert got.get("carbon_exhaustion_arrest") is True
        assert design["carbon_exhaustion_arrest"] is True

    def test_arrest_not_requested_builds_fine_on_a_pre_592_tree(self, monkeypatch):
        """The refusal must be scoped to an explicit True -- the default must stay
        buildable on an old tree, or the guard would impose a landing order on
        #592, which is what the signature guards exist to avoid.

        Does NOT discriminate the fix (it also passes pre-fix); it guards against
        an over-broad raise.
        """
        mod = _runner_module()

        def pre_592(core=None, **kw):
            return {"state": {}}

        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled", pre_592
        )
        assert mod._build_reactor_bird_coupled(None, "out/cache") == {"state": {}}

    @pytest.mark.parametrize("requested", [True, False])
    def test_sidecar_records_the_requested_single_daughters(
        self, monkeypatch, tmp_path, requested
    ):
        """The flag this module is NAMED for, and the one its docstring calls a
        lying artifact -- previously nothing asserted on its sidecar value at all.

        Unlike the arrest, this is truthful even for a builder that cannot take
        the flag, because the runner applies it (see the test below). So the
        sidecar must record the request in both directions.
        """
        mod = _runner_module()
        design, _got = self._run_and_capture(
            mod, monkeypatch, tmp_path, mod._build_baseline,
            single_daughters=requested, carbon_exhaustion_arrest=False,
        )
        assert design["single_daughters"] is requested

    def test_single_daughters_reaches_the_runner_even_when_the_builder_cannot(
        self, monkeypatch, tmp_path
    ):
        """The premise the whole design rests on, finally pinned.

        `single_daughters` needs no refusal because `run_multigen_parquet` prunes
        the sibling lineage runner-side regardless of what the composite accepts
        (parquet_run.py:128,332). If that forwarding is ever dropped, the flag
        silently stops being honoured while the sidecar still records it -- the
        exact defect this module exists to prevent, on the other flag.
        """
        mod = _runner_module()
        _design, got = self._run_and_capture(
            mod, monkeypatch, tmp_path, mod._build_baseline,
            single_daughters=True, carbon_exhaustion_arrest=False,
        )
        assert "single_daughters" not in got, "builder cannot take it"
        assert got["__runner__"].get("single_daughters") is True, (
            "run_multigen_parquet must still receive single_daughters, or the "
            "sidecar's record of it becomes a claim nothing honours"
        )


class TestCliReachesTheDispatcher:
    """The outermost wiring. Everything above calls `_run_one_variant` directly,
    so `main()` could stop passing either flag entirely and every other test here
    would still pass -- the CLI flag would become a silent no-op while
    `run_identity` kept recording the request.
    """

    @staticmethod
    def _main_with(mod, monkeypatch, argv):
        seen = {}

        def spy(**kw):
            seen.update(kw)
            # Shape the end-of-run summary loop consumes.
            return {"sim_name": kw["sim_name"], "wall_time": 0.0,
                    "result_steps": 0, "result_gens": 0, "n_history_rows": 0}

        monkeypatch.setattr(mod, "_run_one_variant", spy)
        monkeypatch.setattr(mod, "build_core", lambda *a, **kw: None)
        monkeypatch.setattr(sys, "argv", ["run_mbp_tracked.py", *argv])
        mod.main()
        return seen

    def test_arrest_flag_reaches_the_dispatcher_on_a_post_592_tree(
        self, monkeypatch
    ):
        """With a composite that can honour it, the CLI value must arrive."""
        mod = _runner_module()

        def post_592(core=None, *, carbon_exhaustion_arrest=False, **kw):
            return {"state": {}}

        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled",
            post_592,
        )
        seen = self._main_with(
            mod, monkeypatch,
            ["--variant", "reactor-bird-coupled-batch-multigen",
             "--carbon-exhaustion-arrest"],
        )
        assert seen.get("carbon_exhaustion_arrest") is True

    def test_preflight_exits_before_any_compute_on_a_pre_592_tree(
        self, monkeypatch
    ):
        """The pre-flight check, pinned.

        The coupled variant is LAST of 15 and carries the longest window, so
        without this the run would burn 14 variants before the builder raised.
        `_run_one_variant` must never be called.

        The composite is STUBBED rather than read from the tree. Reading the live
        signature would make this test pass only until #592 lands and then red
        main's CI -- a landing-order constraint, which is the exact thing the
        signature guards exist to avoid. It would also leave the pre-flight with
        no coverage on the very tree where it matters most.
        """
        mod = _runner_module()

        def pre_592(core=None, **kw):  # no carbon_exhaustion_arrest parameter
            return {"state": {}}

        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled",
            pre_592,
        )
        called = []
        monkeypatch.setattr(
            mod, "_run_one_variant", lambda **kw: called.append(kw)
        )
        monkeypatch.setattr(mod, "build_core", lambda *a, **kw: None)
        monkeypatch.setattr(
            sys, "argv",
            ["run_mbp_tracked.py", "--carbon-exhaustion-arrest"],
        )
        with pytest.raises(SystemExit) as exc:
            mod.main()
        assert "no selected variant can honour it" in str(exc.value)
        assert called == [], "pre-flight must fire before any variant runs"

    def test_arrest_defaults_off_at_the_dispatcher(self, monkeypatch):
        mod = _runner_module()
        seen = self._main_with(
            mod, monkeypatch,
            ["--variant", "reactor-bird-coupled-batch-multigen"],
        )
        assert seen.get("carbon_exhaustion_arrest") is False

    def test_single_daughters_reaches_the_dispatcher(self, monkeypatch):
        mod = _runner_module()
        seen = self._main_with(
            mod, monkeypatch,
            ["--variant", "baseline-reference-multigen"],
        )
        assert seen.get("single_daughters") is True


class TestPreflightScope:
    """The pre-flight narrowed what the previous commit did, and that narrowing
    is a behaviour change worth pinning rather than leaving implicit.
    """

    @staticmethod
    def _main(mod, monkeypatch, argv, composite):
        monkeypatch.setattr(
            "v2ecoli.composites.reactor_bird_coupled.reactor_bird_coupled",
            composite,
        )
        ran = []
        monkeypatch.setattr(
            mod, "_run_one_variant",
            lambda **kw: (ran.append(kw["sim_name"]) or {
                "sim_name": kw["sim_name"], "wall_time": 0.0,
                "result_steps": 0, "result_gens": 0, "n_history_rows": 0,
            }),
        )
        monkeypatch.setattr(mod, "build_core", lambda *a, **kw: None)
        monkeypatch.setattr(sys, "argv", ["run_mbp_tracked.py", *argv])
        return ran

    def test_selection_of_only_inapplicable_variants_refuses(self, monkeypatch):
        """Asking for an arrest on a variant that models no substrate exhaustion
        is a request nothing in the selection can honour, on ANY tree. Refuse
        rather than run something that is not what was asked for.
        """
        mod = _runner_module()

        def post_592(core=None, *, carbon_exhaustion_arrest=False, **kw):
            return {"state": {}}

        ran = self._main(
            mod, monkeypatch,
            ["--variant", "baseline-reference-multigen",
             "--carbon-exhaustion-arrest"],
            post_592,
        )
        with pytest.raises(SystemExit):
            mod.main()
        assert ran == []

    def test_sweep_runs_and_notes_the_inapplicable_variants_post_592(
        self, monkeypatch, capsys
    ):
        """With the coupled variant able to honour it, the sweep proceeds and the
        other 14 are NOTEd rather than refused -- the behaviour review 2 asked
        for, which the pre-flight must narrow without deleting.
        """
        mod = _runner_module()

        def post_592(core=None, *, carbon_exhaustion_arrest=False, **kw):
            return {"state": {}}

        ran = self._main(
            mod, monkeypatch, ["--carbon-exhaustion-arrest"], post_592
        )
        mod.main()
        assert len(ran) == len(mod.VARIANTS)
        assert "baseline-reference-multigen" in ran
