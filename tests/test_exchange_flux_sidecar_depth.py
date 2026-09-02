"""The exchange-flux sidecar must record LINEAGE DEPTH, not an invocation's count.

A chained run is several invocations over one lineage (the induction seam: a
silent stage, then a stage resumed against a different cache). Every stage
rewrites the sidecar, so a stage recording its own ``max_generations``
understates the lineage — and the comparison card's arm-correspondence check
then refuses the grade as "one of them is stale" while both arms have every
generation on disk.

⚠ The asymmetry is the load-bearing part and is easy to "fix" wrongly: only the
v2ecoli arm accepts ``initial_generation``. The wrapped-reference arm has no
resume hook, so ``max_generations`` IS its depth and applying the offset there
would corrupt it.
"""
import inspect
import re

import scripts.run_comparison_ensemble as rce


def test_a_fresh_run_is_a_strict_no_op():
    # initial_generation is 1-based and inclusive, so the default path must be
    # byte-identical to recording max_generations.
    assert rce._lineage_depth(1, 3) == 3
    assert rce._lineage_depth(1, 1) == 1


def test_a_resumed_stage_reports_the_LINEAGE_not_its_own_count():
    # The real case: a 3-generation chain whose last stage ran generations 2-3.
    assert rce._lineage_depth(2, 2) == 3
    # And a per-generation chain, where every stage runs exactly one.
    assert rce._lineage_depth(3, 1) == 3
    assert rce._lineage_depth(8, 1) == 8


def test_depth_is_not_merely_max_generations():
    # Discriminating: this is the assertion that fails against the old code path,
    # where the sidecar recorded max_generations regardless of where the stage
    # started.
    assert rce._lineage_depth(2, 2) != 2
    assert rce._lineage_depth(5, 4) == 8


def test_the_REAL_call_site_writes_lineage_depth_for_a_RESUMED_stage(tmp_path,
                                                                    monkeypatch):
    """Drive ``main()`` through the real ``run_one`` and read the JSON back.

    ⚠ This is the test that has teeth. Source-text assertions about the call site
    can only see that a token is PRESENT — they cannot see which keyword it feeds,
    so ``seeds=_lineage_depth(...)`` with ``generations=max_generations`` restores
    the bug while reading as fixed. Only reading the written file catches that.

    ⚠ Note the pre-existing spy in ``test_exchange_flux_basis.py`` RE-IMPLEMENTS
    the sidecar call (``generations=kw.get("max_generations")``), so it bypasses
    this call site entirely and cannot catch a regression here. Stub the
    simulation, never the writer.
    """
    import json
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import parallel_seeds as ps
    from v2ecoli.library import xarray_run

    monkeypatch.setattr(rce, "_build_v2ecoli",
                        lambda *a, **k: object())
    monkeypatch.setattr(rce, "extract_v2_build_config",
                        lambda *a, **k: {"n_processes": 0})
    monkeypatch.setattr(xarray_run, "run_multigen_xarray",
                        lambda *a, **k: {"generations": 2})
    # Actually invoke run_one — the default stub in the sibling test returns []
    # without running it, which would leave no sidecar to assert on.
    monkeypatch.setattr(ps, "run_seeds_parallel",
                        lambda seeds, run_one, **kw: [run_one(list(seeds)[0])])

    carry = tmp_path / "carry.json"
    carry.write_text("{}")
    rce.main(["--composite", "v2ecoli", "--condition", "basal",
              "--cache-dir", str(tmp_path), "--n-seeds", "1",
              "--max-generations", "2",
              "--initial-generation", "2",
              "--initial-carry-state", str(carry),
              "--append-store",
              "--out-root", str(tmp_path), "--mode", "serial",
              "--exchange-flux", "product_exchange=X[c]",
              "--exchange-flux-basis", "gdcw"])

    doc = json.loads((tmp_path / "v2ecoli_exchange_flux.json").read_text())
    assert doc["generations"] == 3, (
        f"a stage that ran generations 2-3 must record the LINEAGE depth 3, "
        f"not its own count 2; got {doc!r}")
    assert doc["seeds"] == 1, (
        f"depth must not be written into the wrong key; got {doc!r}")


def test_the_REAL_call_site_is_a_NO_OP_for_an_unchained_run(tmp_path,
                                                            monkeypatch):
    """The default path must be byte-identical to recording max_generations."""
    import json
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import parallel_seeds as ps
    from v2ecoli.library import xarray_run

    monkeypatch.setattr(rce, "_build_v2ecoli", lambda *a, **k: object())
    monkeypatch.setattr(rce, "extract_v2_build_config",
                        lambda *a, **k: {"n_processes": 0})
    monkeypatch.setattr(xarray_run, "run_multigen_xarray",
                        lambda *a, **k: {"generations": 3})
    monkeypatch.setattr(ps, "run_seeds_parallel",
                        lambda seeds, run_one, **kw: [run_one(list(seeds)[0])])

    rce.main(["--composite", "v2ecoli", "--condition", "basal",
              "--cache-dir", str(tmp_path), "--n-seeds", "4",
              "--max-generations", "3",
              "--out-root", str(tmp_path), "--mode", "serial",
              "--exchange-flux", "product_exchange=X[c]",
              "--exchange-flux-basis", "gdcw"])

    doc = json.loads((tmp_path / "v2ecoli_exchange_flux.json").read_text())
    assert (doc["seeds"], doc["generations"]) == (4, 3), doc


def test_the_REFERENCE_ARM_is_not_given_a_resume_generation():
    """The premise the asymmetry rests on, asserted against the ENGINE signature.

    Read from the callee's own signature rather than grepping the caller's
    formatting: a comment containing a ``)`` used to truncate the regex that did
    this, leaving the assertion vacuous exactly when the reference arm HAD been
    wrongly given an offset.
    """
    import inspect

    from v2ecoli.library.vivarium_ecoli_engine import (
        run_vivarium_ecoli_pbg_multigen)

    params = inspect.signature(run_vivarium_ecoli_pbg_multigen).parameters
    assert "max_generations" in params
    assert "initial_generation" not in params, (
        "the reference arm gained a resume hook — its sidecar records "
        "max_generations as the lineage depth, which is now wrong")
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in params.values()), (
        "**kwargs would let a resume generation reach the reference arm "
        "without appearing in its signature")


def test_the_REFERENCE_ARM_sidecar_records_max_generations_VERBATIM(tmp_path,
                                                                    monkeypatch):
    """The CONSEQUENCE of the asymmetry, not only its premise.

    ⚠ Asserting only that the engine has no resume hook leaves the symmetric
    mistake uncaught: giving the reference sidecar ``_lineage_depth(...)`` too.
    That is worse than a plain miss, because ``basis_from_runs`` compares the two
    sidecars TO EACH OTHER and never to the store — so both arms would agree at
    depth 3 while the reference really ran 2 generations from scratch, the
    correspondence check would pass, and the card would grade a shape neither arm
    ran. That is the "both arms were equally wrong so the relative delta looked
    fine" failure this sidecar exists to make visible.

    ⊕ Driven through ``make_run_one`` rather than ``main()`` on purpose: the CLI
    now refuses this combination (see the flag test below), so the only way to
    reach the call-site logic is to call it. The guard protects the command line;
    this protects the arithmetic behind it, and one is not a substitute for the
    other — a later relaxation of the guard must not silently un-cover this.
    """
    import json
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import vivarium_ecoli_engine as vee

    monkeypatch.setattr(vee, "run_vivarium_ecoli_pbg_multigen",
                        lambda *a, **k: {"generations": 2, "build_config": None})

    run_one = rce.make_run_one(
        composite_kind="vecoli", condition="basal", cache_dir=str(tmp_path),
        max_generations=2, max_steps=10, chunk=10, out_root=str(tmp_path),
        n_seeds=1, exchange_fluxes={"product_exchange": "X[c]"},
        exchange_flux_basis="gdcw",
        # A resume generation the reference arm cannot honour.
        initial_generation=3)
    run_one(0)

    doc = json.loads((tmp_path / "vecoli_exchange_flux.json").read_text())
    assert doc["generations"] == 2, (
        "the reference arm has no resume hook — it ran generations 1-2 from "
        "scratch, so its sidecar must say 2. Applying the candidate arm's "
        f"lineage-depth offset here would make both arms agree on a shape "
        f"neither ran; got {doc!r}")


def test_chaining_the_REFERENCE_ARM_is_refused_at_the_FLAG(tmp_path, capsys):
    """A flag mistake must surface as a flag error, not as a card verdict.

    The reference arm silently ignores ``--initial-generation``. Before the
    lineage-depth change both arms recorded ``max_generations``, so passing it to
    both produced matching (equally wrong) sidecars and a silent no-op. Now the
    arms encode different conventions, so the same mis-invocation becomes a
    correspondence refusal reading "one of them is stale" — which sends the
    reader looking at their data instead of their command line.
    """
    import pytest as _pytest
    import scripts.run_comparison_ensemble as rce

    carry = tmp_path / "carry.json"
    carry.write_text("{}")
    with _pytest.raises(SystemExit):
        rce.main(["--composite", "vecoli", "--condition", "basal",
                  "--cache-dir", str(tmp_path), "--n-seeds", "1",
                  "--max-generations", "2", "--initial-generation", "2",
                  "--initial-carry-state", str(carry), "--append-store",
                  "--out-root", str(tmp_path), "--mode", "serial"])
    assert "no resume hook" in capsys.readouterr().err


def test_the_candidate_arm_is_STILL_allowed_to_chain(tmp_path, monkeypatch):
    """The guard must not refuse the invocation the seam exists for."""
    import json
    import scripts.run_comparison_ensemble as rce
    from v2ecoli.library import parallel_seeds as ps
    from v2ecoli.library import xarray_run

    monkeypatch.setattr(rce, "_build_v2ecoli", lambda *a, **k: object())
    monkeypatch.setattr(rce, "extract_v2_build_config",
                        lambda *a, **k: {"n_processes": 0})
    monkeypatch.setattr(xarray_run, "run_multigen_xarray",
                        lambda *a, **k: {"generations": 2})
    monkeypatch.setattr(ps, "run_seeds_parallel",
                        lambda seeds, run_one, **kw: [run_one(list(seeds)[0])])

    carry = tmp_path / "carry.json"
    carry.write_text("{}")
    rce.main(["--composite", "v2ecoli",
              "--condition", "basal", "--cache-dir", str(tmp_path),
              "--n-seeds", "1", "--max-generations", "2",
              "--initial-generation", "2", "--initial-carry-state", str(carry),
              "--append-store", "--out-root", str(tmp_path), "--mode", "serial",
              "--exchange-flux", "product_exchange=X[c]",
              "--exchange-flux-basis", "gdcw"])
    doc = json.loads((tmp_path / "v2ecoli_exchange_flux.json").read_text())
    assert doc["generations"] == 3, doc
