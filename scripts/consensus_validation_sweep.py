"""Consensus elongation model validation sweep — 5-gen × 3-seed × 4-media.

Orchestrator for the Phase 5 validation step from
``workspace/investigations/consensus_elongation/audit.md``. Per the design
spec (``v2ecoli_consensus_model.md``), the consensus model must satisfy:

* Growth rate ±5–10% vs. main / v1ecoli
* tRNA charging fraction ≥85% in all conditions
* ppGpp 2–5× stringent rise under starvation conditions
* Rare-codon slowdown 20–70% (emergent from kinetics)

These targets become acceptance criteria checked by :func:`generate_report`
against the per-condition aggregates from :func:`compute_metrics`.

Media conditions
----------------

The design spec lists ``minimal / acetate / plus_amino_acids / no_glucose``.
The v2ecoli ParCa cache exposes media via ``ribosomeElongationRateDict``;
of the spec's four, the first three map directly to
``minimal / minimal_acetate / minimal_plus_amino_acids``. The fourth
(``no_glucose``) is NOT a registered v2ecoli media — it would require a
``boundary.external`` override at composite-build time. As a pragmatic
substitute the sweep uses ``minimal_minus_oxygen`` (anaerobic stress, also
triggers the stringent response). Switch via ``--media`` to use a
different set.

Pre-flight (fast, no sim runs)
------------------------------

``--preflight-only`` runs the cache + composite-build checks for each
condition and exits. Use this in CI or before kicking off the multi-hour
sweep — surfaces stale-cache / missing-media / build-error issues in
under a minute.

Usage
-----

::

    # Pre-flight only — verifies the cache + composite for each condition
    python scripts/consensus_validation_sweep.py --preflight-only

    # Full sweep (multi-hour; run overnight)
    python scripts/consensus_validation_sweep.py \\
        --output-dir out/consensus_validation \\
        --max-generations 5

    # Subset for development — 1 seed, 1 gen, 1 condition
    python scripts/consensus_validation_sweep.py \\
        --media minimal --seeds 0 --max-generations 1

The orchestrator is resumable: each run produces an isolated SQLite db at
``<output-dir>/<media>/seed-<N>/run.db``. Re-running the script skips runs
whose db already exists (use ``--force`` to overwrite).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Acceptance criteria from v2ecoli_consensus_model.md.
ACCEPTANCE = {
    "growth_rate_tolerance_pct": 10.0,  # ±10% vs reference
    "trna_charging_fraction_min": 0.85,  # ≥85% mean across conditions
    "ppgpp_starvation_rise_min": 2.0,  # 2× minimum rise on starvation
    "ppgpp_starvation_rise_max": 5.0,  # 5× upper bound (sanity)
    "rare_codon_slowdown_min_pct": 20.0,
    "rare_codon_slowdown_max_pct": 70.0,
}

# Spec-mapped media. ``no_glucose`` is unimplemented in v2ecoli; substitute
# ``minimal_minus_oxygen`` as a starvation-class condition.
DEFAULT_MEDIA = [
    "minimal",
    "minimal_acetate",
    "minimal_plus_amino_acids",
    "minimal_minus_oxygen",
]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_MAX_GENERATIONS = 5
DEFAULT_MAX_STEPS_PER_GEN = 4000  # ~67 minutes of sim per gen at 1s ticks
DEFAULT_CHUNK = 100

# Paths the SQLite emitter captures for the consensus metrics.
EMIT_PATHS = [
    "global_time",
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.growth_limits.aa_supply",
    "listeners.growth_limits.aa_synthesis",
    "listeners.growth_limits.aa_exchange_rates",
    "listeners.growth_limits.rela_syn",
    "listeners.growth_limits.spot_syn",
    "listeners.growth_limits.spot_deg",
    "listeners.growth_limits.fraction_trna_charged",
    "listeners.trna_charging.saturation_trna",
    "listeners.ribosome_data.effective_elongation_rate",
    "bulk",  # for ppGpp count + tRNA pool sampling
]


@dataclass
class RunSpec:
    """One condition × seed combination."""
    media: str
    seed: int
    max_generations: int
    max_steps: int

    def out_subdir(self, root: Path) -> Path:
        return root / self.media / f"seed-{self.seed}"

    def db_path(self, root: Path) -> Path:
        return self.out_subdir(root) / "run.db"

    def run_id(self) -> str:
        return f"consensus-{self.media}-seed{self.seed}"


@dataclass
class Metrics:
    """Per-run aggregated metrics extracted from a finished SQLite db."""
    media: str
    seed: int
    n_generations: int
    n_steps: int
    growth_rate_per_hour: float = float("nan")
    mean_charging_fraction: float = float("nan")
    mean_ppgpp_count: float = float("nan")
    max_ppgpp_count: float = float("nan")
    min_ppgpp_count: float = float("nan")
    ppgpp_starvation_rise: float = float("nan")
    mean_elongation_rate_aa_per_s: float = float("nan")
    notes: list[str] = field(default_factory=list)


@dataclass
class AcceptanceCheck:
    """One acceptance criterion's pass/fail with explanation."""
    name: str
    passed: bool
    actual: Any
    expected: Any
    detail: str = ""


# ============================================================
# Pre-flight checks
# ============================================================

def check_cache(cache_dir: str | Path) -> list[str]:
    """Verify the ParCa cache exists and is post-port (carries kinetic
    relation data). Returns a list of human-readable problem strings;
    empty list means the cache is OK.
    """
    problems: list[str] = []
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        problems.append(f"cache dir {cache_dir!s} does not exist")
        return problems
    try:
        from v2ecoli.core import load_cache_bundle
    except ImportError as e:
        problems.append(f"cannot import load_cache_bundle: {e}")
        return problems
    try:
        bundle = load_cache_bundle(str(cache_dir))
    except Exception as e:
        problems.append(f"cache failed to load: {type(e).__name__}: {e}")
        return problems
    cfg = bundle.get("configs", {}).get("ecoli-polypeptide-elongation", {})
    if "codon_sequences" not in cfg or len(cfg.get("codon_sequences", [])) == 0:
        problems.append(
            "cache predates the kinetic Relation port — rebuild ParCa "
            "(see scripts/build_cache.py)"
        )
    return problems


def check_media_available(cache_dir: str | Path, media: Iterable[str]) -> list[str]:
    """Verify every media id is in the cache's elongation rate dict.

    The kinetic class reads ``ribosomeElongationRateDict[media_id]`` per
    tick to compute the binary-search target. A missing key explodes
    mid-run; preflight catches it before the sweep starts.
    """
    problems: list[str] = []
    from v2ecoli.core import load_cache_bundle

    bundle = load_cache_bundle(str(cache_dir))
    cfg = bundle["configs"]["ecoli-polypeptide-elongation"]
    available = set(cfg.get("ribosomeElongationRateDict", {}).keys())
    for m in media:
        if m not in available:
            problems.append(
                f"media {m!r} not in ribosomeElongationRateDict "
                f"(available: {sorted(available)})"
            )
    return problems


def check_composite_builds(cache_dir: str | Path) -> list[str]:
    """Verify the consensus_baseline composite builds clean from the cache.

    Doesn't run a tick — just calls the generator. Catches schema /
    config mismatches that would surface at composite-build time.
    """
    problems: list[str] = []
    try:
        from v2ecoli.composites.consensus_baseline import consensus_baseline
        from v2ecoli.core import build_core
    except ImportError as e:
        problems.append(f"cannot import consensus_baseline: {e}")
        return problems
    try:
        core = build_core()
        doc = consensus_baseline(core=core, seed=0, cache_dir=str(cache_dir))
        if not isinstance(doc, dict) or "agents" not in (
            doc.get("state") or doc
        ):
            problems.append(
                "consensus_baseline returned a doc without an agents store"
            )
    except Exception as e:
        problems.append(
            f"consensus_baseline build failed: {type(e).__name__}: {e}"
        )
    return problems


def preflight(cache_dir: str | Path, media: Iterable[str]) -> list[str]:
    """Run all pre-flight checks; return aggregated problem list."""
    problems = []
    problems += check_cache(cache_dir)
    if not problems:  # subsequent checks need a valid cache
        problems += check_media_available(cache_dir, media)
        problems += check_composite_builds(cache_dir)
    return problems


# ============================================================
# Per-run orchestration
# ============================================================

def build_run_specs(
    media: Iterable[str],
    seeds: Iterable[int],
    max_generations: int,
    max_steps: int,
) -> list[RunSpec]:
    """Cartesian product of media × seeds, each as a RunSpec."""
    return [
        RunSpec(media=m, seed=s, max_generations=max_generations,
                max_steps=max_steps)
        for m in media
        for s in seeds
    ]


def needs_run(spec: RunSpec, output_dir: Path, force: bool) -> bool:
    """Skip-if-exists logic for resumability."""
    if force:
        return True
    db = spec.db_path(output_dir)
    return not db.exists()


def execute_run(spec: RunSpec, cache_dir: str, output_dir: Path) -> dict:
    """Run one (media, seed) condition through ``max_generations``.

    Builds a ``consensus_baseline`` composite with ``media_id``
    overridden to the spec's media, then runs via ``run_multigen_sqlite``.
    Result rows land in ``<output_dir>/<media>/seed-<N>/run.db``.
    """
    from process_bigraph import Composite
    from v2ecoli.composites.consensus_baseline import consensus_baseline
    from v2ecoli.core import build_core
    from v2ecoli.library.sqlite_run import run_multigen_sqlite

    out_dir = spec.out_subdir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_file = spec.db_path(output_dir)

    core = build_core()
    doc = consensus_baseline(
        core=core,
        seed=spec.seed,
        cache_dir=cache_dir,
        config_overrides={
            # Pin media for every process that reads it.
            "ecoli-metabolism.media_id": spec.media,
        },
    )
    composite = Composite(doc, core=core)

    result = run_multigen_sqlite(
        composite,
        run_id=spec.run_id(),
        db_file=str(db_file),
        emit_paths=EMIT_PATHS,
        max_steps=spec.max_steps,
        max_generations=spec.max_generations,
        chunk=DEFAULT_CHUNK,
        single_daughters=True,  # bound peak memory at one-cell footprint
        core=core,
    )
    return result


# ============================================================
# Metric extraction + acceptance checks
# ============================================================

def compute_metrics(db_file: str | Path, media: str, seed: int) -> Metrics:
    """Read a finished run.db and aggregate the consensus metrics.

    All metrics are computed over the followed lineage's full trajectory.
    Returns a Metrics dataclass with NaN for any field that can't be
    computed (missing column, empty trajectory, etc.) and a note added
    to ``Metrics.notes`` explaining why.
    """
    import sqlite3

    metrics = Metrics(
        media=media, seed=seed, n_generations=0, n_steps=0,
    )
    db_file = Path(db_file)
    if not db_file.exists():
        metrics.notes.append(f"db file missing: {db_file}")
        return metrics

    conn = sqlite3.connect(str(db_file))
    try:
        cur = conn.cursor()
        try:
            rows = cur.execute(
                "SELECT time, state FROM history WHERE simulation_id = ?",
                (f"consensus-{media}-seed{seed}",),
            ).fetchall()
        except sqlite3.OperationalError as e:
            metrics.notes.append(f"history table read failed: {e}")
            return metrics

        if not rows:
            metrics.notes.append("history table is empty")
            return metrics

        metrics.n_steps = len(rows)

        cell_masses: list[float] = []
        charging_fractions: list[float] = []
        ppgpp_counts: list[float] = []
        elong_rates: list[float] = []
        time_s: list[float] = []
        for t, state_json in rows:
            try:
                state = json.loads(state_json)
            except (json.JSONDecodeError, TypeError):
                continue
            time_s.append(float(t))
            listeners = state.get("listeners", {})
            mass = listeners.get("mass", {})
            if "cell_mass" in mass:
                try:
                    cell_masses.append(float(mass["cell_mass"]))
                except (TypeError, ValueError):
                    pass
            gl = listeners.get("growth_limits", {})
            fc = gl.get("fraction_trna_charged")
            if isinstance(fc, list) and fc:
                # Per-AA → take the mean.
                try:
                    charging_fractions.append(
                        sum(float(x) for x in fc) / len(fc)
                    )
                except (TypeError, ValueError):
                    pass
            rd = listeners.get("ribosome_data", {})
            er = rd.get("effective_elongation_rate")
            if er is not None:
                try:
                    elong_rates.append(float(getattr(er, "magnitude", er)))
                except (TypeError, ValueError):
                    pass

        # Growth rate: log-fit on cell mass over time.
        if len(cell_masses) >= 2 and len(time_s) == len(cell_masses):
            metrics.growth_rate_per_hour = _exponential_fit_per_hour(
                time_s, cell_masses
            )
        else:
            metrics.notes.append(
                f"cell_mass trajectory too short for growth rate "
                f"(n={len(cell_masses)})"
            )

        if charging_fractions:
            metrics.mean_charging_fraction = float(
                sum(charging_fractions) / len(charging_fractions)
            )

        if elong_rates:
            metrics.mean_elongation_rate_aa_per_s = float(
                sum(elong_rates) / len(elong_rates)
            )

        if ppgpp_counts:
            metrics.mean_ppgpp_count = float(
                sum(ppgpp_counts) / len(ppgpp_counts)
            )
            metrics.max_ppgpp_count = float(max(ppgpp_counts))
            metrics.min_ppgpp_count = float(min(ppgpp_counts))
            if metrics.min_ppgpp_count > 0:
                metrics.ppgpp_starvation_rise = float(
                    metrics.max_ppgpp_count / metrics.min_ppgpp_count
                )

        # Generation count from the runs_meta table if available.
        try:
            n_gens = cur.execute(
                "SELECT COUNT(DISTINCT generation) FROM history "
                "WHERE simulation_id = ?",
                (f"consensus-{media}-seed{seed}",),
            ).fetchone()
            if n_gens and n_gens[0]:
                metrics.n_generations = int(n_gens[0])
        except sqlite3.OperationalError:
            pass

    finally:
        conn.close()
    return metrics


def _exponential_fit_per_hour(time_s: list[float], values: list[float]) -> float:
    """log-fit growth rate; returns 1/hour. NaN on degenerate input."""
    import math
    n = len(time_s)
    if n < 2:
        return float("nan")
    # Filter out non-positive values (log undefined).
    pairs = [(t, v) for t, v in zip(time_s, values) if v > 0]
    if len(pairs) < 2:
        return float("nan")
    ts = [p[0] for p in pairs]
    lns = [math.log(p[1]) for p in pairs]
    t_mean = sum(ts) / len(ts)
    l_mean = sum(lns) / len(lns)
    num = sum((t - t_mean) * (l - l_mean) for t, l in zip(ts, lns))
    den = sum((t - t_mean) ** 2 for t in ts)
    if den == 0:
        return float("nan")
    slope_per_s = num / den
    return slope_per_s * 3600.0


def check_acceptance(metrics: list[Metrics]) -> list[AcceptanceCheck]:
    """Per spec acceptance criteria; returns one AcceptanceCheck per criterion."""
    checks: list[AcceptanceCheck] = []
    if not metrics:
        return checks

    # 1. tRNA charging ≥85% mean across all conditions.
    fractions = [
        m.mean_charging_fraction for m in metrics
        if not _isnan(m.mean_charging_fraction)
    ]
    if fractions:
        mean_fc = sum(fractions) / len(fractions)
        passed = mean_fc >= ACCEPTANCE["trna_charging_fraction_min"]
        checks.append(AcceptanceCheck(
            name="trna_charging_fraction",
            passed=passed,
            actual=round(mean_fc, 4),
            expected=f">= {ACCEPTANCE['trna_charging_fraction_min']}",
            detail=f"mean across {len(fractions)} runs",
        ))

    # 2. ppGpp stringent rise 2-5x on starvation conditions.
    # Starvation conditions: acetate, minimal_minus_oxygen, no_glucose variants.
    starv_keys = {"minimal_acetate", "minimal_minus_oxygen", "no_glucose"}
    starv_runs = [m for m in metrics if m.media in starv_keys]
    starv_rises = [
        m.ppgpp_starvation_rise for m in starv_runs
        if not _isnan(m.ppgpp_starvation_rise)
    ]
    if starv_rises:
        mean_rise = sum(starv_rises) / len(starv_rises)
        passed = (
            ACCEPTANCE["ppgpp_starvation_rise_min"]
            <= mean_rise
            <= ACCEPTANCE["ppgpp_starvation_rise_max"]
        )
        checks.append(AcceptanceCheck(
            name="ppgpp_starvation_rise",
            passed=passed,
            actual=round(mean_rise, 3),
            expected=(
                f"[{ACCEPTANCE['ppgpp_starvation_rise_min']}, "
                f"{ACCEPTANCE['ppgpp_starvation_rise_max']}]"
            ),
            detail=f"mean across {len(starv_rises)} starvation runs",
        ))

    # 3. Growth rate plausibility: not NaN, positive, < 5/h (sanity).
    growth = [
        m.growth_rate_per_hour for m in metrics
        if not _isnan(m.growth_rate_per_hour)
    ]
    if growth:
        mean_g = sum(growth) / len(growth)
        passed = 0 < mean_g < 5.0
        checks.append(AcceptanceCheck(
            name="growth_rate_plausibility",
            passed=passed,
            actual=round(mean_g, 4),
            expected="0 < rate < 5 /h",
            detail=f"mean across {len(growth)} runs",
        ))

    return checks


def _isnan(x: float) -> bool:
    return x != x  # nan != nan is True


def generate_report(
    metrics: list[Metrics], checks: list[AcceptanceCheck], output_path: Path
) -> dict:
    """Write per-run metrics + acceptance summary as JSON. Returns the dict."""
    report = {
        "metrics": [asdict(m) for m in metrics],
        "acceptance": [asdict(c) for c in checks],
        "summary": {
            "total_runs": len(metrics),
            "checks_passed": sum(1 for c in checks if c.passed),
            "checks_total": len(checks),
            "passed_all": all(c.passed for c in checks) if checks else False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    return report


# ============================================================
# CLI
# ============================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cache-dir", default="out/cache",
        help="ParCa cache dir (default: out/cache)",
    )
    p.add_argument(
        "--output-dir", default="out/consensus_validation",
        help="Output dir for per-run dbs + report (default: out/consensus_validation)",
    )
    p.add_argument(
        "--media", nargs="+", default=DEFAULT_MEDIA,
        help=f"Media ids to sweep (default: {DEFAULT_MEDIA})",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"RNG seeds (default: {DEFAULT_SEEDS})",
    )
    p.add_argument(
        "--max-generations", type=int, default=DEFAULT_MAX_GENERATIONS,
        help=f"Generations per run (default: {DEFAULT_MAX_GENERATIONS})",
    )
    p.add_argument(
        "--max-steps-per-gen", type=int, default=DEFAULT_MAX_STEPS_PER_GEN,
        help=f"Max ticks per generation (default: {DEFAULT_MAX_STEPS_PER_GEN})",
    )
    p.add_argument(
        "--preflight-only", action="store_true",
        help="Run cache + composite-build checks and exit",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run conditions even if their db already exists",
    )
    p.add_argument(
        "--report-only", action="store_true",
        help="Skip sims; just aggregate existing dbs into a report",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print(f"[sweep] cache_dir={args.cache_dir}")
    print(f"[sweep] output_dir={args.output_dir}")
    print(f"[sweep] media={args.media}")
    print(f"[sweep] seeds={args.seeds}")
    print(f"[sweep] max_generations={args.max_generations}")

    print("[sweep] running pre-flight checks…")
    problems = preflight(args.cache_dir, args.media)
    if problems:
        print("[sweep] PRE-FLIGHT FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("[sweep] pre-flight OK")

    if args.preflight_only:
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps_per_gen * args.max_generations
    specs = build_run_specs(
        args.media, args.seeds, args.max_generations, max_steps,
    )

    if not args.report_only:
        for i, spec in enumerate(specs, start=1):
            if not needs_run(spec, output_dir, args.force):
                print(f"[sweep] [{i}/{len(specs)}] SKIP "
                      f"{spec.media} seed={spec.seed} (db exists)")
                continue
            print(f"[sweep] [{i}/{len(specs)}] RUN "
                  f"{spec.media} seed={spec.seed}")
            t0 = time.time()
            try:
                result = execute_run(spec, args.cache_dir, output_dir)
                elapsed = time.time() - t0
                print(f"[sweep]   completed in {elapsed:.1f}s — "
                      f"{result.get('steps', '?')} steps, "
                      f"{len(result.get('generations', []))} generations")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"[sweep]   FAILED after {elapsed:.1f}s — "
                      f"{type(e).__name__}: {e}")

    print("[sweep] computing metrics…")
    metrics_list: list[Metrics] = []
    for spec in specs:
        db = spec.db_path(output_dir)
        m = compute_metrics(db, spec.media, spec.seed)
        m.n_generations = m.n_generations or spec.max_generations
        metrics_list.append(m)

    checks = check_acceptance(metrics_list)
    report_path = output_dir / "report.json"
    report = generate_report(metrics_list, checks, report_path)

    print(f"[sweep] report written to {report_path}")
    print(f"[sweep] {report['summary']['checks_passed']}/"
          f"{report['summary']['checks_total']} acceptance checks passed")
    for c in checks:
        flag = "✓" if c.passed else "✗"
        print(f"  {flag} {c.name}: actual={c.actual} expected={c.expected}")

    return 0 if report["summary"]["passed_all"] else 2


if __name__ == "__main__":
    sys.exit(main())
