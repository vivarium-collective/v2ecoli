"""Grade dnag-locus-targeted-fix against its five pre-registered tests.

Definitions are taken from the study's own behavior_tests, and the peer set /
deficit come from v2ecoli.library.generational_decay so this study measures the
SAME quantity as dnag-generational-decay (idx_rprotein minus dnaG itself; see
project-dnag-comparison-group-flaw). Nothing here re-derives a definition.
"""
from __future__ import annotations
import glob, json, math, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from v2ecoli.library import generational_decay as gd  # noqa: E402

ROOT = Path("out/dnag-locus-targeted-fix")
CORRECTED, CONTROL = "lexa-edge-removed", "unmodified-control"
CACHE = {CORRECTED: "out/cache_lexafix", CONTROL: "out/cache"}
BASELINE_DEFICIT = 47.4          # ppgpp-off-rescue / dnag-generational-decay
SEEDS = (0, 1, 2)


def taus(arm: str, seed: int) -> dict[int, float]:
    """Per-generation tau (min) parsed from the run log's own summary lines."""
    log = ROOT / arm / f"seed{seed}" / "run.log"
    out = {}
    for m in re.finditer(r"gen (\d+) summary: tau=\s*([\d.]+) min", log.read_text()):
        out[int(m.group(1))] = float(m.group(2))
    return out


def divided(arm: str, seed: int) -> int:
    s = glob.glob(str(ROOT / arm / f"seed{seed}" / "*_summary.json"))
    return json.loads(Path(s[0]).read_text())["generations_completed"]


MIN_TICKS = 500          # a generation below this is a stub, not a generation


def deficit_at(arm: str, gen: int) -> dict[int, float]:
    """{seed: basal-normalised peer deficit at `gen`}.

    Normalising each TU by its own basal_prob is the definition
    dnag-generational-decay pinned at 47.42x. A raw actual-value ratio measures
    a different quantity (it reads ~134x at the same point, because dnaG's basal
    sits 2.6x below the peer-median basal) -- see that study's own spec note.
    """
    pg = gd.per_generation(str(ROOT / arm), seeds=SEEDS, generations=(gen,),
                           cache_dir=CACHE[arm])
    out = {}
    for s, d in pg.items():
        v = d.get(gen)
        if not v or v["n"] < MIN_TICKS:
            continue
        out[s] = (v["peer_median_norm"] / v["dnag_norm"]
                  if v["dnag_norm"] else float("inf"))
    return out


def last_full_gen(arm: str, seed: int) -> int:
    """Highest generation carrying a real number of timesteps (not a stub)."""
    gens = sorted({int(p.split("generation=")[1].split("/")[0])
                   for p in glob.glob(str(ROOT / arm / f"seed{seed}" /
                                          "**/generation=*"), recursive=True)})
    for g in reversed(gens):
        pg = gd.per_generation(str(ROOT / arm), seeds=(seed,), generations=(g,),
                               cache_dir=CACHE[arm])
        v = pg.get(seed, {}).get(g)
        if v and v["n"] >= MIN_TICKS:
            return g
    return gens[-1] if gens else 0


def last_gen_deficit(arm: str) -> dict[int, tuple[int, float]]:
    out = {}
    for seed in SEEDS:
        g = last_full_gen(arm, seed)
        d = deficit_at(arm, g)
        if seed in d:
            out[seed] = (g, d[seed])
    return out


def single_edge_fraction() -> tuple[float, dict]:
    """Fraction of changed delta_prob entries that are the intended LexA edge."""
    import dill
    from v2ecoli.core import build_core
    build_core()
    def dp(cache):
        with open(Path(cache) / "sim_data_cache.dill", "rb") as f:
            sd = dill.load(f)
        ti = sd["configs"]["ecoli-transcript-initiation"]
        D = ti["delta_prob"]          # COO triplets: deltaI/deltaJ/deltaV/shape
        m = {(int(i), int(j)): float(v)
             for i, j, v in zip(D["deltaI"], D["deltaJ"], D["deltaV"])}
        return m, [str(x) for x in ti["rna_data"]["id"]]
    A, ids = dp(CACHE[CONTROL])
    B, _ = dp(CACHE[CORRECTED])
    changed = sorted({k for k in set(A) | set(B)
                      if A.get(k, 0.0) != B.get(k, 0.0)})
    tu = ids.index(gd.DNAG_TU)
    intended = [k for k in changed if k[0] == tu]
    frac = (len(intended) / len(changed)) if changed else 0.0
    return frac, {"n_changed": len(changed),
                  "changed_entries": [list(k) for k in changed[:10]],
                  "changed_values": {str(list(k)): [A.get(k), B.get(k)]
                                     for k in changed[:10]},
                  "dnag_tu_index": int(tu)}


if __name__ == "__main__":
    res = {}

    frac, detail = single_edge_fraction()
    res["intervention-is-a-single-edge"] = {
        "measured": frac, "band": [0.99, 1.0],
        "passed": 0.99 <= frac <= 1.0, "detail": detail}

    corr = last_gen_deficit(CORRECTED)
    after = float(np.median([d for _, d in corr.values()])) if corr else float("nan")
    closed = 1 - math.log(after) / math.log(BASELINE_DEFICIT)
    res["dnag-transcription-recovers"] = {
        "measured": closed, "band": [0.8, 1.0],
        "passed": 0.8 <= closed <= 1.0,
        "detail": {"deficit_after_median": after,
                   "per_seed": {s: {"generation": g, "deficit": d}
                                for s, (g, d) in corr.items()},
                   "baseline_deficit": BASELINE_DEFICIT}}

    dv = {s: divided(CORRECTED, s) for s in SEEDS}
    mn = min(dv.values())
    res["lineage-completes"] = {
        "measured": mn, "band": [8, 12], "passed": 8 <= mn <= 12,
        "detail": {"per_seed": dv}}

    tc = [taus(CORRECTED, s).get(g) for s in SEEDS for g in (1, 2, 3)]
    tk = [taus(CONTROL, s).get(g) for s in SEEDS for g in (1, 2, 3)]
    tc = [x for x in tc if x]; tk = [x for x in tk if x]
    ratio = float(np.median(tc) / np.median(tk))
    res["rescue-is-not-a-global-growth-artefact"] = {
        "measured": ratio, "band": [0.67, 1.5],
        "passed": 0.67 <= ratio <= 1.5,
        "detail": {"median_tau_corrected": float(np.median(tc)),
                   "median_tau_control": float(np.median(tk)),
                   "n_corrected": len(tc), "n_control": len(tk)}}

    ctl = last_gen_deficit(CONTROL)
    base = float(np.median([d for _, d in ctl.values()])) if ctl else float("nan")
    g1 = deficit_at(CONTROL, 1)
    base_g1 = float(np.median(list(g1.values()))) if g1 else float("nan")
    res["dnag-deficit-baseline"] = {
        "measured": base, "threshold": {"op": "<=", "value": 142.2},
        "passed": base <= 142.2,
        "detail": {
            "per_seed_last_gen": {s: {"generation": g, "deficit": d}
                                  for s, (g, d) in ctl.items()},
            "SPEC_DEFECT": (
                "measure.path says 'last generation', but the 47.4x anchor this "
                "pin is 3x tolerance around was measured at GENERATION 1 "
                "(dnag-generational-decay). This study's own established finding "
                "is that dnaG decays across generations, so grading a last-"
                "generation value against a generation-1 anchor compares two "
                "different quantities and must fail by construction. Graded as "
                "written; the drift check the pin intended is reported below."),
            "intended_drift_check_gen1": {
                "measured": base_g1, "anchor": 47.42,
                "ratio_to_anchor": base_g1 / 47.42 if base_g1 == base_g1 else None,
                "passes_142.2": base_g1 <= 142.2,
                "per_seed": {s: d for s, d in g1.items()}}}}

    print(json.dumps(res, indent=2, default=float))
    Path("workspace/studies/dnag-locus-targeted-fix/analyses/outcomes.json"
         ).write_text(json.dumps(res, indent=2, default=float))
