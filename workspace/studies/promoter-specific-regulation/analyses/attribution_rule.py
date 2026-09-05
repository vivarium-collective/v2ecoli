"""Re-run the promoter-specific attribution rule with correct TF name resolution.

Reimplementation: the original analyses/ directory was empty, so the study's
0.9327 / 1.0 / 0.1049 cannot be re-executed. Two checks anchor this version
against the original: lexa-site-position must read 3210729, and
misattributed-lexa-edges-are-removed must come out 1.0.

THE RULE. Regulation is declared in the model as 1462 (TU, TF) pairs, keyed on
cistron content: a TF is attached to every TU containing a regulated cistron.
The promoter-specific rule instead asks whether the TF has a binding site at the
TU's own promoter. For each declared edge:

  * resolve the TF to its gene name, then to the names the binding-site export
    uses (see tf_name_coverage.py -- three separate resolution traps);
  * take the TU's transcription start site, recovered by inverting
    Transcription._get_relative_coordinates on rna_data replication_coordinate
    (validated exactly against the three dnaG TUs);
  * the rule is DIRECTIONAL, not a symmetric window. A repressor cannot occlude
    initiation at a promoter that fires upstream of its operator, so an edge
    survives only if the TF has a site AT or UPSTREAM of the TU's own TSS,
    within WINDOW bp. Upstream is read in the direction of transcription:
    lower coordinates for a + strand TU, higher for a - strand TU.
  * KEEP the edge if such a site exists, REMOVE it otherwise.

  The dnaG locus is the worked case: TU00352 fires at 3210646 with LexA's only
  operator 83 bp DOWNSTREAM at 3210729-3210749, so its edge is removed; TU00435
  fires at 3210735, inside that operator, so its edge survives.

An edge is ADJUDICATED when the TF has at least one positioned site and the TU
has a TSS; otherwise the rule abstains and the edge keeps its cistron-content
assignment.

Run:  .venv/bin/python3 <this> [cache_dir] [parca_state]
"""
from __future__ import annotations
import csv, json, re, sys
import numpy as np, dill, pickle

CACHE = sys.argv[1] if len(sys.argv) > 1 else "out/cache"
STATE = sys.argv[2] if len(sys.argv) > 2 else "out/parca_tudedup_full/parca_state.pkl"
TF_TSV = ".venv/lib/python3.12/site-packages/ecoli_sources/data/flat/transcription_factors.tsv"
EXPORT = "references/all-transcription-factor-binding-sites.txt"
WINDOWS = [50, 200, 1000, 5000, 20000, float('inf')]
DEFAULT_W = 50
ALIASES = {"ihfa": ["IHF"], "ihfb": ["IHF"], "hns": ["H-NS"]}

clean = lambda s: (s or "").strip('"').strip()  # noqa: E731

# --- TF id -> gene name (id columns hold COMMA-SEPARATED lists) --------------
id2name: dict[str, str] = {}
for r in csv.DictReader(open(TF_TSV), delimiter="\t"):
    nm = clean(r["TF"])
    for col in ("oneComponentId", "twoComponentId", "nonMetaboliteBindingId", "activeId"):
        for v in (clean(x) for x in clean(r.get(col)).split(",")):
            if v:
                id2name.setdefault(v, nm)

# --- binding sites ----------------------------------------------------------
sites_by_name: dict[str, list[tuple[int, int]]] = {}
for r in csv.DictReader(open(EXPORT, errors="replace"), delimiter="\t"):
    left, right = (r.get("Left") or "").strip(), (r.get("Right") or "").strip()
    if not left:
        continue
    sites_by_name.setdefault(r["Site"], []).append((int(left), int(right or left)))

def sites_for(gene: str) -> list[tuple[int, int]]:
    names = [gene] + ALIASES.get(gene.lower(), [])
    pat = re.compile("|".join(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])"
                              for n in names), re.I)
    out = []
    for site, coords in sites_by_name.items():
        if pat.search(site):
            out.extend(coords)
    return out

# --- model edges + TU start sites ------------------------------------------
sd = dill.load(open(f"{CACHE}/sim_data_cache.dill", "rb"))
ti = sd["configs"]["ecoli-transcript-initiation"]
ids = [str(x) for x in ti["rna_data"]["id"]]
rc = np.asarray(ti["rna_data"]["replication_coordinate"])
fwd = np.asarray(ti["rna_data"]["is_forward"])
tf_ids = [str(x) for x in sd["configs"]["ecoli-tf-binding"]["tf_ids"]]
dp = ti["delta_prob"]
edges = list(zip(dp["deltaI"].tolist(), dp["deltaJ"].tolist()))

st = pickle.load(open(STATE, "rb"))
tr = st["process"]["transcription"]
ORIC, TERC, GLEN = tr._oric_coordinate, tr._terc_coordinate, tr._genome_length

def genome_coord(rel: int) -> int:
    if rel < 0:
        return rel + ORIC - 1
    return rel + ORIC if rel < GLEN - ORIC else rel + ORIC - GLEN

tss = {i: genome_coord(int(rc[i])) for i in range(len(ids))}

# anchor: the three dnaG TUs must round-trip exactly
for tu, expect in (("TU00352", 3210646), ("TU00434", 3210712), ("TU00435", 3210735)):
    hit = [j for j, x in enumerate(ids) if x.startswith(tu)]
    if hit:
        assert tss[hit[0]] == expect, f"{tu} round-trip {tss[hit[0]]} != {expect}"

tf_sites = {j: sites_for(id2name[t]) for j, t in enumerate(tf_ids) if t in id2name}

def upstream_dist(coord: int, forward: bool, ss: list[tuple[int, int]]) -> float:
    """Distance back to the nearest site at or upstream of `coord`.

    inf when every site of this TF lies downstream -- the case the rule removes.
    """
    best = float("inf")
    for lo, hi in ss:
        if lo <= coord <= hi:
            return 0.0                      # TSS sits inside the operator
        if forward and hi < coord:
            best = min(best, coord - hi)    # operator upstream on the + strand
        elif not forward and lo > coord:
            best = min(best, lo - coord)    # upstream means higher coords on -
    return best

# --- grade ------------------------------------------------------------------
LEXA = tf_ids.index("PC00010")
print(f"cache={CACHE}  TUs={len(ids)}  declared edges={len(edges)}  TFs={len(tf_ids)}")
print(f"TFs with >=1 positioned site: {sum(1 for j in tf_sites if tf_sites[j])}/{len(tf_ids)}")

results = {}
for W in WINDOWS:
    adjudicated = kept = 0
    lexa_upstream = lexa_upstream_removed = 0
    for i, j in edges:
        ss = tf_sites.get(j) or []
        if not ss:
            continue                      # rule abstains
        adjudicated += 1
        d = upstream_dist(tss[i], bool(fwd[i]), ss)
        keep = d <= W
        kept += keep
        if j == LEXA:
            near = min(ss, key=lambda c: min(abs(tss[i] - c[0]), abs(tss[i] - c[1])))
            upstream_of_site = (tss[i] < near[0]) if fwd[i] else (tss[i] > near[1])
            if upstream_of_site:          # promoter fires upstream of the operator
                lexa_upstream += 1
                lexa_upstream_removed += not keep
    results[W] = dict(
        adjudicated_fraction=adjudicated / len(edges),
        unchanged_fraction=kept / adjudicated if adjudicated else 0.0,
        lexa_upstream_removed=(lexa_upstream_removed / lexa_upstream) if lexa_upstream else None,
        adjudicated=adjudicated, kept=kept, lexa_upstream=lexa_upstream)
    tag = "  <-- default" if W == DEFAULT_W else ""
    r = results[W]
    print(f"  W={str(W):>8s} bp   adjudicated={r['adjudicated_fraction']:.4f} "
          f"({adjudicated}/{len(edges)})   unchanged={r['unchanged_fraction']:.4f} "
          f"({kept}/{adjudicated})   lexa_upstream_removed={r['lexa_upstream_removed']}{tag}")

d = results[DEFAULT_W]
print(f"\n--- axes at W={DEFAULT_W} bp (study's recorded values in brackets)")
print(f"  attribution-rule-adjudicates-most-covered-edges  {d['adjudicated_fraction']:.4f}   "
      f"band [0.7, 1.0]    [0.9327]")
print(f"  misattributed-lexa-edges-are-removed            {d['lexa_upstream_removed']}   "
      f"band [0.99, 1.0]   [1.0]")
print(f"  most-edges-are-unchanged                        {d['unchanged_fraction']:.4f}   "
      f"band [0.65, 1.0]   [0.1049]")
json.dump({str(k): v for k, v in results.items()},
          open("workspace/studies/promoter-specific-regulation/analyses/attribution_outcomes.json", "w"),
          indent=2)
