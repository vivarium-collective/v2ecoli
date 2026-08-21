"""Build a NAMED ParCa cache with a heterologous gene set INDUCED.

ParCa inserts a new gene *silent* — its ``rna_expression`` entries are exactly
zero, so nothing is transcribed and nothing is translated. The wired v2ecoli
path (``parca_options.new_genes``) is therefore presence-or-absence only: two
builds that differ in how strongly the construct is expressed are
indistinguishable. This script is the other half. It:

  1. hydrates the live SimulationDataEcoli from a ParCa state file (no ParCa
     re-run — the same fast path as scripts/build_cache.py and
     scripts/build_condition_cache.py),
  2. applies ``v2ecoli.perturbations.build_new_gene_cache``, which sets the
     new-gene expression + translation efficiency on a deep copy and writes the
     bundle via ``save_sim_input``,
  3. writes a NEW-GENE MANIFEST (base state, the resolved per-target values,
     git sha) next to the cache so the induction level a run used is provable.

⚠ The state file MUST have been built with new genes. The shipped fixture
``models/parca/parca_state.pkl.gz`` is a basal fit with no new-gene cistrons and
will fail fast with "no new-gene cistrons in this sim_data". Point ``--state``
at a state produced by ``v2ecoli-parca --new-genes <gene_set>``.

⚠ ``--translation-efficiency`` is a WEIGHT, not an achieved rate — the cached
array is L1-normalised across every monomer, so only ratios survive. See the
:mod:`v2ecoli.perturbations.new_gene_cache` docstring.

One induction level per invocation; a design grid is a loop over invocations
(or over ``build_new_gene_cache`` directly, which is why the deep copy lives in
the library function rather than here).

Usage:
    python scripts/build_new_gene_cache.py \
        --state out/parca-gfp/parca_state.pkl \
        --expression 1e6 --translation-efficiency 1.0
    # -> out/cache-new-genes/ + its new_genes.json manifest

    # per-gene relative weights, paired positionally against the new-gene RNAs
    # / monomers in the order v2ecoli.perturbations.new_gene_indices returns
    python scripts/build_new_gene_cache.py --state ... \
        --expression 1e6 --translation-efficiency 1.0 \
        --rel-exp-adj 1,2,4 --rel-trl-eff-adj 1,1,1
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2ecoli.perturbations import build_new_gene_cache
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)

# No default state: unlike build_condition_cache.py there is no shipped fixture
# that would work here (the committed one carries no new genes), so defaulting
# to it would only produce a confusing failure one step later.
DEFAULT_CACHE_DIR = "out/cache-new-genes"


def _weights(raw: str | None, name: str) -> list[float] | None:
    """Parse a comma-separated relative-weight vector, or None."""
    if raw is None:
        return None
    try:
        return [float(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise SystemExit(f"--{name} must be comma-separated numbers: {exc}")


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def build(state_path: str, cache_dir: str, expression: float,
          translation_efficiency: float,
          rel_exp_adj: list[float] | None = None,
          rel_trl_eff_adj: list[float] | None = None,
          seed: int = 0,
          media_condition: str | None = None,
          fixed_media: str | None = None) -> dict:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading ParCa state {state_path} ...")
    state = load_parca_state(state_path)
    print(f"[{time.strftime('%H:%M:%S')}] Hydrating sim_data ...")
    sim_data = hydrate_sim_data_from_state(state)

    print(f"[{time.strftime('%H:%M:%S')}] Inducing new genes "
          f"(expression x{expression}, translation efficiency "
          f"{translation_efficiency}) -> {cache_dir} ...")
    result = build_new_gene_cache(
        sim_data, cache_dir,
        expression=expression,
        translation_efficiency=translation_efficiency,
        rel_exp_adj=rel_exp_adj,
        rel_trl_eff_adj=rel_trl_eff_adj,
        seed=seed,
        condition=media_condition,
        fixed_media=fixed_media,
    )
    applied = result["applied"]
    print(f"    RNAs      {applied['rna_ids']} @ {applied['expression_factors']}")
    print(f"    monomers  {applied['monomer_ids']} @ "
          f"{applied['translation_efficiencies']} (as assigned; the cached "
          f"array is L1-normalised, so only ratios survive)")

    manifest = {
        "created_at": datetime.datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "base_state": state_path,
        "seed": seed,
        "media_condition": media_condition,
        "fixed_media": fixed_media,
        "requested": {
            "expression": expression,
            "translation_efficiency": translation_efficiency,
            "rel_exp_adj": rel_exp_adj,
            "rel_trl_eff_adj": rel_trl_eff_adj,
        },
        # As ASSIGNED to sim_data, not as cached: get_polypeptide_initiation_config
        # normalises the efficiency array, so an "as-cached" number would be
        # cache-relative and would move whenever any other monomer moved.
        "applied": applied,
    }
    manifest_path = os.path.join(cache_dir, "new_genes.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    new-gene manifest -> {manifest_path}")
    print(f"\nTotal: {time.time()-t0:.1f}s")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True,
                    help="parca_state.pkl[.gz] from a build WITH new genes "
                         "(v2ecoli-parca --new-genes ...)")
    ap.add_argument("--cache", dest="cache_dir", default=DEFAULT_CACHE_DIR,
                    help=f"output cache dir (default: {DEFAULT_CACHE_DIR})")
    ap.add_argument("--expression", type=float, required=True,
                    help="multiplier on the baseline new-gene expression")
    ap.add_argument("--translation-efficiency", type=float, required=True,
                    help="efficiency assigned to each new-gene monomer "
                         "(a weight — the cached array is L1-normalised)")
    ap.add_argument("--rel-exp-adj", default=None,
                    help="comma-separated per-RNA relative expression weights "
                         "(default: all 1.0)")
    ap.add_argument("--rel-trl-eff-adj", default=None,
                    help="comma-separated per-monomer relative efficiency "
                         "weights (default: all 1.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--media-condition", default=None,
                    help="ParCa nutrient condition for the initial state / "
                         "doubling time (default basal)")
    ap.add_argument("--fixed-media", default=None,
                    help="media id pinned for the whole run")
    args = ap.parse_args()

    build(args.state, args.cache_dir, args.expression,
          args.translation_efficiency,
          rel_exp_adj=_weights(args.rel_exp_adj, "rel-exp-adj"),
          rel_trl_eff_adj=_weights(args.rel_trl_eff_adj, "rel-trl-eff-adj"),
          seed=args.seed,
          media_condition=args.media_condition,
          fixed_media=args.fixed_media)


if __name__ == "__main__":
    main()
