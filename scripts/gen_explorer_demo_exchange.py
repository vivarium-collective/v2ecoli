"""Regenerate the dashboard 'explorer-demo' run WITH environment exchange fluxes
emitted, so the Data Explorer flux map can colour EX_* / uptake reactions.

Reproduces the original explorer-demo emit set (monomer_counts, rna_counts, mass,
base_reaction_fluxes, bulk) and ADDS listeners.fba_results.external_exchange_fluxes.
Also dumps the external-molecule id order (self.externalMoleculeIDs), which the
flux array is aligned to, so the dashboard can label/map the exchange fluxes.

Usage:
    .venv/bin/python scripts/gen_explorer_demo_exchange.py \
        --out-dir .pbg/runs/explorer-demo-exch --variants 2 --max-steps 15
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = "out/cache"
EXPERIMENT_ID = "explorer_demo_exch"
IDS_ASSET = Path("/Users/eranagmon/code/vivarium-dashboard-explorer/"
                 "vivarium_dashboard/static/explorer/exchange_molecule_ids.json")

EMIT_PATHS = [
    "listeners.monomer_counts",
    "listeners.rna_counts.mRNA_counts",
    "listeners.mass.cell_mass", "listeners.mass.dry_mass", "listeners.mass.protein_mass",
    "listeners.mass.rna_mass", "listeners.mass.mRna_mass", "listeners.mass.rRna_mass",
    "listeners.mass.tRna_mass", "listeners.mass.dna_mass", "listeners.mass.smallMolecule_mass",
    "listeners.mass.water_mass", "listeners.mass.volume",
    "listeners.mass.instantaneous_growth_rate",
    "listeners.fba_results.base_reaction_fluxes",
    "listeners.fba_results.external_exchange_fluxes",  # <-- the new one
    "bulk",
]


def _find_external_ids(obj, seen=None, depth=0):
    """Recursively hunt the composite for a metabolism step carrying
    .externalMoleculeIDs (the order the exchange-flux array is aligned to)."""
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return None
    seen.add(id(obj))
    ids = getattr(obj, "externalMoleculeIDs", None)
    if isinstance(ids, (list, tuple)) and ids:
        return [str(x) for x in ids]
    children = []
    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple)):
        children = obj
    else:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            children = d.values()
    for c in children:
        r = _find_external_ids(c, seen, depth + 1)
        if r:
            return r
    return None


def run_variant(variant: int, out_dir: str, max_steps: int, chunk: int):
    from v2ecoli import build_composite
    from v2ecoli.library.parquet_run import run_multigen_parquet
    t0 = time.time()
    composite = build_composite("baseline", cache_dir=CACHE_DIR, seed=variant)
    ids = _find_external_ids(getattr(composite, "state", composite)) or \
          _find_external_ids(composite)
    if ids and not IDS_ASSET.exists():
        IDS_ASSET.write_text(json.dumps(ids, separators=(",", ":")))
        print(f"  wrote {len(ids)} external-molecule ids -> {IDS_ASSET.name}")
    result = run_multigen_parquet(
        composite, experiment_id=EXPERIMENT_ID, out_dir=out_dir,
        emit_paths=EMIT_PATHS, max_steps=max_steps, max_generations=1,
        chunk=chunk, initial_agent_id="0", initial_variant=variant,
        initial_lineage_seed=0,
    )
    print(f"  variant={variant}: {result.get('steps')} steps, "
          f"wall={time.time()-t0:.1f}s", flush=True)
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=".pbg/runs/explorer-demo-exch")
    p.add_argument("--variants", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--chunk", type=int, default=20)
    args = p.parse_args()
    if not Path(CACHE_DIR).is_dir():
        sys.exit(f"cache dir {CACHE_DIR!r} not found")
    out_dir = str(Path(args.out_dir).resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ids = None
    for v in range(args.variants):
        ids = run_variant(v, out_dir, args.max_steps, args.chunk) or ids
    print(f"done -> {out_dir}  (external ids: {len(ids) if ids else 'NOT CAPTURED'})")


if __name__ == "__main__":
    main()
