"""scripts/s3_compare_report.py — population v2ecoli↔vEcoli comparison from S3 zarrs.

Lists the per-seed compact-emitter zarr stores each engine wrote to S3
(run_comparison_ensemble.py), opens each into an xarray DataTree, extracts the
per-observable final value per seed, pairs v2ecoli-vs-vEcoli by lineage seed,
and feeds the per-seed relative differences into the EXISTING population-CI
verdict logic in scripts/_compare/multiseed_report.py (_stats / _verdict / TOL).
Emits a JSON report + a human-readable markdown table.

    .venv/bin/python scripts/s3_compare_report.py \
        --experiment cmp-basal-16x16 \
        --bucket smsvpctest-shared-sharedbucket60d199d6-abfvwv0day91 \
        --prefix vecoli-output -o out/s3_compare

Stores are expected at  s3://<bucket>/<prefix>/<experiment>/<engine>_seed<NN>.zarr
Use --local-dir <dir> to aggregate stores already synced down (offline).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the population-CI thresholds — do NOT reimplement them.
from scripts._compare.multiseed_report import _stats, _verdict, TOL
from scripts.run_comparison_ensemble import COMPARISON_PATHS

DEFAULT_OBS = [p.split(".")[-1] for p in COMPARISON_PATHS]  # leaf names in the zarr
ENGINES = ("v2ecoli", "vecoli")
_SEED_RE = re.compile(r"_seed(\d+)\.zarr/?$")


def _storage_options(anon: bool, region: str | None) -> dict:
    opts: dict = {"anon": anon}
    if region:
        opts["client_kwargs"] = {"region_name": region}
    return opts


def list_engine_uris(engine, *, bucket, prefix, experiment, region, anon,
                     local_dir=None) -> list[str]:
    """Return sorted store URIs for one engine (S3 glob, or local dir).

    ``experiment`` is matched as a SUBSTRING of the S3 experiment directory:
    sms-api wraps the submitted experiment_id as ``sim<ID>-<experiment>-<hash>``,
    so an exact name never matches. ``*<experiment>*`` catches the wrapper.
    """
    pat = f"{engine}_seed*.zarr"
    if local_dir:
        base = Path(local_dir) / experiment
        hits = (glob.glob(str(base / pat))
                or glob.glob(str(Path(local_dir) / f"*{experiment}*" / pat))
                or glob.glob(str(Path(local_dir) / pat)))
        return sorted(hits)
    try:
        import s3fs  # noqa: F401
    except ImportError as e:  # fail fast with the fix
        raise SystemExit(
            "s3fs is required to read s3:// stores — `uv pip install s3fs`, "
            "or pass --local-dir after `aws s3 sync`.") from e
    import s3fs
    fs = s3fs.S3FileSystem(**_storage_options(anon, region))
    root = f"{bucket}/{prefix.strip('/')}"
    # exact dir first, then substring-wrapped (sim<ID>-…-<hash>)
    hits = fs.glob(f"{root}/{experiment}/{pat}") or fs.glob(f"{root}/*{experiment}*/{pat}")
    return ["s3://" + h if not h.startswith("s3://") else h for h in sorted(hits)]


def extract_finals(tree, observables, summary):
    """{observable: final_value} for the single lineage node in one store."""
    import numpy as np
    groupset = set(tree.groups)
    lpaths = [p for p in tree.groups if re.search(r"lineage_seed=\d+$", p)]
    if not lpaths:
        return {}
    lpath = lpaths[0]
    out: dict[str, float] = {}
    for obs in observables:
        vpath = f"{lpath}/{obs}"
        if vpath not in groupset:
            continue
        ds = tree[vpath].to_dataset()
        gens = sorted((n for n in ds.data_vars if n.startswith("generation=")),
                      key=lambda n: int(re.search(r"generation=(\d+)", n).group(1)))
        if not gens:
            continue
        if summary == "mean-last-gen":
            arr = np.asarray(ds[gens[-1]]).ravel().astype(float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[obs] = float(arr.mean())
        else:  # "final" — last finite value of the last non-empty generation
            for n in reversed(gens):
                arr = np.asarray(ds[n]).ravel().astype(float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    out[obs] = float(arr[-1])
                    break
    return out


def collect_engine(uris, observables, summary, storage_options):
    """{seed: {observable: final_value}} across an engine's per-seed stores."""
    import xarray as xr
    per_seed: dict[int, dict] = {}
    for uri in uris:
        m = _SEED_RE.search(uri)
        seed = int(m.group(1)) if m else None
        kw = {} if "://" not in uri else {"storage_options": storage_options}
        try:
            tree = xr.open_datatree(uri, engine="zarr", consolidated=False, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] open failed {uri}: {type(e).__name__}: {str(e)[:90]}")
            continue
        finals = extract_finals(tree, observables, summary)
        if finals:
            per_seed[seed] = finals
        else:
            print(f"[warn] no observables extracted from {uri}")
    return per_seed


def paired_deltas(v2, ve, observables):
    """observable -> [per-seed (v2-ve)/ve], paired on common seeds."""
    common = sorted(set(v2) & set(ve))
    out: dict[str, list[float]] = {}
    for obs in observables:
        ds = []
        for s in common:
            a, b = v2[s].get(obs), ve[s].get(obs)
            if a is None or b is None:
                continue
            denom = b if abs(b) > 1e-30 else (a if abs(a) > 1e-30 else None)
            if denom is None:
                continue
            ds.append((a - b) / denom)
        if ds:
            out[obs] = ds
    return out, common


def build_report(deltas, observables):
    """observable -> {n, mean, median, ci_lo, ci_hi, verdict, vals}."""
    rep = {}
    for obs in observables:
        d = deltas.get(obs)
        if not d:
            continue
        st = _stats(d)
        rep[obs] = {"n": st["n"], "mean": st["mean"], "median": st["median"],
                    "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"],
                    "verdict": _verdict(st), "vals": st["vals"]}
    return rep


def to_markdown(rep, *, experiment, n_common, summary):
    def pct(x):
        return "—" if x is None or math.isnan(x) else f"{x * 100:+.1f}%"
    lines = [f"# v2ecoli vs vEcoli — population comparison: `{experiment}`", "",
             f"Paired seeds: **{n_common}** · summary: `{summary}` · "
             f"band ±{TOL * 100:.0f}% · Δ = (v2ecoli − vEcoli)/vEcoli", "",
             "| observable | n | mean Δ | median | 95% CI | verdict |",
             "|---|---:|---:|---:|:---:|:---|"]
    for obs, r in rep.items():
        ci = ("—" if math.isnan(r["ci_lo"])
              else f"[{r['ci_lo'] * 100:+.1f}, {r['ci_hi'] * 100:+.1f}]")
        lines.append(f"| `{obs}` | {r['n']} | {pct(r['mean'])} | "
                     f"{pct(r['median'])} | {ci} | **{r['verdict']}** |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True,
                   help="experiment substring under <prefix>; matched as *<experiment>* "
                        "(sms-api wraps it as sim<ID>-<experiment>-<hash>). Used for both "
                        "engines unless --v2-experiment/--ve-experiment override it.")
    p.add_argument("--v2-experiment", default=None,
                   help="override experiment substring for the v2ecoli engine")
    p.add_argument("--ve-experiment", default=None,
                   help="override experiment substring for the vecoli engine")
    p.add_argument("--bucket",
                   default="smsvpctest-shared-sharedbucket60d199d6-abfvwv0day91")
    p.add_argument("--prefix", default="vecoli-output")
    p.add_argument("--engines", nargs=2, default=list(ENGINES),
                   metavar=("V2", "VE"), help="store name prefixes (v2 first)")
    p.add_argument("--observables", nargs="*", default=DEFAULT_OBS)
    p.add_argument("--summary", choices=("final", "mean-last-gen"), default="final")
    p.add_argument("--vecoli-source", choices=("zarr", "parquet"), default="zarr",
                   help="where the vEcoli engine finals come from: 'zarr' (default, "
                        "compact-emitter stores written by run_comparison_ensemble) or "
                        "'parquet' (vEcoli's Nextflow ParquetEmitter history under "
                        "<prefix>/<ve-experiment>/.../history/). v2ecoli always reads zarr.")
    p.add_argument("--region", default="us-gov-west-1")
    p.add_argument("--anon", action="store_true")
    p.add_argument("--local-dir", default=None,
                   help="read already-synced .zarr stores from here instead of S3")
    p.add_argument("-o", "--out", default="out/s3_compare",
                   help="output path stem (.json + .md are written)")
    args = p.parse_args(argv)

    so = _storage_options(args.anon, args.region)
    v2_eng, ve_eng = args.engines
    exp_for = {v2_eng: args.v2_experiment or args.experiment,
               ve_eng: args.ve_experiment or args.experiment}
    per = {}
    # v2ecoli always reads the compact-emitter zarr stores.
    v2_uris = list_engine_uris(v2_eng, bucket=args.bucket, prefix=args.prefix,
                               experiment=exp_for[v2_eng], region=args.region,
                               anon=args.anon, local_dir=args.local_dir)
    print(f"[{v2_eng}] experiment={exp_for[v2_eng]!r} → {len(v2_uris)} store(s)")
    per[v2_eng] = collect_engine(v2_uris, args.observables, args.summary, so)

    # vEcoli: either the same zarr path, or its Nextflow ParquetEmitter history.
    if args.vecoli_source == "parquet":
        from scripts._compare.vecoli_parquet_reader import read_vecoli_finals
        ve_prefix = f"{args.prefix.strip('/')}/{exp_for[ve_eng]}"
        print(f"[{ve_eng}] parquet history experiment_prefix={ve_prefix!r}")
        per[ve_eng] = read_vecoli_finals(ve_prefix, args.bucket, args.observables,
                                         region=args.region, summary=args.summary)
        print(f"[{ve_eng}] → {len(per[ve_eng])} seed(s) from parquet")
    else:
        ve_uris = list_engine_uris(ve_eng, bucket=args.bucket, prefix=args.prefix,
                                   experiment=exp_for[ve_eng], region=args.region,
                                   anon=args.anon, local_dir=args.local_dir)
        print(f"[{ve_eng}] experiment={exp_for[ve_eng]!r} → {len(ve_uris)} store(s)")
        per[ve_eng] = collect_engine(ve_uris, args.observables, args.summary, so)

    deltas, common = paired_deltas(per[v2_eng], per[ve_eng], args.observables)
    if not common:
        raise SystemExit("no seeds common to both engines — nothing to compare")
    rep = build_report(deltas, args.observables)

    out_json = {"experiment": args.experiment, "bucket": args.bucket,
                "engines": {"v2ecoli": v2_eng, "vecoli": ve_eng},
                "summary": args.summary, "tol": TOL,
                "n_common_seeds": len(common), "common_seeds": common,
                "observables": rep,
                "per_seed_finals": {e: per[e] for e in (v2_eng, ve_eng)}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(out_json, indent=2))
    md = to_markdown(rep, experiment=args.experiment,
                     n_common=len(common), summary=args.summary)
    out.with_suffix(".md").write_text(md)
    print(f"\nwrote {out.with_suffix('.json')}\nwrote {out.with_suffix('.md')}\n")
    print(md)


if __name__ == "__main__":
    main()
