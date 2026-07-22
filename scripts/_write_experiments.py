#!/usr/bin/env python3
"""Write out/full_compare/experiments.json from a TSV of launched experiments.

TSV columns (per condition):
    cond <TAB> v2_exp <TAB> ve_exp [<TAB> seeds <TAB> gens]
The trailing seeds/gens are per-condition (from the spec). When absent they fall
back to the SEEDS/GENS environment variables (legacy --seeds/--gens behaviour).

Emits the shape comparison_report_card.py consumes UNCHANGED:
    {"conditions": {cond: [v2_exp, ve_exp]}, ...}   -- a LIST per condition, v2 first.
PLUS a parallel provenance/meta block the harness `wait` step reads for
per-condition seed counts (the report card ignores unknown keys):
    {"meta": {cond: {"seeds": N, "gens": M}}, "v2ecoli": {...}, "vecoli": {...}}

Kept standalone so the writer is unit-testable offline (feed a fake TSV, diff JSON).
Usage: _write_experiments.py <in.tsv> <out.json>
Reads V2_SIM/VE_SIM/SEEDS/GENS and V2_REPO/V2_COMMIT/V2_BRANCH/VE_* from env.
"""
import datetime
import json
import os
import sys


def build(tsv_path, env=None):
    env = os.environ if env is None else env
    def_seeds = int(env.get("SEEDS", "0") or 0)
    def_gens = int(env.get("GENS", "0") or 0)
    conds, meta = {}, {}
    with open(tsv_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            cond, v2, ve = parts[0], parts[1], parts[2]
            seeds = int(parts[3]) if len(parts) > 3 and parts[3] else def_seeds
            gens = int(parts[4]) if len(parts) > 4 and parts[4] else def_gens
            conds[cond] = [v2, ve]  # v2 first, vEcoli second -- report card order
            meta[cond] = {"seeds": seeds, "gens": gens}
    return conds, meta


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: _write_experiments.py <in.tsv> <out.json>")
    conds, meta = build(argv[1])
    if not conds:
        sys.exit("no conditions parsed from %s" % argv[1])
    env = os.environ
    doc = {
        "generated": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "v2_simulator_id": env.get("V2_SIM", ""),
        "ve_simulator_id": env.get("VE_SIM", ""),
        "v2ecoli": {"repo": env.get("V2_REPO", ""),
                    "commit": env.get("V2_COMMIT", ""),
                    "branch": env.get("V2_BRANCH", "")},
        "vecoli": {"repo": env.get("VE_REPO", ""),
                   "commit": env.get("VE_COMMIT", ""),
                   "branch": env.get("VE_BRANCH", "")},
        "seeds": int(env.get("SEEDS", "0") or 0),  # defaults for back-compat
        "gens": int(env.get("GENS", "0") or 0),
        "conditions": conds,
        "meta": meta,
    }
    with open(argv[2], "w") as fh:
        json.dump(doc, fh, indent=2)
    print("wrote", argv[2], "(%d conditions)" % len(conds))


if __name__ == "__main__":
    main(sys.argv)
