#!/usr/bin/env python3
"""Write out/full_compare/experiments.json from a TSV of `cond<TAB>v2_exp<TAB>ve_exp`.

Emits exactly the shape comparison_report_card.py consumes:
    {"conditions": {cond: [v2_exp, ve_exp]}, ...}  -- a LIST per condition, v2 first.

Used by comparison_harness.sh `launch`; kept as a standalone file so the writer
logic is unit-testable offline (feed it a fake TSV, diff the JSON).

Usage: _write_experiments.py <in.tsv> <out.json>
Reads V2_SIM/VE_SIM/SEEDS/GENS from the environment for provenance (optional).
"""
import datetime
import json
import os
import sys


def build(tsv_path):
    conds = {}
    with open(tsv_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cond, v2, ve = line.split("\t")
            conds[cond] = [v2, ve]  # v2 first, vEcoli second -- report card order
    return conds


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: _write_experiments.py <in.tsv> <out.json>")
    conds = build(argv[1])
    if not conds:
        sys.exit("no conditions parsed from %s" % argv[1])
    doc = {
        "generated": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "v2_simulator_id": os.environ.get("V2_SIM", ""),
        "ve_simulator_id": os.environ.get("VE_SIM", ""),
        "seeds": int(os.environ.get("SEEDS", "0") or 0),
        "gens": int(os.environ.get("GENS", "0") or 0),
        "conditions": conds,
    }
    with open(argv[2], "w") as fh:
        json.dump(doc, fh, indent=2)
    print("wrote", argv[2], "(%d conditions)" % len(conds))


if __name__ == "__main__":
    main(sys.argv)
