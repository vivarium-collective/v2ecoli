"""Guard: every render script that computes a DnaA NUCLEOTIDE fraction must use the
canonical scripts/dnaa_observables.py — it must not re-derive the fraction inline.

This prevents the class of bug Rashmi caught (2026-06-20): the dnaa-3 readouts chart
computed the DnaA-ATP fraction free-only while dnaa-4 used (free+bound)/total, so the
same run produced contradictory charts. See docs/conventions/dnaa-observable-definitions.md.

A script is considered "nucleotide-fraction-computing" if it references a DnaA bulk
monomer id (the free ATP/ADP/apo pools) AND a fraction token. Such a script must
`import dnaa_observables`. Scripts that legitimately predate box binding (dnaa-0/1/2,
where free == total) or that plot box OCCUPANCY fractions (not the nucleotide fraction)
are allowlisted below with a reason.

Exit non-zero on any violation. Run in the report-prep flow / CI.
"""
from __future__ import annotations

import glob
import os
import re
import sys

DNAA_BULK_IDS = ("MONOMER0-160[c]", "MONOMER0-4565[c]", "PD03831[c]")
FRACTION_TOKENS = ("fraction", "frac_atp", "atp_fraction", "adp_fraction")
CANONICAL_MODULE = "dnaa_observables"

# scripts allowed to compute a fraction WITHOUT the canonical module, with reason.
ALLOWLIST = {
    "render_dnaa1_v12_multiseed.py": "dnaa-1 predates box binding — bulk pools ARE total (free==total).",
    "render_dnaa2_atp_band.py": "dnaa-2 predates box binding — bulk pools ARE total (free==total).",
    "render_dnaa2_sixpanel.py": "dnaa-2 predates box binding — bulk pools ARE total (free==total).",
    "render_dnaa3_occupancy.py": "plots box OCCUPANCY fractions (bound/total per region), not the nucleotide fraction.",
    "render_dnaa3_binding_analysis.py": "plots per-pool OCCUPANCY / free-vs-Kd, not the nucleotide fraction.",
    "render_dnaa_runs.py": "diagnostic CONSOLE print for the pre-binding dnaa-0/1 runs (no bound pools loaded; free==total). Carries a warning comment; do NOT use on dnaa-3+ runs.",
}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    violations, ok, skipped = [], [], []
    for path in sorted(glob.glob(os.path.join(here, "render_dnaa*.py"))):
        name = os.path.basename(path)
        src = open(path).read()
        has_bulk = any(b in src for b in DNAA_BULK_IDS)
        has_frac = any(t in src.lower() for t in FRACTION_TOKENS)
        if not (has_bulk and has_frac):
            continue  # not a nucleotide-fraction-computing script
        imports_canonical = re.search(rf"\bimport\s+{CANONICAL_MODULE}\b", src) is not None
        if imports_canonical:
            ok.append(name)
        elif name in ALLOWLIST:
            skipped.append((name, ALLOWLIST[name]))
        else:
            violations.append(name)

    print("DnaA fraction-source check")
    print(f"  OK (use {CANONICAL_MODULE}): {', '.join(ok) or '-'}")
    for n, why in skipped:
        print(f"  allowlisted: {n} — {why}")
    if violations:
        print("\nVIOLATIONS — these compute a DnaA nucleotide fraction inline and must")
        print(f"import {CANONICAL_MODULE} (or be allowlisted with a reason):")
        for n in violations:
            print(f"  - {n}")
        print("\nSee docs/conventions/dnaa-observable-definitions.md")
        return 1
    print("\nAll DnaA nucleotide-fraction scripts use the canonical definition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
