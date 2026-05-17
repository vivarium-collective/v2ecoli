"""TE-sweep summary — memory-lean.

Pulls baseline-te{N}x-seed{N} (and baseline-seed{N} for 1×) sims from
runs.db, groups by TE multiplier, and per-multiplier computes:
  - median DnaA monomer count (second-half pool)
  - mean dnaA mRNA copy number
  - Pearson r between (DnaA TF binding sum) and (dnaA mRNA copy)

Lean: parses each state JSON inline, extracts ONLY the three fields the
sweep needs (monomer_counts[3861], rna_synth_prob.n_bound_TF_per_TU,
rna_counts.mRNA_cistron_counts[227]), and drops the rest. Holds ~600 floats
per sim instead of ~600 full states.
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]

DNAA_MONOMER_IDX  = 3861   # PD03831[c] in monomer_ids
DNAA_TF_IDX       = 12     # MONOMER0-160 in tf_binding.tf_ids
DNAA_CISTRON_IDX  = 227    # EG10235_RNA in mRNA_cistron_ids


def _classify(name: str) -> int | None:
    m = re.match(r"^baseline-te(\d+)x-seed\d+$", name)
    if m:
        return int(m.group(1))
    if re.match(r"^baseline-seed\d+$", name):
        return 1
    return None


def _extract(state_json: str) -> tuple[float, float, float] | None:
    """Pull (dnaA_count, dnaA_TF_binding_sum, dnaA_mRNA) from one state row."""
    try:
        s = json.loads(state_json)
        listeners = s.get("listeners") or {}

        # DnaA count
        mc = listeners.get("monomer_counts")
        if not isinstance(mc, list) or len(mc) <= DNAA_MONOMER_IDX:
            return None
        dnaa_count = float(mc[DNAA_MONOMER_IDX])

        # dnaA TF binding sum (column DNAA_TF_IDX, summed over TUs)
        rsp = listeners.get("rna_synth_prob") or {}
        nb = rsp.get("n_bound_TF_per_TU")
        if isinstance(nb, list) and nb and isinstance(nb[0], list):
            binding = sum(row[DNAA_TF_IDX] for row in nb if len(row) > DNAA_TF_IDX)
        else:
            binding = 0
        binding = float(binding)

        # dnaA mRNA copy number
        rc = listeners.get("rna_counts") or {}
        mrna = rc.get("mRNA_cistron_counts")
        if not isinstance(mrna, list) or len(mrna) <= DNAA_CISTRON_IDX:
            return None
        dnaa_mrna = float(mrna[DNAA_CISTRON_IDX])

        return (dnaa_count, binding, dnaa_mrna)
    except Exception:
        return None


def _pearson(xs, ys):
    if not xs or len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sdx, sdy = statistics.stdev(xs), statistics.stdev(ys)
    if sdx == 0 or sdy == 0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    return cov / (sdx * sdy * (len(xs) - 1))


def main() -> int:
    db_path = STUDY_DIR / "runs.db"
    conn = sqlite3.connect(str(db_path))

    sims = conn.execute(
        "SELECT simulation_id, name FROM simulations ORDER BY name"
    ).fetchall()

    by_mult: dict[int, list[tuple[str, str]]] = {}
    for sim_id, name in sims:
        mult = _classify(name)
        if mult is None:
            continue
        by_mult.setdefault(mult, []).append((sim_id, name))

    if not by_mult:
        print("No baseline-te*-seed* runs in runs.db", file=sys.stderr)
        return 2

    print(f"{'TE×':>5} | {'seeds':>5} | {'samples':>8} | "
          f"{'DnaA median':>11} | {'mRNA mean':>10} | {'pearson r':>10} | {'count':>6} | {'autorep':>8}")
    print("-" * 96)

    for mult in sorted(by_mult.keys()):
        sims_in_mult = by_mult[mult]
        # Streaming extract — second half per sim.
        dnaa_all, binding_all, mrna_all = [], [], []
        for sim_id, _ in sims_in_mult:
            rows = conn.execute(
                "SELECT state FROM history WHERE simulation_id=? ORDER BY step ASC",
                (sim_id,)
            ).fetchall()
            n = len(rows)
            for (state_json,) in rows[n//2:]:
                triple = _extract(state_json)
                if triple is not None:
                    dnaa_all.append(triple[0])
                    binding_all.append(triple[1])
                    mrna_all.append(triple[2])

        if not dnaa_all:
            continue
        med_dnaa = statistics.median(dnaa_all)
        mean_mrna = statistics.mean(mrna_all)
        r = _pearson(binding_all, mrna_all)
        r_str = f"{r:+.3f}" if r is not None else "N/A"

        # Count test: in [300, 800]
        count_pass = 300 <= med_dnaa <= 800
        # Autorep test: r ≤ −0.3
        autorep_pass = r is not None and r <= -0.3

        print(f"{mult:>5}× | {len(sims_in_mult):>5} | {len(dnaa_all):>8} | "
              f"{med_dnaa:>11.0f} | {mean_mrna:>10.3f} | {r_str:>10} | "
              f"{'PASS' if count_pass else 'FAIL':>6} | "
              f"{'PASS' if autorep_pass else 'FAIL':>8}")

    conn.close()
    print()
    print("Sign-flip on pearson r marks the autorepression saturation point.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
