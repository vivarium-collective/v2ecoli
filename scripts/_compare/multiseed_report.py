"""Aggregate per-seed report-card verdicts into one comprehensive report.

The harness report card is per-lineage (one seed, one 3-/16-gen trajectory),
so a single verdict is dominated by division-timing noise. This module pools N
seeds per condition into population statistics — mean, median, and a 95%
confidence interval on the v2ecoli-vs-vEcoli relative difference — and renders a
single HTML page covering every condition. The population verdict is driven by
the CI, not one noisy seed: ``within_tol`` when the whole 95% CI sits inside
±tol, ``drift`` when the CI straddles ±tol, ``mismatch`` only when the CI lies
entirely beyond ±tol (a genuine systematic difference, not noise).

Inputs are the ``report_card_verdict_*.json`` files the harness already writes
(one per seed per condition). Usage::

    python -m scripts._compare.multiseed_report \
        --workdir out/regen_seeds -o out/multiseed_report.html
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

TOL = 0.05  # ±5% relative band for the population verdict


def _t_crit(df: int) -> float:
    """Two-sided 95% t critical value (small-sample table; ~1.96 for large df)."""
    table = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
             8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09,
             25: 2.06, 30: 2.04, 40: 2.02, 60: 2.00}
    if df <= 0:
        return float("nan")
    for k in sorted(table):
        if df <= k:
            return table[k]
    return 1.96


def _stats(xs: list[float]) -> dict:
    n = len(xs)
    mean = sum(xs) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        sd = math.sqrt(var)
        sem = sd / math.sqrt(n)
        half = _t_crit(n - 1) * sem
    else:
        sd = sem = half = float("nan")
    s = sorted(xs)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "mean": mean, "median": median, "sd": sd, "sem": sem,
            "ci_lo": mean - half, "ci_hi": mean + half, "vals": xs}


def _verdict(st: dict, tol: float = TOL) -> str:
    """Population verdict from the 95% CI of the relative difference."""
    lo, hi = st["ci_lo"], st["ci_hi"]
    if math.isnan(lo):                      # n == 1: fall back to point value
        return "within_tol" if abs(st["mean"]) <= tol else (
            "drift" if abs(st["mean"]) <= 2 * tol else "mismatch")
    if -tol <= lo and hi <= tol:
        return "within_tol"                 # whole CI inside ±tol
    if lo > 2 * tol or hi < -2 * tol:
        return "mismatch"                   # CI entirely beyond ±2·tol
    return "drift"


def collect(workdir: str, axes=("Growth rate", "Cell mass")) -> dict:
    """condition -> axis -> list of per-seed relative differences."""
    out: dict = defaultdict(lambda: defaultdict(list))
    seeds: dict = defaultdict(set)
    for f in glob.glob(str(Path(workdir) / "report_card_verdict_*.json")):
        m = re.search(r"verdict_cond-(.+?)(?:-seed(\d+))?\.json$", Path(f).name)
        if not m:
            continue
        cond = re.sub(r"-(diag|seed\d+)$", "", m.group(1))
        seed = m.group(2) or "0"
        try:
            d = json.load(open(f))
        except Exception:
            continue
        seeds[cond].add(seed)
        for g in d.get("groups", {}).values():
            for a in g.get("axes", []):
                if a["label"] in axes:
                    dr = a.get("detail", {}).get("delta_rel")
                    if dr is not None:
                        out[cond][a["label"]].append(float(dr))
    return {c: {ax: _stats(v) for ax, v in d.items()} for c, d in out.items()}, \
           {c: len(s) for c, s in seeds.items()}


_BADGE = {"within_tol": ("#16a34a", "within tol"),
          "drift": ("#d97706", "drift"),
          "mismatch": ("#dc2626", "mismatch")}


def _bar(st: dict, scale: float = 40.0) -> str:
    """A tiny inline SVG: per-seed dots, mean, and 95% CI on a ±scale% axis."""
    w, h, mid = 260, 34, 130
    def x(p):  # percent -> px
        return mid + max(-1, min(1, p / scale)) * (w / 2 - 8)
    dots = "".join(
        f'<circle cx="{x(v*100):.1f}" cy="{h/2:.0f}" r="2.4" '
        f'fill="#94a3b8" opacity="0.8"/>' for v in st["vals"])
    ci = ""
    if not math.isnan(st["ci_lo"]):
        ci = (f'<line x1="{x(st["ci_lo"]*100):.1f}" y1="{h/2:.0f}" '
              f'x2="{x(st["ci_hi"]*100):.1f}" y2="{h/2:.0f}" '
              f'stroke="#475569" stroke-width="2"/>')
    mean = (f'<circle cx="{x(st["mean"]*100):.1f}" cy="{h/2:.0f}" r="4" '
            f'fill="#1e293b"/>')
    zero = f'<line x1="{mid}" y1="4" x2="{mid}" y2="{h-4}" stroke="#cbd5e1" stroke-dasharray="2 2"/>'
    band = (f'<rect x="{x(-TOL*100):.1f}" y="4" width="{x(TOL*100)-x(-TOL*100):.1f}" '
            f'height="{h-8}" fill="#16a34a" opacity="0.08"/>')
    return (f'<svg width="{w}" height="{h}" style="vertical-align:middle">'
            f'{band}{zero}{ci}{dots}{mean}</svg>')


def render(stats: dict, nseeds: dict, *, title: str, generated_at: str = "") -> str:
    order = ["basal", "with-aa", "with_aa", "succinate", "no-oxygen",
             "no_oxygen", "acetate"]
    conds = sorted(stats, key=lambda c: (order.index(c) if c in order else 99, c))
    rows = []
    for c in conds:
        for ax in ("Cell mass", "Growth rate"):
            st = stats[c].get(ax)
            if not st:
                continue
            v = _verdict(st)
            color, txt = _BADGE[v]
            ci = ("—" if math.isnan(st["ci_lo"]) else
                  f'[{st["ci_lo"]*100:+.1f}, {st["ci_hi"]*100:+.1f}]')
            first = ax == "Cell mass"
            cond_cell = (f'<td rowspan="2" style="font-weight:600;vertical-align:top;'
                         f'padding-top:10px">{c}<br><span style="font-weight:400;'
                         f'color:#64748b;font-size:12px">n={st["n"]} seeds</span></td>'
                         if first else "")
            rows.append(
                f'<tr style="border-top:{"2px solid #e2e8f0" if first else "1px solid #f1f5f9"}">'
                f'{cond_cell}'
                f'<td style="color:#475569">{ax}</td>'
                f'<td style="text-align:right;font-variant-numeric:tabular-nums">'
                f'{st["mean"]*100:+.1f}%</td>'
                f'<td style="text-align:right;color:#64748b;font-variant-numeric:tabular-nums">'
                f'{st["median"]*100:+.1f}%</td>'
                f'<td style="text-align:right;color:#64748b;font-variant-numeric:tabular-nums">'
                f'{ci}</td>'
                f'<td>{_bar(st)}</td>'
                f'<td><span style="background:{color};color:#fff;padding:2px 8px;'
                f'border-radius:10px;font-size:12px">{txt}</span></td></tr>')
    gen = f'<span style="color:#94a3b8;font-size:13px">generated {generated_at}</span>' if generated_at else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title></head>
<body style="font:14px/1.5 system-ui,sans-serif;color:#1e293b;max-width:980px;margin:30px auto;padding:0 16px">
<h1 style="margin-bottom:2px">{title}</h1>{gen}
<p style="color:#475569">Population statistics pooling all seeds per condition. The
relative difference is v2ecoli vs vEcoli; the verdict is driven by the <b>95%
confidence interval</b> (green band = ±{TOL*100:.0f}% tolerance), not any single
noisy lineage. <b>within_tol</b> = whole CI inside ±{TOL*100:.0f}%;
<b>mismatch</b> = CI entirely beyond ±{2*TOL*100:.0f}%; <b>drift</b> between.</p>
<table style="border-collapse:collapse;width:100%">
<thead><tr style="text-align:left;color:#64748b;font-size:13px;border-bottom:2px solid #cbd5e1">
<th style="padding:6px 8px">condition</th><th>axis</th>
<th style="text-align:right">mean Δ</th><th style="text-align:right">median</th>
<th style="text-align:right">95% CI</th>
<th>per-seed · mean · CI (±{40}% axis)</th><th>verdict</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p style="color:#94a3b8;font-size:12.5px;margin-top:18px">Each dot is one seed's
lineage. A wide spread with a CI that brackets 0 means the engines agree on
average and the per-seed scatter is intrinsic division-timing noise (largest for
slow-growth carbon-poor media).</p>
</body></html>"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", action="append", required=True,
                   help="Dir(s) with report_card_verdict_*.json (repeatable).")
    p.add_argument("-o", "--out", default="out/multiseed_report.html")
    p.add_argument("--title", default="v2ecoli vs vEcoli — multi-seed report card")
    args = p.parse_args(argv)

    merged: dict = defaultdict(lambda: defaultdict(list))
    nseeds: dict = defaultdict(int)
    for wd in args.workdir:
        st, ns = collect(wd)
        for c, axd in st.items():
            for ax, s in axd.items():
                merged[c][ax].extend(s["vals"])
            nseeds[c] += ns.get(c, 0)
    stats = {c: {ax: _stats(v) for ax, v in d.items()} for c, d in merged.items()}

    import datetime
    out = render(stats, nseeds, title=args.title,
                 generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out, encoding="utf-8")
    print(f"wrote {args.out}")
    for c in sorted(stats):
        for ax, s in stats[c].items():
            print(f"  {c:<11} {ax:<12} n={s['n']:>2} mean={s['mean']*100:+5.1f}% "
                  f"CI[{s['ci_lo']*100:+.1f},{s['ci_hi']*100:+.1f}] -> {_verdict(s)}")


if __name__ == "__main__":
    main()
