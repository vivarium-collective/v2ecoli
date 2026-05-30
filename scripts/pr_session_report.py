#!/usr/bin/env python3
"""Standard PR session report for v2ecoli.

Generates a self-describing HTML report for a PR / work session:

  * a **provenance banner** (timestamp, git SHA/branch, dirty badge, last
    commit, generator path, host/OS/Python) so the artefact stays meaningful
    once it is months old;
  * **before/after parity plots** — the same short baseline simulation run on
    two git refs (e.g. ``main`` vs the feature branch), overlaid, so a reviewer
    can see at a glance whether a refactor preserved behavior;
  * a free-text **summary** of what the PR changed.

This is intended to be the repo's standard way to attach evidence to a
substantial PR (see AGENTS.md "HTML reports with provenance banners").

Usage
-----
Capture one trajectory (run this on each git ref you want to compare)::

    python scripts/pr_session_report.py capture --out /tmp/after.json --steps 60

    # then, for the "before" ref (the script is copied out so it exists there):
    cp scripts/pr_session_report.py /tmp/prr.py
    git checkout main && python /tmp/prr.py capture --out /tmp/before.json --steps 60
    git checkout -

Render the report from two captures::

    python scripts/pr_session_report.py render \
        --before /tmp/before.json --after /tmp/after.json \
        --out reports/figures/<study>/report.html --title "..." --summary-file notes.md

The ``capture`` subcommand is deliberately self-contained (no imports from
modules that may only exist on one branch) so the same file can run on any ref.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Provenance (house style — mirrors scripts/compare_pdmp_vs_baseline.py)
# ---------------------------------------------------------------------------
def collect_provenance() -> dict:
    """Gather git + runtime metadata for the HTML provenance banner."""
    import platform

    def _git(args, default=""):
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return default

    return {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_short": _git(["rev-parse", "--short", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_git(["status", "--porcelain"])),
        "last_commit_msg": _git(["log", "-1", "--pretty=%s"]),
        "last_commit_author": _git(["log", "-1", "--pretty=%an"]),
        "last_commit_date": _git(["log", "-1", "--pretty=%ci"]),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def render_provenance_banner(prov: dict) -> str:
    repo = "vivarium-collective/v2ecoli"
    commit_url = f"https://github.com/{repo}/commit/{prov['git_sha']}"
    dirty = (
        '<span style="background:#c0392b;color:#fff;padding:2px 8px;'
        'border-radius:4px;font-size:12px;font-weight:600">DIRTY TREE</span>'
        if prov["dirty"] else
        '<span style="background:#27ae60;color:#fff;padding:2px 8px;'
        'border-radius:4px;font-size:12px;font-weight:600">clean</span>'
    )
    return f"""
<div style="border:1px solid #ddd;border-left:5px solid #2c3e50;background:#fafafa;
            padding:12px 16px;margin:0 0 20px;font:13px/1.5 monospace;color:#333">
  <div style="font-weight:700;font-size:14px;margin-bottom:6px">Provenance</div>
  <div>generated: {prov['generated']}</div>
  <div>commit: <a href="{commit_url}">{prov['git_short']}</a>
       ({prov['git_sha']}) on branch <b>{prov['git_branch']}</b> {dirty}</div>
  <div>last commit: "{prov['last_commit_msg']}" — {prov['last_commit_author']},
       {prov['last_commit_date']}</div>
  <div>generator: scripts/pr_session_report.py</div>
  <div>host: {prov['host']} · {prov['os']} · Python {prov['python']}</div>
</div>
""".strip()


# ---------------------------------------------------------------------------
# Capture — run a short baseline sim, record mass observables (self-contained)
# ---------------------------------------------------------------------------
def _mag(x):
    """Plain-float magnitude of a value that may be a bare float or a
    pint.Quantity (so this works on refs where masses are bare floats AND on
    refs where they are Quantities). Duck-typed: no branch-only imports."""
    return float(getattr(x, "magnitude", x))


def _find(d, key, depth=0):
    if not isinstance(d, dict) or depth > 12:
        return None
    if key in d:
        return d[key]
    for v in d.values():
        r = _find(v, key, depth + 1)
        if r is not None:
            return r
    return None


def capture(out_path: str, steps: int, seed: int, cache_dir: str) -> None:
    import v2ecoli

    comp = v2ecoli.build_composite("baseline", seed=seed, cache_dir=cache_dir)
    fields = ["cell_mass", "dry_mass", "protein_mass", "rna_mass", "volume"]
    series: dict[str, list] = {f: [] for f in fields}
    series["time"] = []
    t = 0.0
    for _ in range(steps):
        comp.run(1)
        mass = _find(comp.state, "cell_mass")
        # cell_mass lives in a dict alongside the rest of the mass listener:
        mass_node = None

        def _find_node(d, depth=0):
            if not isinstance(d, dict) or depth > 12:
                return None
            if "cell_mass" in d and "dry_mass" in d:
                return d
            for v in d.values():
                r = _find_node(v, depth + 1)
                if r is not None:
                    return r
            return None

        mass_node = _find_node(comp.state) or {}
        gt = _find(comp.state, "global_time")
        t = _mag(gt) if gt is not None else t + 1.0
        series["time"].append(t)
        for f in fields:
            v = mass_node.get(f)
            series[f].append(_mag(v) if v is not None else float("nan"))

    prov = collect_provenance()
    payload = {"provenance": prov, "steps": steps, "seed": seed, "series": series}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"captured {steps} steps -> {out_path} "
          f"(ref {prov['git_branch']}@{prov['git_short']})")


# ---------------------------------------------------------------------------
# Render — overlay before/after into an HTML report
# ---------------------------------------------------------------------------
def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _parity_plots(before: dict, after: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [("cell_mass", "Cell mass (fg)"), ("dry_mass", "Dry mass (fg)"),
              ("protein_mass", "Protein mass (fg)"), ("volume", "Volume (fL)")]
    bs, as_ = before["series"], after["series"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, label) in zip(axes.ravel(), fields):
        if key in bs:
            ax.plot(bs["time"], bs[key], lw=3, alpha=0.45,
                    label=f"before ({before['provenance']['git_short']})")
        ax.plot(as_["time"], as_[key], lw=1.3, color="#c0392b",
                label=f"after ({after['provenance']['git_short']})")
        ax.set_title(label)
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Before / after parity — overlaid baseline trajectories "
                 "(curves should coincide for a parity-preserving refactor)")
    fig.tight_layout()
    out = _png_b64(fig)
    plt.close(fig)

    # numeric parity table (final-step relative difference)
    rows = []
    for key, label in fields:
        if key not in bs or not bs[key] or not as_[key]:
            continue
        b, a = bs[key][-1], as_[key][-1]
        rel = abs(a - b) / abs(b) if b else float("nan")
        flag = "✅" if (rel == rel and rel < 1e-9) else (
            "≈" if (rel == rel and rel < 1e-3) else "⚠️")
        rows.append(f"<tr><td>{label}</td><td>{b:.6g}</td><td>{a:.6g}</td>"
                    f"<td>{rel:.2e}</td><td>{flag}</td></tr>")
    table = (
        "<table style='border-collapse:collapse;font:13px monospace' border=1 "
        "cellpadding=6><tr><th>field</th><th>before (final)</th>"
        "<th>after (final)</th><th>rel diff</th><th></th></tr>"
        + "".join(rows) + "</table>"
    )
    return (f'<img src="data:image/png;base64,{out}" style="max-width:100%"/>'
            f'<p>Final-step parity:</p>{table}')


def render(before_path, after_path, out_path, title, summary_html) -> None:
    with open(after_path) as fh:
        after = json.load(fh)
    before = None
    if before_path and os.path.exists(before_path):
        with open(before_path) as fh:
            before = json.load(fh)

    prov = collect_provenance()
    parts = [
        "<!doctype html><meta charset=utf-8>",
        f"<title>{title}</title>",
        "<div style='max-width:980px;margin:24px auto;font:15px/1.6 "
        "-apple-system,Segoe UI,Roboto,sans-serif;color:#222;padding:0 16px'>",
        f"<h1>{title}</h1>",
        render_provenance_banner(prov),
        "<h2>Summary</h2>", summary_html,
    ]
    if before is not None:
        parts += ["<h2>Before / after parity</h2>", _parity_plots(before, after)]
    else:
        parts += ["<h2>Trajectory (after)</h2>",
                  "<p><em>No 'before' capture supplied — showing after only.</em></p>",
                  _parity_plots(after, after)]
    parts.append("</div>")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(parts))
    # archival timestamped copy alongside the latest (AGENTS.md convention)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    root, ext = os.path.splitext(out_path)
    archival = f"{root}_{stamp}_{prov['git_short']}{ext}"
    with open(archival, "w") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {out_path}\narchival {archival}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="run a short baseline sim, dump a trajectory JSON")
    c.add_argument("--out", required=True)
    c.add_argument("--steps", type=int, default=60)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--cache-dir", default="out/cache")

    r = sub.add_parser("render", help="overlay before/after into an HTML report")
    r.add_argument("--before", default=None)
    r.add_argument("--after", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--title", default="v2ecoli — PR session report")
    r.add_argument("--summary-file", default=None,
                   help="path to an HTML/markdown-ish snippet for the Summary section")

    a = p.parse_args(argv)
    if a.cmd == "capture":
        capture(a.out, a.steps, a.seed, a.cache_dir)
    elif a.cmd == "render":
        summary = ""
        if a.summary_file and os.path.exists(a.summary_file):
            with open(a.summary_file) as fh:
                summary = fh.read()
        render(a.before, a.after, a.out, a.title, summary)


if __name__ == "__main__":
    sys.exit(main())
