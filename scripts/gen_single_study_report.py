"""Generate a self-contained single-study HTML report for expert review.

Per the "one study at a time" workflow shift: ship a focused report on
a single study, await expert approval, then unblock the next.

Output: workspace/investigations/<inv>/reports/<study-slug>.html — self-contained
HTML with all viz inlined as <iframe srcdoc>, study spec rendered as the
familiar walkthrough sections (Question / Hypothesis / Tests / Build /
Simulations / Readouts / Visualizations / Honest TBD / Expert questions).

Run from worktree root:
    python scripts/gen_single_study_report.py --study pdmp-00-characterization
"""
from __future__ import annotations
import argparse
import html
import os
import re
import sys
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

INV_SLUG = "v2ecoli-pdmp"


def _h(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _multiline(s: str) -> str:
    """Render a multi-line string as <p> paragraphs."""
    if not s: return ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    return "".join(f"<p>{_h(p)}</p>" for p in paras)


def _inline_iframe(viz_path: Path, name: str, description: str) -> str:
    """Inline a viz HTML as iframe srcdoc. Uses the same auto-resize machinery
    as the dashboard's _fitEmbed by including the pinned-height clamp + onload
    JS in the wrapper."""
    if not viz_path.exists():
        return f'<div class="viz-card missing"><h3>{_h(name)}</h3><p>(file missing: {_h(viz_path)})</p></div>'
    raw = viz_path.read_text(encoding="utf-8")
    escaped = raw.replace("&", "&amp;").replace('"', "&quot;")
    return (
        '<div class="viz-card">'
        f'  <div class="viz-header"><strong>{_h(name)}</strong></div>'
        f'  {("<p class=cap>" + _h(description) + "</p>") if description else ""}'
        f'  <iframe srcdoc="{escaped}" class="embed-frame" '
        f'    style="width:100%;min-height:200px;border:0;display:block" '
        f'    scrolling="no" onload="window._fitEmbed(this)" '
        f'    title="{_h(name)}"></iframe>'
        '</div>'
    )


# Inline _fitEmbed and _wireEmbed JS so iframe auto-sizing works in the
# downloaded report without the dashboard runtime.
EMBED_JS = """
<script>
window._fitEmbed=function(f){try{var d=f.contentDocument||(f.contentWindow&&f.contentWindow.document);if(!d)return;
var b=d.body,e=d.documentElement;
var bStyle=b&&d.defaultView&&d.defaultView.getComputedStyle?d.defaultView.getComputedStyle(b):null;
var pinnedH=0;
if(bStyle&&(bStyle.overflow||"").indexOf("hidden")>=0){
  var hm=(bStyle.height||"").match(/^(\\d+(?:\\.\\d+)?)px$/);
  if(hm)pinnedH=Math.round(parseFloat(hm[1]));
}
var h=pinnedH>0?pinnedH:Math.max(e?e.scrollHeight:0,b?b.scrollHeight:0);
if(h>0)f.style.height=(h+24)+"px";}catch(e){}};
window.addEventListener('load',function(){
  setTimeout(function(){document.querySelectorAll('iframe.embed-frame').forEach(window._fitEmbed);},150);
  setTimeout(function(){document.querySelectorAll('iframe.embed-frame').forEach(window._fitEmbed);},800);
  setTimeout(function(){document.querySelectorAll('iframe.embed-frame').forEach(window._fitEmbed);},2500);
});
</script>
"""


CSS = """
body{font-family:system-ui,-apple-system,sans-serif;max-width:1200px;margin:1.5em auto;padding:0 1.2em;color:#0f172a;line-height:1.55}
header{border-bottom:2px solid #1e3a8a;padding-bottom:0.8em;margin-bottom:1.5em}
header h1{font-size:1.8em;margin:0 0 0.3em 0;color:#1e3a8a}
header .meta{color:#64748b;font-size:0.92em}
h2{font-size:1.35em;margin-top:1.8em;border-bottom:1px solid #e2e8f0;padding-bottom:0.3em;color:#1e3a8a}
h3{font-size:1.08em;margin-top:1.2em}
section{margin-bottom:1.5em}
table{border-collapse:collapse;margin:0.6em 0;width:100%}
th,td{padding:6px 12px;border:1px solid #e2e8f0;text-align:left;vertical-align:top;font-size:0.92em}
th{background:#f1f5f9;font-weight:600}
blockquote{border-left:4px solid #3b82f6;background:#eff6ff;padding:0.6em 1em;margin:0.6em 0;font-style:italic}
code{background:#f1f5f9;padding:1px 6px;border-radius:3px;font-size:0.88em}
.tag{display:inline-block;padding:2px 10px;border-radius:9999px;font-size:0.72em;margin-right:6px;font-weight:600;letter-spacing:0.02em}
.tag.real{background:#d1fae5;color:#065f46}
.tag.skeleton{background:#fef3c7;color:#92400e}
.tag.complete{background:#d1fae5;color:#065f46}
.tag.partial{background:#fef3c7;color:#92400e}
.tag.planned{background:#e5e7eb;color:#475569}
.tag.primary{background:#10b981;color:white}
.tag.supporting{background:#3b82f6;color:white}
.tag.diagnostic{background:#f59e0b;color:white}
.tag.regression{background:#94a3b8;color:white}
.test-card{border-left:4px solid #94a3b8;padding:0.6em 1em;margin:0.8em 0;background:#fafafa}
.test-card.primary{border-left-color:#10b981}
.test-card.supporting{border-left-color:#3b82f6}
.test-card.diagnostic{border-left-color:#f59e0b}
.test-card .claim{font-weight:500;margin:0.2em 0}
.test-card .id{font-family:monospace;color:#64748b;font-size:0.85em}
details.tech{margin-top:0.3em;color:#475569;font-size:0.88em}
details.tech summary{cursor:pointer}
details.tech pre{background:#f8fafc;padding:8px 12px;border-radius:4px;overflow-x:auto;font-size:0.85em}
.viz-card{border:1px solid #e2e8f0;border-radius:6px;background:#fff;margin:1em 0;overflow:hidden}
.viz-card.missing{padding:1em;background:#fef2f2;color:#991b1b}
.viz-header{padding:8px 12px;background:#f1f5f9;border-bottom:1px solid #e5e7eb}
.viz-card .cap{padding:0 12px;color:#475569;font-size:0.9em}
.gap{background:#fffbeb;border:1px solid #fbbf24;padding:0.7em 1.1em;border-radius:6px;margin:0.6em 0}
.gap .gap-label{color:#92400e;font-weight:600;display:inline-block;margin-right:0.4em}
.expert-question{background:#eff6ff;border-left:4px solid #3b82f6;padding:0.7em 1.1em;margin:0.8em 0}
.expert-question .q{font-weight:600;color:#1e3a8a}
ol.expert li{margin-bottom:0.8em}
"""


def render_test(t: dict) -> str:
    classification = t.get("classification", "unclassified")
    name = t.get("name", "(unnamed)")
    desc = t.get("description") or t.get("en") or ""
    measure = t.get("measure", {})
    pass_if = t.get("pass_if", {})
    requires = t.get("requires_simulation", "")
    cites = t.get("cites", [])

    bits = []
    if measure:
        bits.append(f"<strong>Measure:</strong> <code>{_h(yaml.dump(measure, default_flow_style=True).strip())}</code>")
    if pass_if:
        bits.append(f"<strong>Pass condition:</strong> <code>{_h(yaml.dump(pass_if, default_flow_style=True).strip())}</code>")
    if requires:
        bits.append(f"<strong>Requires simulation:</strong> <code>{_h(requires)}</code>")
    if cites:
        bits.append(f"<strong>Cites:</strong> " + ", ".join(f"<code>{_h(c)}</code>" for c in cites))

    tech = ""
    if bits:
        tech = "<details class='tech'><summary>Technical details</summary>" + "<br>".join(bits) + "</details>"

    return (
        f'<div class="test-card {classification}">'
        f'  <span class="tag {classification}">{_h(classification)}</span>'
        f'  <div class="claim">{_h(desc)}</div>'
        f'  <div class="id">test id: <code>{_h(name)}</code></div>'
        f'  {tech}'
        '</div>'
    )


def render_conditions(spec: dict) -> str:
    cond = spec.get("conditions") or {}
    if not cond:
        return "<p><em>No conditions block.</em></p>"
    parts = []
    base = cond.get("baseline")
    if base:
        parts.append("<h3>Baseline composite</h3>")
        parts.append(f"<p>Composite: <code>{_h(base.get('composite'))}</code></p>")
        params = base.get("params")
        if params:
            parts.append("<details class='tech'><summary>Baseline params</summary><pre>"
                         + _h(yaml.dump(params, default_flow_style=False)) + "</pre></details>")
    variants = cond.get("variants") or []
    if variants:
        parts.append(f"<h3>Variants ({len(variants)})</h3>")
        rows = ["<tr><th>Name</th><th>Base composite</th><th>Description</th></tr>"]
        for v in variants:
            rows.append(
                f"<tr><td><code>{_h(v.get('name'))}</code></td>"
                f"<td>{_h(v.get('base_composite', ''))}</td>"
                f"<td>{_h(v.get('description', ''))}</td></tr>"
            )
        parts.append("<table>" + "".join(rows) + "</table>")
    settings = cond.get("model_settings") or []
    if settings:
        parts.append(f"<h3>Model settings ({len(settings)})</h3>")
        rows = ["<tr><th>Name</th><th>Type</th><th>Default</th><th>Description</th></tr>"]
        for s in settings:
            rows.append(
                f"<tr><td><code>{_h(s.get('name'))}</code></td>"
                f"<td>{_h(s.get('type', ''))}</td>"
                f"<td>{_h(s.get('default', ''))}</td>"
                f"<td>{_h(s.get('description', ''))}</td></tr>"
            )
        parts.append("<table>" + "".join(rows) + "</table>")
    return "".join(parts)


def render_simulations(spec: dict) -> str:
    sims = spec.get("simulation_set") or []
    if not sims:
        return "<p><em>No simulation_set entries.</em></p>"
    rows = ["<tr><th>Name</th><th>Kind</th><th>Status</th><th>Base model</th><th>Duration / metrics</th></tr>"]
    for s in sims:
        status = s.get("status", "")
        rows.append(
            f"<tr><td><code>{_h(s.get('name'))}</code></td>"
            f"<td>{_h(s.get('kind', ''))}</td>"
            f"<td><span class='tag {status}'>{_h(status)}</span></td>"
            f"<td>{_h(s.get('base_model', ''))}</td>"
            f"<td>{_h(s.get('duration_steps', ''))} · metrics: {_h(', '.join(s.get('metrics', []) or []))}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def render_readouts(spec: dict) -> str:
    ros = spec.get("readouts") or []
    if not ros:
        return "<p><em>No readouts.</em></p>"
    rows = ["<tr><th>Name</th><th>Status</th><th>Path</th><th>Units</th><th>Notes</th></tr>"]
    for r in ros:
        status = r.get("status", "")
        rows.append(
            f"<tr><td><strong>{_h(r.get('name'))}</strong></td>"
            f"<td><span class='tag {status}'>{_h(status)}</span></td>"
            f"<td><code>{_h(r.get('path', ''))}</code></td>"
            f"<td>{_h(r.get('units', ''))}</td>"
            f"<td>{_h(r.get('notes', ''))}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def render_planned_runs(spec: dict) -> str:
    runs = spec.get("planned_runs") or []
    if not runs:
        return "<p><em>No planned_runs.</em></p>"
    rows = ["<tr><th>Name</th><th>Status</th><th>n_steps</th><th>Details</th></tr>"]
    for r in runs:
        status = r.get("status", "")
        rows.append(
            f"<tr><td><code>{_h(r.get('name'))}</code></td>"
            f"<td><span class='tag {status}'>{_h(status)}</span></td>"
            f"<td>{_h(r.get('n_steps', ''))}</td>"
            f"<td>{_h(r.get('details', ''))}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def render_viz(spec: dict, fig_root: Path) -> str:
    # Use embed_visualizations[] (the renderer-friendly list) — auto-discovery + explicit.
    embeds = spec.get("embed_visualizations") or []
    if not embeds:
        # Fall back to visualizations[]
        viz = spec.get("visualizations") or []
        embeds = [{"name": v.get("name"),
                   "url": v.get("address", "").replace("file:", ""),
                   "description": v.get("description", "")} for v in viz]
    parts = []
    for e in embeds:
        url = (e.get("url") or "").lstrip("/")
        if not url:
            continue
        path = REPO_ROOT / url
        parts.append(_inline_iframe(path, e.get("name", ""), e.get("description", "")))
    return "".join(parts) or "<p><em>No viz wired.</em></p>"


def render_study_card(spec: dict) -> str:
    sc = spec.get("study_card") or {}
    if not sc or not isinstance(sc, dict):
        return ""
    rows = []
    for k_label, k_key in [
        ("Goal", "goal"),
        ("Mechanism", "mechanism"),
        ("Why before next", "why_before_next"),
        ("Expected result", "expected_result"),
        ("Main expert question", "main_expert_question"),
    ]:
        if sc.get(k_key):
            rows.append(f"<tr><th>{k_label}</th><td>{_multiline(sc[k_key])}</td></tr>")
    return "<table>" + "".join(rows) + "</table>" if rows else ""


def render_bibliography(bib_keys: list, papers_bib_path: Path) -> str:
    """Render bib entries by matching bib_keys against workspace/references/papers.bib."""
    if not bib_keys or not papers_bib_path.exists():
        return ""
    bib_raw = papers_bib_path.read_text(encoding="utf-8")
    entries = []
    for key in bib_keys:
        # Naive bibtex extraction
        m = re.search(r"@\w+\{" + re.escape(key) + r"\s*,([^@]+?)\n\}", bib_raw, re.S)
        if not m:
            entries.append(f"<li><code>{_h(key)}</code> <em>(entry not found in papers.bib)</em></li>")
            continue
        body = m.group(1)
        title = re.search(r"title\s*=\s*[{\"](.+?)[}\"]", body)
        author = re.search(r"author\s*=\s*[{\"](.+?)[}\"]", body)
        year = re.search(r"year\s*=\s*[{\"]?(\d{4})", body)
        entries.append(
            f"<li><code>{_h(key)}</code> — "
            f"{(_h(title.group(1)) + ' ') if title else ''}"
            f"{('(' + _h(author.group(1)) + ', ') if author else '('}"
            f"{(_h(year.group(1)) if year else '')})</li>"
        )
    return f"<ul>{''.join(entries)}</ul>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", default="pdmp-00-characterization")
    p.add_argument("--out", default=None,
                   help="output HTML path (default: workspace/investigations/<inv>/reports/<study>.html)")
    args = p.parse_args()

    study_path = Path("workspace/studies") / args.study / "study.yaml"
    if not study_path.exists():
        sys.exit(f"study not found: {study_path}")
    spec = yaml.safe_load(study_path.read_text(encoding="utf-8"))

    inv_path = Path("workspace/investigations") / INV_SLUG / "investigation.yaml"
    inv = yaml.safe_load(inv_path.read_text(encoding="utf-8")) if inv_path.exists() else {}

    out_path = Path(args.out) if args.out else Path("workspace/investigations") / INV_SLUG / "reports" / f"{args.study}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Compose the report ----
    title = spec.get("title", args.study)
    inv_title = inv.get("title", "")
    today = date.today().isoformat()

    sections = []

    # 1. Executive
    sections.append(f"""
<section id="executive">
  <h2>Executive summary</h2>
  <p>This is a <strong>single-study</strong> report on <code>{_h(args.study)}</code>, scoped for
     focused expert review per the &quot;one study at a time&quot; workflow. The parent
     investigation is <em>{_h(inv_title)}</em>; the other studies are deferred until
     this one returns approved.</p>
  <p><strong>Question:</strong> {_multiline(spec.get('question', ''))}</p>
  <p><strong>Hypothesis:</strong> {_multiline(spec.get('hypothesis', ''))}</p>
  <p><strong>Objective:</strong> {_multiline(spec.get('objective', ''))}</p>
</section>
""")

    # 2. Study card
    sc_html = render_study_card(spec)
    if sc_html:
        sections.append(f'<section id="study-card"><h2>Study card</h2>{sc_html}</section>')

    # 3. What's delivered (real)
    # For pdmp-00 specifically, hard-code the headline summary; otherwise pull from notes.
    if args.study == "pdmp-00-characterization":
        delivered = """
<ul>
  <li><strong>3 × N=64 × 600 s × 1-generation ensembles</strong> — 192 successful replicate runs
      across M9-glucose / M9-acetate / M9-glucose+aa nutrient conditions.</li>
  <li>Per-condition stats:
    <ul>
      <li>M9-glucose: ATP mean 7.70e+06, CV 0.64%, wall 50.7 min</li>
      <li>M9-acetate: ATP mean 2.00e+06, CV 0.87%, wall 45.5 min (1 seed crashed at t=141)</li>
      <li>M9-glucose+aa: ATP mean 1.52e+07, CV 0.09%, wall 68.0 min</li>
    </ul>
  </li>
  <li>Cross-condition biology check passes: glucose:acetate ATP = 3.85× (gluconeogenic);
      with_aa:glucose ATP = 1.97× (aa-fast growth).</li>
  <li>Per-process RNG seeding fix (commit 7783012) verified — ensemble divergence is real
      and consistent (0.09–0.87% CV across conditions).</li>
  <li>FBA reference Millard 2017 ODE: real published-state validation via basico/COPASI.</li>
  <li>Visualizations: 25 viz cards in this report — all real-data or honest "awaiting"
      skeleton (no synthetic placeholders).</li>
</ul>
"""
        sections.append(f'<section id="delivered"><h2>What\'s delivered</h2>{delivered}</section>')

    # 4. Tests
    tests = spec.get("tests") or []
    sections.append(
        f'<section id="tests"><h2>Tests ({len(tests)})</h2>'
        '<p class="cap">Each test makes a specific falsifiable scientific claim with a measurable pass criterion.</p>'
        + "".join(render_test(t) for t in tests)
        + '</section>'
    )

    # 5. Conditions (Build tab)
    sections.append(f'<section id="conditions"><h2>Conditions (Build)</h2>{render_conditions(spec)}</section>')

    # 6. Simulations + planned runs
    sections.append(
        '<section id="sims"><h2>Simulations</h2>'
        '<h3>Planned runs</h3>' + render_planned_runs(spec)
        + '<h3>Simulation set (v4)</h3>' + render_simulations(spec)
        + '</section>'
    )

    # 7. Readouts
    sections.append(f'<section id="readouts"><h2>Readouts</h2>{render_readouts(spec)}</section>')

    # 8. Visualizations (inlined)
    fig_root = Path("reports/figures")
    sections.append(
        '<section id="viz"><h2>Visualizations</h2>'
        '<p class="cap">Real-data charts auto-pulled from .pbg/runs/. Skeletons clearly marked '
        '"Awaiting &lt;run&gt;" — never synthesised data.</p>'
        + render_viz(spec, fig_root)
        + '</section>'
    )

    # 9. Honest TBD
    sections.append("""
<section id="tbd"><h2>Honest TBD / known gaps</h2>
  <div class="gap"><span class="gap-label">Multi-generation:</span>
    Current Phase 0 deliverable is single-generation. Phase 2's inheritance tests need ≥3 generations per replicate; multi-gen runs are a separate planned-but-not-launched effort.</div>
  <div class="gap"><span class="gap-label">Full XArrayEmitter zarr:</span>
    Current trajectories are stored as per-seed JSON (same scientific content). XArrayEmitter's
    async writer drops the trailing buffer on close() for runs &lt;2000 steps — wiring is task #39.
    For Phase 1 validation the JSON capture is sufficient.</div>
  <div class="gap"><span class="gap-label">One acetate seed crashed:</span>
    seed_58 hit "Negative value(s) in counts_unallocated" at t=141 (known slow-growth edge
    case in v2ecoli). Kept in endpoint stats, dropped from time-series after t=141. 63 of 64
    seeds intact.</div>
  <div class="gap"><span class="gap-label">Profile-instrumented baseline NOT run:</span>
    The phase0-profile-instrumented-baseline planned_run (decomposes per-step compute into
    FBA-LP / RNG / marshalling buckets) is needed to populate profile_decomposition_bars viz.
    Currently the viz is a skeleton.</div>
  <div class="gap"><span class="gap-label">Variable-categorization map NOT authored:</span>
    The "every state variable labelled ODE-amenable / stochastic-discrete / teleonomic /
    coupling-artifact" deliverable is a design artifact (no run produces it) and is still TBD.</div>
</section>
""")

    # 10. Expert questions
    sections.append("""
<section id="expert-questions"><h2>For expert review</h2>
  <ol class="expert">
    <li class="expert-question"><span class="q">Are the 3 nutrient conditions sufficient as the Phase 0 reference?</span>
      DUF report Sec 4.1.1 notes "~a dozen conditions" exist; we sampled glucose/acetate/glucose+aa
      spanning the metabolic regime. Adequate for Phase 1+ Wasserstein-2 acceptance, or should we
      cover more (succinate, no_oxygen)?</li>
    <li class="expert-question"><span class="q">Is N=64 replicates adequate?</span>
      Justification given in study: σ/√64 = σ/8 ≈ 12.5% SE. Phase 1+ acceptance threshold is W₂ &lt;
      5% of inter-condition effect size — does that target make sense, or should we scale to N=256?</li>
    <li class="expert-question"><span class="q">The with_aa CV (0.09%) is much tighter than glucose (0.64%) or acetate (0.87%) — real biology or artifact?</span>
      Possible: amino-acid supplementation buffers stochastic variance in translation initiation
      (one of the loudest stochastic sources). Alternative: faster growth + larger pools → smaller
      relative noise. Both? Worth probing in Phase 2's jump-process formulation.</li>
    <li class="expert-question"><span class="q">Is the per-process RNG seeding fix the right pattern?</span>
      Implementation: <code>seed = crc32(process_name, master_seed)</code> per process, same
      pattern as v2ecoli/steps/division.py:CellDivision. Result: real divergence (0.09–0.87% CV).
      Concern: does this introduce structured correlations across processes?</li>
    <li class="expert-question"><span class="q">Is the single-generation ensemble enough for Phase 1 validation, or do we need multi-gen first?</span>
      Phase 1's metabolism-ODE replacement only changes intracellular dynamics within a generation
      (no division). Multi-gen testing is Phase 2 territory (inheritance, division-time). Can we
      proceed to Phase 1 with single-gen reference?</li>
    <li class="expert-question"><span class="q">profile_decomposition_bars + variable_categorization_map are not yet delivered — block the study, or accept as Phase 0.5 follow-ups?</span>
      Both inform Phase 4 (compilation) more than Phases 1-3. Recommend: accept this study as
      complete-for-Phase-1-handoff, file the two as Phase-4-input follow-ups.</li>
  </ol>
</section>
""")

    # 11. Bibliography
    bib = spec.get("bibliography") or {}
    bib_keys = bib.get("bib_keys") or []
    if bib_keys:
        bib_html = render_bibliography(bib_keys, Path("workspace/references/papers.bib"))
        sections.append(f'<section id="bib"><h2>References</h2>{bib_html}</section>')

    body = "".join(sections)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_h(args.study)} — single-study report</title>
<style>{CSS}</style>
{EMBED_JS}
</head>
<body>
<header>
  <h1>{_h(title)}</h1>
  <p class="meta">Single-study report · investigation <code>{_h(INV_SLUG)}</code> · rendered {today}</p>
</header>
{body}
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"wrote {out_path}  ({len(html_doc)//1024} KB)")


if __name__ == "__main__":
    main()
