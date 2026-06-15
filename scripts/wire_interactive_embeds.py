"""Append a curated ``embed_visualizations:`` block to each mbp study.yaml,
pointing at the interactive Plotly figures under reports/figures/<study>/ so the
report surfaces them with a takeaway title + caption (overriding the generic
auto-discovered entry, which dedups by url).

Textual append (NOT a YAML round-trip): the studies carry hand-formatted
block scalars and research-log comments that a ruamel re-dump reflows; appending
the new top-level key as text leaves every existing byte untouched. Idempotent —
re-running removes a prior appended block (between the BEGIN/END markers) first.

Run from the worktree root:  ./.venv/bin/python scripts/wire_interactive_embeds.py
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ROOT / "workspace" / "studies"

BEGIN = "# >>> interactive-plotly embeds (scripts/wire_interactive_embeds.py) >>>"
END = "# <<< interactive-plotly embeds <<<"

# study dir -> list of (figure-file-stem, curated name, takeaway description)
ENTRIES = {
    "mbp-01-time-varying-environment": [
        ("env_coupling_interactive",
         "Environment coupling — uptake & growth track the medium (interactive)",
         "baseline WCM, 40 steps: glucose & O2 uptake fluxes and the FBA growth "
         "objective respond to the external medium. Hover for per-step values; "
         "toggle traces in the legend. Establishes that the cell is "
         "environment-responsive — the prerequisite for reactor coupling."),
    ],
    "mbp-03-bird-reactor-coupling": [
        ("reactor_coupling_interactive",
         "Reactor-cell coupling — dissolved O2 falls under cell load (interactive)",
         "reactor_bird_coupled: the BiRD reactor's dissolved O2 and biomass over "
         "time as the cell draws O2/glucose and secretes byproducts through the "
         "ReactorCellCoupler. The keystone bidirectional coupling, now interactive "
         "(hover/zoom/toggle)."),
    ],
    "mbp-04-multigeneration-runs": [
        ("biomass_accumulation_interactive",
         "Multi-generation accumulation — WCM x53 over 6 generations vs Millard flat (interactive)",
         "The headline result. WCM-FBA divides 5x to 6 generations, representative "
         "biomass x53 (the representative_doubling path toward batch scale); "
         "Millard-ODE stays at gen 1 (x1.04) — the documented inherent "
         "reduced-model slow-growth. Toggle arms in the legend; hover for per-step mass."),
    ],
    "mbp-05-palsson-benchmark": [
        ("beulig_3arm_biomass_overlay_interactive",
         "3-arm Beulig overlay — WCM-FBA / Millard / iML1515 vs reference (interactive)",
         "Sim arms vs the Beulig 2025 batch reference for biomass/growth. Each arm "
         "is a toggleable trace; the published reference is shown as markers. Makes "
         "the ~3-order batch-scale gap directly comparable."),
        ("beulig_byproduct_reference_interactive",
         "Byproduct reference — secretion vs Beulig peaks (interactive)",
         "Byproduct trajectories (acetate/lactate/formate/...) against the Beulig "
         "reference peaks. Hover for values; toggle species in the legend."),
        ("beulig_verdict_heatmap_interactive",
         "Report-card verdict heatmap — graded axes (interactive)",
         "groups x axes colored within_tol / drift / mismatch from "
         "report_card_verdict.json; hover for the per-axis sim/ref/tolerance "
         "detail. The graded summary of the 3-arm comparison."),
    ],
    "mbp-06-gap-analysis": [
        ("gap_inventory_interactive",
         "Gap inventory — 15 gaps by axis (interactive)",
         "The 15-gap synthesis grouped by axis; hover for each gap's title and "
         "status. Makes the gap landscape scannable at a glance."),
    ],
    "mbp-07-millard-kinetic-metabolism": [
        ("metabolism_swap_growth_interactive",
         "Metabolism-swap growth — cell_mass + central fluxes under Millard (interactive)",
         "baseline_millard, 60 steps: cell_mass and live Millard central-carbon "
         "fluxes with the kinetic ODE replacing FBA. Hover/zoom; toggle fluxes."),
        ("env_responsiveness_interactive",
         "Millard env-responsiveness — PTS uptake vs external glucose (interactive)",
         "Glucose sweep: the Millard PTS uptake flux responds to external glucose "
         "(0.05 to 50 mM), confirming the kinetic arm is environment-coupled."),
    ],
}


def _dq(s: str) -> str:
    """Double-quoted YAML scalar (our strings have no backslashes/dquotes)."""
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _strip_prior_block(text: str) -> str:
    if BEGIN not in text:
        return text
    pre, _, rest = text.partition(BEGIN)
    _, _, post = rest.partition(END)
    return pre.rstrip("\n") + "\n" + post.lstrip("\n")


changed = 0
for study, figs in ENTRIES.items():
    sy = STUDIES / study / "study.yaml"
    if not sy.is_file():
        print(f"SKIP {study}: no study.yaml")
        continue
    present = [(stem, name, desc) for stem, name, desc in figs
              if (ROOT / "reports" / "figures" / study / f"{stem}.html").is_file()]
    missing = [stem for stem, _, _ in figs if (stem, None, None) not in
              [(p[0], None, None) for p in present]]
    if missing:
        print(f"WARN {study}: missing figures {missing} — skipped")
    if not present:
        continue

    lines = [BEGIN, "embed_visualizations:"]
    for stem, name, desc in present:
        lines.append(f"- name: {_dq(name)}")
        lines.append(f"  url: {_dq(f'/reports/figures/{study}/{stem}.html')}")
        lines.append(f"  description: {_dq(desc)}")
    lines.append(END)
    block = "\n".join(lines)

    text = _strip_prior_block(sy.read_text())
    if not text.endswith("\n"):
        text += "\n"
    sy.write_text(text + block + "\n")
    print(f"OK   {study}: {len(present)} embed_visualizations entries appended")
    changed += 1

print(f"\ndone: {changed} study.yaml files updated (textual append, zero reflow)")
