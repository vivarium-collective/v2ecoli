"""
ecoli-sources bundle integration report
========================================

Documents the ecoli-sources data bundle as it is used in v2ecoli:
  1. Bundle inputs — full inventory, download links, override annotations
  2. How ParCa uses them — HTML flow diagram + prose
  3. ParCa output summary — sim_data introspection (requires a real run)
  4. Multi-ParCa variant comparison — rnaseq dataset swap (requires a real run)

Usage::

    # Full report (runs ParCa — intended for a remote machine):
    python reports/ecoli_sources_report.py

    # No-compute path (sections 1 & 2 only, 3 & 4 as placeholders):
    python reports/ecoli_sources_report.py --skip-runs

    # Custom options:
    python reports/ecoli_sources_report.py --skip-runs --out /tmp/report.html
    python reports/ecoli_sources_report.py --cpus 4 --date "2026-06-08"
    python reports/ecoli_sources_report.py --variants plus_aas=vecoli_m9_glucose_plus_aas.tsv
"""
from __future__ import annotations

import argparse
import base64
import html
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, system-ui, sans-serif;
       color: #1e293b; background: #f8fafc; }
header { background: #0f172a; color: #f1f5f9; padding: 24px 36px; }
header h1 { margin: 0 0 6px; font-size: 22px; }
header p { margin: 0; color: #94a3b8; font-size: 13px; }
main { max-width: 1100px; margin: 0 auto; padding: 28px 36px 70px; }
h2 { font-size: 17px; margin: 36px 0 8px; color: #0f172a;
     border-left: 4px solid #6366f1; padding-left: 10px; }
h3 { font-size: 14px; margin: 20px 0 6px; color: #334155; }
.note { font-size: 12px; color: #64748b; margin: 0 0 12px; }
.strategy { font-size: 12px; color: #0f766e; background: #f0fdfa;
            border: 1px solid #99f6e4; border-radius: 6px;
            padding: 8px 12px; margin: 0 0 14px; }
table { border-collapse: collapse; width: 100%; background: #fff;
        font-size: 13px; margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
        border-radius: 8px; overflow: hidden; }
th, td { padding: 7px 13px; text-align: left; border-bottom: 1px solid #eef2f7; }
thead th { background: #1e293b; color: #f1f5f9; font-weight: 600; }
tr:last-child td { border-bottom: none; }
td.key { font-family: monospace; font-size: 12px; color: #334155; word-break: break-all; }
td.file { font-family: monospace; font-size: 11px; color: #64748b; word-break: break-all; }
td.size { text-align: right; color: #64748b; }
.badge { display: inline-block; font-size: 10px; font-weight: 700;
         padding: 2px 7px; border-radius: 10px; white-space: nowrap; }
.badge-override { background: #fef9c3; color: #92400e; border: 1px solid #fde68a; }
.badge-bundle   { background: #e0f2fe; color: #0c4a6e; border: 1px solid #bae6fd; }
a.dl { font-size: 11px; text-decoration: none; color: #6366f1; }
a.dl:hover { text-decoration: underline; }
a.flink { font-size: 11px; text-decoration: none; color: #64748b; font-family: monospace; }
.placeholder { background: #f1f5f9; border: 2px dashed #cbd5e1;
               border-radius: 8px; padding: 24px 32px; color: #64748b;
               font-size: 13px; margin: 12px 0; }
.placeholder code { font-size: 12px; color: #475569; }
/* Flow diagram */
.flow-wrap { overflow-x: auto; padding: 16px 0; }
.flow { display: flex; align-items: center; gap: 0; flex-wrap: nowrap; }
.box { background: #fff; border: 2px solid #6366f1; border-radius: 8px;
       padding: 10px 14px; min-width: 130px; text-align: center; flex-shrink: 0; }
.box.src   { border-color: #0ea5e9; background: #f0f9ff; }
.box.proc  { border-color: #6366f1; background: #eef2ff; }
.box.out   { border-color: #10b981; background: #ecfdf5; }
.box.ovr   { border-color: #f59e0b; background: #fffbeb; }
.box-title { font-size: 12px; font-weight: 700; color: #0f172a; }
.box-sub   { font-size: 10px; color: #64748b; margin-top: 3px; }
.arrow     { font-size: 22px; color: #94a3b8; padding: 0 4px; flex-shrink: 0; }
.override-merge { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.override-merge .box { min-width: 120px; }
.plus { font-size: 14px; font-weight: 700; color: #f59e0b; }
/* collapsible */
details { margin: 10px 0; }
summary { cursor: pointer; font-size: 13px; color: #4f46e5; user-select: none; }
summary::-webkit-details-marker { color: #4f46e5; }
.detail-inner { padding: 10px 0 4px; }
/* parca summary table */
.metric-label { color: #475569; font-size: 12px; }
.metric-val   { font-weight: 600; font-family: monospace; }
.metric-na    { color: #94a3b8; font-style: italic; }
.err-col      { background: #fff7f7; }
.err-msg      { font-size: 11px; color: #b91c1c; font-family: monospace; word-break: break-all; }
footer { max-width: 1100px; margin: 0 auto; padding: 0 36px 40px;
         color: #64748b; font-size: 12px; }
"""

# ---------------------------------------------------------------------------
# Helper: base64 encode a file for a data: URI
# ---------------------------------------------------------------------------
def _file_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _download_link(path: Path, label: str, mime: str = "text/tab-separated-values") -> str:
    """Return an <a href="data:..."> download link for a file, or a file:// link on failure."""
    try:
        b64 = _file_b64(path)
        return (f'<a class="dl" href="data:{mime};base64,{b64}" '
                f'download="{html.escape(path.name)}">{html.escape(label)}</a>')
    except Exception:
        return f'<a class="flink" href="file://{path}">{html.escape(label)}</a>'


def _filelink(path: Path, label: str) -> str:
    return f'<a class="flink" href="file://{path}">{html.escape(label)}</a>'


# ---------------------------------------------------------------------------
# Section 1: Bundle inputs
# ---------------------------------------------------------------------------
def section_inputs() -> str:
    from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle

    b = SourceBundle()
    index = b._index  # {canonical_key: Path}

    # Classify each key
    overrides = {k for k, p in index.items() if "flat_overrides" in str(p)}

    total_bytes = sum(p.stat().st_size for p in index.values() if p.exists())
    total_mb = total_bytes / 1024 / 1024

    EMBED_THRESHOLD_MB = 25.0
    embed_all = total_mb <= EMBED_THRESHOLD_MB

    if embed_all:
        strategy = (
            f"Embedding strategy: <strong>all {len(index)} files inline</strong> "
            f"as <code>data:</code> URIs (total {total_mb:.2f} MB ≤ {EMBED_THRESHOLD_MB:.0f} MB threshold)."
        )
        print(f"[inputs] Embedding ALL files (total {total_mb:.2f} MB, threshold {EMBED_THRESHOLD_MB} MB)")
    else:
        strategy = (
            f"Embedding strategy: <strong>partial</strong> — "
            f"only the 3 v2ecoli overrides + rnaseq datasets are embedded as <code>data:</code> URIs; "
            f"remaining {len(index) - 3} files are <code>file://</code> links "
            f"(total {total_mb:.2f} MB exceeds {EMBED_THRESHOLD_MB:.0f} MB threshold)."
        )
        print(f"[inputs] Partial embed (total {total_mb:.2f} MB > {EMBED_THRESHOLD_MB} MB; "
              f"embedding overrides + rnaseq only)")

    # Sort: overrides first, then by category (key prefix)
    def sort_key(k):
        is_ov = k in overrides
        parts = k.split("__")
        cat = parts[0] if len(parts) > 1 else "root"
        return (0 if is_ov else 1, cat, k)

    sorted_keys = sorted(index.keys(), key=sort_key)

    rows = []
    for k in sorted_keys:
        p = index[k]
        is_ov = k in overrides
        badge_html = (
            '<span class="badge badge-override">v2ecoli override</span>'
            if is_ov else
            '<span class="badge badge-bundle">ecoli-sources</span>'
        )
        size_kb = p.stat().st_size / 1024 if p.exists() else 0

        # Decide whether to embed
        do_embed = embed_all or is_ov or "rnaseq" in k.lower()
        if p.exists():
            if do_embed:
                link = _download_link(p, "download")
            else:
                link = _filelink(p, "file://")
        else:
            link = "<span class='metric-na'>missing</span>"

        rows.append(
            f"<tr>"
            f"<td class='key'>{html.escape(k)}</td>"
            f"<td>{badge_html}</td>"
            f"<td class='file'>{html.escape(p.name)}</td>"
            f"<td class='size'>{size_kb:.1f}</td>"
            f"<td>{link}</td>"
            f"</tr>"
        )

    n_ov = len(overrides)
    n_bundle = len(index) - n_ov

    header_row = (
        "<tr><th>Canonical key</th><th>Source</th>"
        "<th>Filename</th><th style='text-align:right'>Size (KB)</th>"
        "<th>Download</th></tr>"
    )

    summary = (
        f"<p class='note'>{len(index)} keys total &mdash; "
        f"<strong>{n_ov} v2ecoli overrides</strong> (diverged flat files), "
        f"<strong>{n_bundle} from ecoli-sources</strong>. "
        f"Total raw size: {total_mb:.2f} MB.</p>"
    )

    return f"""
<h2>1 &nbsp; Bundle inputs</h2>
{summary}
<div class="strategy">{strategy}</div>
<table>
<thead>{header_row}</thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


# ---------------------------------------------------------------------------
# Section 2: How ParCa uses them
# ---------------------------------------------------------------------------
def section_flow() -> str:
    flow_html = """
<div class="flow-wrap">
<div class="flow">

  <div class="override-merge">
    <div class="box src">
      <div class="box-title">ecoli-sources bundle</div>
      <div class="box-sub">reference_bundle.tsv<br>135 canonical keys</div>
    </div>
    <div class="plus">+</div>
    <div class="box ovr">
      <div class="box-title">v2ecoli overrides</div>
      <div class="box-sub">parca_overrides.tsv<br>3 diverged flat files</div>
    </div>
  </div>

  <div class="arrow">&#8594;</div>

  <div class="box proc">
    <div class="box-title">SourceBundle resolver</div>
    <div class="box-sub">canonical_key &rarr; Path<br>overrides win on merge</div>
  </div>

  <div class="arrow">&#8594;</div>

  <div class="box proc">
    <div class="box-title">KnowledgeBaseEcoli</div>
    <div class="box-sub">reads TSV flat files<br>into raw_data structs</div>
  </div>

  <div class="arrow">&#8594;</div>

  <div class="box proc">
    <div class="box-title">build_parca_composite</div>
    <div class="box-sub">9 bigraph Steps wired<br>in a DAG composite</div>
  </div>

  <div class="arrow">&#8594;</div>

  <div class="box out">
    <div class="box-title">sim_data</div>
    <div class="box-sub">SimulationDataEcoli<br>parca_state.pkl</div>
  </div>

</div>
</div>
"""

    step_rows = "".join(
        f"<tr><td class='key'>Step {n} &mdash; {name}</td><td>{desc}</td></tr>"
        for n, name, desc in [
            (1, "InitializeStep",
             "Loads <code>KnowledgeBaseEcoli</code> raw flat-file tables; "
             "instantiates an empty <code>SimulationDataEcoli</code>; "
             "populates compartments, MW keys, base codes, and helper objects."),
            (2, "InputAdjustmentsStep",
             "Applies optional gene-knockouts and expression adjustments from "
             "<code>adjustments/</code> keys."),
            (3, "BasalSpecsStep",
             "Fits basal transcription and translation parameters (RNA/protein "
             "fractions, elongation rates). Caches Km optimisation results to disk."),
            (4, "TfConditionSpecsStep",
             "Fits per-TF-condition expression specs in parallel "
             "(<code>--cpus</code> controls fan-out)."),
            (5, "FitConditionStep",
             "Iterates promoter-binding/ribosome-capacity fitting per nutrient "
             "condition; the most CPU-intensive step (~60 min total)."),
            (6, "PromoterBindingStep",
             "Sets promoter-bound RNAP fractions using the fitted TF activities."),
            (7, "AdjustPromotersStep",
             "Applies ppGpp-based promoter-expression adjustments."),
            (8, "SetConditionsStep",
             "Writes per-condition biomass/mass targets into <code>cell_specs</code>; "
             "emits <code>expected_dry_mass_increase_dict</code>."),
            (9, "FinalAdjustmentsStep",
             "Normalises remaining fitted parameters; finalises "
             "<code>sim_data_root</code>."),
        ]
    )

    return f"""
<h2>2 &nbsp; How ParCa runs them</h2>

<h3>Pipeline overview</h3>
<p class="note">
  <code>v2ecoli-parca</code> builds a <strong>9-step process-bigraph composite</strong>
  from the bundle&#8217;s flat TSV files. Each step is a <code>_type: step</code> node
  in the bigraph store; steps fire sequentially via <code>run_steps_on_init</code>.
  The diagram below shows the data-flow from bundle manifest to pickled
  <code>sim_data</code>.
</p>

{flow_html}

<h3>Override mechanism</h3>
<p class="note">
  v2ecoli keeps three biology-diverged files locally in
  <code>v2ecoli/processes/parca/reconstruction/ecoli/flat_overrides/</code>
  and lists them in <code>parca_overrides.tsv</code>. <code>SourceBundle.__init__</code>
  reads the ecoli-sources <code>reference_bundle.tsv</code> first, then applies the
  override manifest on top — later entries overwrite earlier ones for the same
  <code>canonical_key</code>. The three overridden keys are:
  <code>equilibrium_reactions</code>, <code>equilibrium_reaction_rates</code>,
  and <code>metabolic_reactions_added</code> (DnaA-ATP hydrolysis + metabolic
  additions from PRs #123 and v2parca merge #16).
</p>

<h3>Bundle swap point for variants</h3>
<p class="note">
  Passing <code>--bundle-manifest-path</code> to <code>v2ecoli-parca</code> replaces
  the default ecoli-sources <code>BUNDLE_PATH</code> as the base manifest. Rows in
  the replacement manifest that match a <code>canonical_key</code> from the default
  bundle override its <code>source_path</code>. v2ecoli&#8217;s override manifest is
  then layered on top as usual. This is the mechanism used in section 4 to swap
  the <code>rnaseq_experimental_tpms</code> dataset per variant run.
</p>

<h3>9 pipeline steps</h3>
<table>
<thead><tr><th>Step</th><th>What it does</th></tr></thead>
<tbody>{step_rows}</tbody>
</table>
"""


# ---------------------------------------------------------------------------
# Section 3: ParCa output (default bundle)
# ---------------------------------------------------------------------------
def _safe(obj, *attrs, fmt=None):
    """Walk nested attributes defensively; return formatted value or '—'."""
    try:
        v = obj
        for a in attrs:
            v = getattr(v, a)
        if fmt:
            return fmt.format(v)
        # Try len() for array-like things
        try:
            return str(len(v))
        except TypeError:
            return str(v)
    except Exception:
        return "<span class='metric-na'>&mdash;</span>"


def _extract_metrics(state: dict) -> dict:
    """Defensively extract key metrics from a parca composite state dict."""
    sd = state.get("sim_data_root")
    if sd is None:
        return {"error": "sim_data_root key missing from state dict"}

    metrics = {}

    # Genes / cistrons (transcription)
    metrics["n_cistrons"] = _safe(sd, "process", "transcription", "cistron_data")
    metrics["n_rnas"]     = _safe(sd, "process", "transcription", "rna_data")

    # Proteins (translation monomers)
    metrics["n_monomers"] = _safe(sd, "process", "translation", "n_monomers")

    # Metabolic reactions
    metrics["n_rxns"]     = _safe(sd, "process", "metabolism", "base_reaction_ids")

    # Bulk molecules
    metrics["n_bulk"]     = _safe(sd, "internal_state", "bulk_molecules", "bulk_data")

    # Mass parameters
    try:
        m = sd.mass.avg_cell_dry_mass
        metrics["avg_dry_mass_fg"] = f"{float(m.asNumber()):.2f}" if hasattr(m, "asNumber") else str(m)
    except Exception:
        metrics["avg_dry_mass_fg"] = "<span class='metric-na'>&mdash;</span>"

    try:
        dt = sd.doubling_time
        metrics["doubling_time_min"] = f"{float(dt.asNumber()):.1f}" if hasattr(dt, "asNumber") else str(dt)
    except Exception:
        metrics["doubling_time_min"] = "<span class='metric-na'>&mdash;</span>"

    # Conditions
    try:
        metrics["n_conditions"] = str(len(sd.condition_to_doubling_time))
    except Exception:
        metrics["n_conditions"] = "<span class='metric-na'>&mdash;</span>"

    # Translation efficiencies
    try:
        import numpy as np
        te = sd.process.translation.translation_efficiencies_by_monomer
        metrics["mean_trl_efficiency"] = f"{float(np.mean(te)):.4f}"
    except Exception:
        metrics["mean_trl_efficiency"] = "<span class='metric-na'>&mdash;</span>"

    # Genome length
    try:
        metrics["genome_length_bp"] = _safe(sd, "process", "transcription", "_genome_length",
                                             fmt="{:,}")
    except Exception:
        metrics["genome_length_bp"] = "<span class='metric-na'>&mdash;</span>"

    return metrics


_METRIC_LABELS = [
    ("n_cistrons",           "Cistrons (gene models)"),
    ("n_rnas",               "RNA species"),
    ("n_monomers",           "Protein monomers"),
    ("n_rxns",               "Metabolic reactions (base)"),
    ("n_bulk",               "Bulk molecule species"),
    ("avg_dry_mass_fg",      "Avg cell dry mass (fg)"),
    ("doubling_time_min",    "Doubling time (min)"),
    ("n_conditions",         "Nutrient conditions"),
    ("mean_trl_efficiency",  "Mean translation efficiency"),
    ("genome_length_bp",     "Genome length (bp)"),
]


def section_parca_output(outdir: Path, cpus: int, skip: bool) -> str:
    if skip:
        return """
<h2>3 &nbsp; ParCa output (default bundle)</h2>
<div class="placeholder">
  <strong>Not run</strong> &mdash; pass without <code>--skip-runs</code> to execute ParCa
  and populate this section.<br><br>
  <code>v2ecoli-parca --mode fast -o out/sim_data --cpus N</code>
</div>
"""

    pkl_path = outdir / "parca_state.pkl"
    # Run ParCa if pkl doesn't exist
    if not pkl_path.exists():
        print(f"[parca/default] Running ParCa → {outdir} ...")
        argv = [
            str(_REPO_ROOT / ".venv" / "bin" / "v2ecoli-parca"),
            "--mode", "fast",
            "-o", str(outdir),
            "--cpus", str(cpus),
        ]
        result = subprocess.run(argv, cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            return f"""
<h2>3 &nbsp; ParCa output (default bundle)</h2>
<div class="placeholder" style="border-color:#fca5a5;background:#fff7f7">
  <strong>ParCa run FAILED</strong> (exit code {result.returncode}).
  Check stderr for details.
</div>
"""
    if not pkl_path.exists():
        return f"""
<h2>3 &nbsp; ParCa output (default bundle)</h2>
<div class="placeholder" style="border-color:#fca5a5;background:#fff7f7">
  <strong>parca_state.pkl not found</strong> in {html.escape(str(outdir))}.
</div>
"""

    print(f"[parca/default] Loading {pkl_path} ...")
    try:
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
    except Exception as e:
        return f"""
<h2>3 &nbsp; ParCa output (default bundle)</h2>
<div class="placeholder" style="border-color:#fca5a5;background:#fff7f7">
  <strong>Failed to load pkl:</strong> {html.escape(str(e))}
</div>
"""

    metrics = _extract_metrics(state)

    if "error" in metrics:
        err_html = f'<tr><td colspan="2" class="err-msg">{html.escape(metrics["error"])}</td></tr>'
        metric_rows = err_html
    else:
        metric_rows = "".join(
            f"<tr>"
            f"<td class='metric-label'>{html.escape(label)}</td>"
            f"<td class='metric-val'>{metrics.get(key, '<span class=\"metric-na\">&mdash;</span>')}</td>"
            f"</tr>"
            for key, label in _METRIC_LABELS
        )

    pkl_size_mb = pkl_path.stat().st_size / 1024 / 1024
    if pkl_size_mb < 25:
        pkl_download = _download_link(pkl_path, f"download pkl ({pkl_size_mb:.1f} MB)",
                                       mime="application/octet-stream")
    else:
        pkl_download = (f"pkl too large to embed ({pkl_size_mb:.1f} MB); "
                        f"path: <code>{html.escape(str(pkl_path))}</code>")

    return f"""
<h2>3 &nbsp; ParCa output (default bundle)</h2>
<p class="note">
  SimulationDataEcoli loaded from <code>{html.escape(str(pkl_path))}</code>
  ({pkl_size_mb:.1f} MB). Attributes accessed defensively; &mdash; = not reachable.
  {pkl_download}
</p>
<p class="note"><strong>Attribute paths attempted:</strong>
  <code>sd.process.transcription.cistron_data</code>,
  <code>sd.process.transcription.rna_data</code>,
  <code>sd.process.translation.n_monomers</code>,
  <code>sd.process.translation.translation_efficiencies_by_monomer</code>,
  <code>sd.process.metabolism.base_reaction_ids</code>,
  <code>sd.internal_state.bulk_molecules.bulk_data</code>,
  <code>sd.mass.avg_cell_dry_mass</code>,
  <code>sd.doubling_time</code>,
  <code>sd.condition_to_doubling_time</code>,
  <code>sd.process.transcription._genome_length</code>.
</p>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{metric_rows}</tbody>
</table>
"""


# ---------------------------------------------------------------------------
# Section 4: Multi-ParCa variants
# ---------------------------------------------------------------------------
def _build_variant_manifest(base_key: str, dataset_path: Path, data_dir: Path) -> Path:
    """Write a variant bundle manifest that overrides rnaseq_experimental_tpms.

    Writes the temp file INSIDE data_dir so other rows' relative paths
    still resolve correctly.
    """
    from ecoli_sources import BUNDLE_PATH
    import pandas as pd

    df = pd.read_csv(str(BUNDLE_PATH), sep="\t", comment="#")
    mask = df["canonical_key"] == "rnaseq_experimental_tpms"
    if not mask.any():
        raise ValueError("rnaseq_experimental_tpms not found in base manifest")
    df.loc[mask, "source_path"] = str(dataset_path.resolve())

    # Write temp manifest inside DATA_DIR so relative paths in other rows resolve
    fd, tmp_path = tempfile.mkstemp(
        suffix=f"_variant_{base_key}.tsv", dir=str(data_dir), prefix="_manifest_"
    )
    os.close(fd)
    df.to_csv(tmp_path, sep="\t", index=False)
    return Path(tmp_path)


def _run_variant(name: str, dataset_path: Path, data_dir: Path,
                 outdir: Path, cpus: int) -> dict:
    """Run ParCa for one rnaseq variant. Returns dict with 'metrics' or 'error'."""
    manifest_path = None
    try:
        manifest_path = _build_variant_manifest(name, dataset_path, data_dir)
        pkl_path = outdir / "parca_state.pkl"
        argv = [
            str(_REPO_ROOT / ".venv" / "bin" / "v2ecoli-parca"),
            "--mode", "fast",
            "-o", str(outdir),
            "--cpus", str(cpus),
            "--bundle-manifest-path", str(manifest_path),
        ]
        print(f"[parca/variant={name}] Running ParCa → {outdir} ...")
        result = subprocess.run(argv, cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            return {"error": f"ParCa exited {result.returncode}"}
        if not pkl_path.exists():
            return {"error": "parca_state.pkl not produced"}
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        return {"metrics": _extract_metrics(state)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if manifest_path and manifest_path.exists():
            try:
                manifest_path.unlink()
            except Exception:
                pass


def section_multi_parca(
    default_outdir: Path, cpus: int, skip: bool,
    variants: dict[str, str],
) -> str:
    if skip:
        variant_list = ", ".join(f"{k}={v}" for k, v in variants.items())
        return f"""
<h2>4 &nbsp; Multi-ParCa variant comparison</h2>
<div class="placeholder">
  <strong>Not run</strong> &mdash; pass without <code>--skip-runs</code> to execute.<br><br>
  Default variants: <code>{html.escape(variant_list)}</code><br>
  Each run swaps <code>rnaseq_experimental_tpms</code> in a temp copy of the bundle
  manifest written to <code>DATA_DIR</code>, runs <code>v2ecoli-parca --mode fast</code>,
  then deletes the temp manifest. A failed variant run is shown as an isolated
  error column rather than aborting the report.
</div>
"""

    from ecoli_sources import DATA_DIR

    # --- resolve variant datasets ---
    rnaseq_dir = DATA_DIR / "rnaseq_experimental"
    resolved_variants: dict[str, Path] = {}
    for vname, vfile in variants.items():
        # Accept either bare filename or absolute path
        p = Path(vfile)
        if not p.is_absolute():
            p = rnaseq_dir / vfile
        resolved_variants[vname] = p

    # --- run default if needed ---
    default_pkl = default_outdir / "parca_state.pkl"
    cols: list[dict] = []

    if default_pkl.exists():
        print("[parca/multi] Loading cached default ...")
        try:
            with open(default_pkl, "rb") as f:
                state = pickle.load(f)
            cols.append({"name": "default (basal)", "result": {"metrics": _extract_metrics(state)}})
        except Exception as e:
            cols.append({"name": "default (basal)", "result": {"error": str(e)}})
    else:
        print("[parca/multi] Running default ParCa for comparison baseline ...")
        argv = [
            str(_REPO_ROOT / ".venv" / "bin" / "v2ecoli-parca"),
            "--mode", "fast", "-o", str(default_outdir), "--cpus", str(cpus),
        ]
        r = subprocess.run(argv, cwd=str(_REPO_ROOT))
        if r.returncode != 0 or not default_pkl.exists():
            cols.append({"name": "default (basal)", "result": {"error": f"ParCa exited {r.returncode}"}})
        else:
            with open(default_pkl, "rb") as f:
                state = pickle.load(f)
            cols.append({"name": "default (basal)", "result": {"metrics": _extract_metrics(state)}})

    # --- run each variant ---
    for vname, vpath in resolved_variants.items():
        if not vpath.exists():
            cols.append({"name": vname, "result": {"error": f"Dataset not found: {vpath}"}})
            continue
        variant_outdir = default_outdir.parent / f"parca_variant_{vname}"
        variant_outdir.mkdir(parents=True, exist_ok=True)
        result = _run_variant(vname, vpath, DATA_DIR, variant_outdir, cpus)
        cols.append({"name": vname, "result": result})

    # --- build HTML ---
    header_cols = "".join(
        f"<th>{html.escape(c['name'])}"
        f"{'<br><span class=\"err-msg\">FAILED</span>' if 'error' in c['result'] else ''}"
        f"</th>"
        for c in cols
    )
    header = f"<tr><th>Metric</th>{header_cols}</tr>"

    body_rows = []
    for key, label in _METRIC_LABELS:
        cells = [f"<td class='metric-label'>{html.escape(label)}</td>"]
        for c in cols:
            r = c["result"]
            if "error" in r:
                cells.append(
                    f"<td class='err-col'>"
                    f"<span class='err-msg'>{html.escape(r['error'][:120])}</span></td>"
                )
            else:
                v = r.get("metrics", {}).get(key, "<span class='metric-na'>&mdash;</span>")
                cells.append(f"<td class='metric-val'>{v}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""
<h2>4 &nbsp; Multi-ParCa variant comparison</h2>
<p class="note">
  Each column is an independent ParCa run with a different
  <code>rnaseq_experimental_tpms</code> dataset. Failed runs show as isolated
  error columns; they do not abort the report.
</p>
<table>
<thead>{header}</thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
"""


# ---------------------------------------------------------------------------
# HTML skeleton
# ---------------------------------------------------------------------------
def build_html(
    sections: list[str],
    date: str,
    skip: bool,
) -> str:
    skip_note = " (sections 3 &amp; 4 skipped — <code>--skip-runs</code>)" if skip else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v2ecoli — ecoli-sources bundle report</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <h1>ecoli-sources bundle integration report</h1>
  <p>v2ecoli &nbsp;&bull;&nbsp; branch: <code>feat/ecoli-sources-bundle</code>
     &nbsp;&bull;&nbsp; generated: {html.escape(date or 'unknown')}{skip_note}</p>
</header>
<main>
{''.join(sections)}
</main>
<footer>
  Generated by <code>reports/ecoli_sources_report.py</code>. &nbsp;
  Bundle resolver: <code>v2ecoli.processes.parca.reconstruction.ecoli.sources.SourceBundle</code>. &nbsp;
  ecoli-sources: installed package; v2ecoli overrides: <code>parca_overrides.tsv</code>.
</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate the ecoli-sources bundle integration HTML report."
    )
    ap.add_argument(
        "--skip-runs", action="store_true",
        help="Render sections 1 & 2 only; 3 & 4 become 'not run' placeholders.",
    )
    ap.add_argument(
        "--out", default=str(_DOCS_DIR / "ecoli_sources_report.html"),
        help="Output HTML path (default: docs/ecoli_sources_report.html).",
    )
    ap.add_argument(
        "--cpus", type=int, default=2,
        help="CPU count passed to v2ecoli-parca (default: 2).",
    )
    ap.add_argument(
        "--date", default="",
        help="Date string embedded in report header (default: empty/unknown).",
    )
    ap.add_argument(
        "--variants",
        default="plus_aas=vecoli_m9_glucose_plus_aas.tsv",
        help=(
            "Comma-separated name=dataset.tsv pairs for multi-ParCa variants "
            "(default: plus_aas=vecoli_m9_glucose_plus_aas.tsv). "
            "Filenames are relative to DATA_DIR/rnaseq_experimental/ unless absolute."
        ),
    )
    args = ap.parse_args()

    # Parse variants
    variants: dict[str, str] = {}
    if args.variants.strip():
        for item in args.variants.split(","):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                variants[k.strip()] = v.strip()
            else:
                print(f"WARNING: ignoring malformed --variants item: {item!r}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    default_outdir = _REPO_ROOT / "out" / "parca_report_default"
    default_outdir.mkdir(parents=True, exist_ok=True)

    # --- build sections ---
    print("Building section 1 (inputs) ...")
    s1 = section_inputs()

    print("Building section 2 (flow diagram) ...")
    s2 = section_flow()

    print("Building section 3 (parca output) ...")
    s3 = section_parca_output(default_outdir, args.cpus, args.skip_runs)

    print("Building section 4 (multi-parca variants) ...")
    s4 = section_multi_parca(default_outdir, args.cpus, args.skip_runs, variants)

    print("Assembling HTML ...")
    doc = build_html([s1, s2, s3, s4], args.date, args.skip_runs)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)

    size_kb = out_path.stat().st_size / 1024
    print(f"\nReport written to: {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
