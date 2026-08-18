"""scripts/code_audit_report.py — a deterministic, config-driven code-transfer
report: what was added to the candidate from the private reference repository,
grouped into plain-English "additions" and backed by the EXACT transferred
source (embedded, not linked).

Fully generic: this script carries no biology-specific content. Everything
shown — the addition titles, descriptions, which processes they cover, and
which lines of a shared file belong to which addition — is declared in a
``code_audit`` block inside the private reference repo's OWN config file
(the same file an investigation's config already points each study at), e.g.:

    "code_audit": [
      {
        "title": "<plain-English name for this addition>",
        "description": "...",
        "entries": [
          {"process": "<process-name>", "mode": "full"},
          {"process": "<process-name>", "mode": "excerpt",
           "label": "<what this excerpt covers>",
           "match": ["SOME_CONSTANT\\\\s*=", "def some_function\\\\("], "context": 2}
        ]
      }
    ]

An entry with ``"mode": "full"`` embeds the whole transferred file. An entry
with ``"mode": "excerpt"`` greps the file for any line matching one of
``match`` (regexes), expands each hit by ``context`` lines, and merges
overlapping/adjacent ranges — deterministic and reproducible from source, not
hand-picked line numbers that drift as the file changes. ``process`` is
resolved to a source file the same way the full comparison report does
(``_transferred_source`` — a heuristic module-name lookup under the fork's
``ecoli/`` tree, reading the file directly with no import, so this never
pulls in the fork's runtime dependencies).

A study whose config declares no ``code_audit`` (e.g. a plain baseline) is
skipped — there is nothing to audit.

    .venv/bin/python scripts/code_audit_report.py \\
        --investigation <investigation-name> --output-file out/code_audit_report.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

from scripts.comparison_report_card import _git_provenance, _transferred_source


def _match_ranges(lines: list[str], patterns: list[str], context: int) -> list[list[int]]:
    compiled = [re.compile(p) for p in patterns]
    hit = {i for i, line in enumerate(lines) if any(c.search(line) for c in compiled)}
    if not hit:
        return []
    ranges: list[list[int]] = []
    for i in sorted(hit):
        lo, hi = max(0, i - context), min(len(lines) - 1, i + context)
        if ranges and lo <= ranges[-1][1] + 1:
            ranges[-1][1] = max(ranges[-1][1], hi)
        else:
            ranges.append([lo, hi])
    return ranges


def _resolve_entry(fork: str, entry: dict) -> dict:
    """Resolve one ``code_audit`` entry to its embeddable blocks.

    Returns ``{process, rel_path, label, blocks: [(start_line, end_line, code)]}``.
    ``start_line``/``end_line`` are None (and the block a bracketed notice) when
    the process or its matches can't be resolved — fails loud, in place, rather
    than silently dropping content from the report.
    """
    proc = entry["process"]
    rel, code = _transferred_source(fork, proc)
    label = entry.get("label")
    if code is None:
        return {"process": proc, "rel_path": None, "label": label,
                "blocks": [(None, None, f"[code not found for process {proc!r} "
                                        f"under fork {fork!r}]")]}
    lines = code.split("\n")
    mode = entry.get("mode", "full")
    if mode == "full":
        return {"process": proc, "rel_path": rel, "label": label,
                "blocks": [(1, len(lines), code)]}
    if mode != "excerpt":
        raise ValueError(f"code_audit entry for {proc!r}: unknown mode {mode!r} "
                         f"(expected 'full' or 'excerpt')")
    patterns = entry.get("match") or []
    context = int(entry.get("context", 3))
    ranges = _match_ranges(lines, patterns, context)
    if not ranges:
        return {"process": proc, "rel_path": rel, "label": label,
                "blocks": [(None, None, f"[no lines in {rel} matched "
                                        f"{patterns!r} — check the pattern or "
                                        f"whether the source moved]")]}
    blocks = [(lo + 1, hi + 1, "\n".join(lines[lo:hi + 1])) for lo, hi in ranges]
    return {"process": proc, "rel_path": rel, "label": label, "blocks": blocks}


def _load_code_audit(fork: str, config_ref: str) -> list[dict]:
    """The ``code_audit`` list from a study's private config, or [] when the
    config is a bare condition name (no private config file) or declares none."""
    if not config_ref or not config_ref.endswith(".json"):
        return []
    p = Path(fork) / config_ref
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.get("code_audit") or [])


# ---------------------------------------------------------------------------
# Rendering — a single self-contained accordion page. No biology-specific
# strings live here; every piece of copy is threaded through from the config.
# ---------------------------------------------------------------------------

_STYLE = """
:root {
  --bg: #faf9f7; --surface: #ffffff; --border: #e2e0dc; --text: #201f1c;
  --text-muted: #6b6862; --accent: #3d3a8f; --accent-soft: #eeecfa;
  --code-bg: #f4f2ee; --code-text: #2c2a26;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --sans: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16151a; --surface: #1e1d24; --border: #33313c; --text: #ece9f2;
    --text-muted: #9994a6; --accent: #9b95e8; --accent-soft: #29273a;
    --code-bg: #131218; --code-text: #d9d6e3;
  }
}
:root[data-theme="dark"] {
  --bg: #16151a; --surface: #1e1d24; --border: #33313c; --text: #ece9f2;
  --text-muted: #9994a6; --accent: #9b95e8; --accent-soft: #29273a;
  --code-bg: #131218; --code-text: #d9d6e3;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: var(--sans);
  margin: 0; padding: 48px 20px 80px; line-height: 1.5; }
.page { max-width: 840px; margin: 0 auto; }
.eyebrow { font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--text-muted); font-weight: 600; margin: 0 0 10px; }
h1 { font-size: 26px; font-weight: 700; margin: 0 0 6px; letter-spacing: -.01em;
  text-wrap: balance; }
.subtitle { color: var(--text-muted); font-size: 14.5px; margin: 0 0 20px; }
.provenance { font-size: 12.5px; color: var(--text-muted); margin: 0 0 28px; }
.provenance code { font-family: var(--mono); }
.banner { background: var(--accent-soft); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px 24px; font-size: 18px; font-weight: 700;
  color: var(--text); margin: 0 0 20px; }
.addition { border: 1px solid var(--border); border-radius: 10px;
  margin-bottom: 12px; overflow: hidden; background: var(--surface); }
.addhead { width: 100%; display: flex; align-items: center; gap: 14px;
  background: none; border: none; padding: 16px 20px; font-size: 15.5px;
  font-weight: 600; color: var(--text); cursor: pointer; text-align: left;
  font-family: inherit; }
.addhead:hover { background: var(--accent-soft); }
.addnum { display: inline-flex; align-items: center; justify-content: center;
  width: 24px; height: 24px; border-radius: 6px; background: var(--accent);
  color: #fff; font-size: 12.5px; font-weight: 700; flex-shrink: 0; }
.addtitle { flex: 1; }
.chev { font-size: 20px; color: var(--text-muted); transition: transform .15s ease;
  flex-shrink: 0; }
.chev.open { transform: rotate(90deg); }
.addbody { padding: 4px 20px 20px 58px; }
.adddesc { font-size: 14px; color: var(--text-muted); margin: 0 0 16px;
  line-height: 1.6; }
.bullet { border-top: 1px solid var(--border); padding: 14px 0 4px; }
.bullet:first-child { border-top: none; padding-top: 0; }
.bullettext { font-size: 14px; margin: 0 0 8px; line-height: 1.55; }
.bullettext code { font-family: var(--mono); font-size: 12.5px;
  background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }
.codewrap { margin: 0 0 8px; }
.codetoggle { display: flex; align-items: center; gap: 8px; width: 100%;
  background: var(--code-bg); border: 1px solid var(--border); border-radius: 7px;
  padding: 8px 12px; cursor: pointer; font-family: inherit; text-align: left;
  color: var(--text); }
.codetoggle:hover { border-color: var(--accent); }
.codetoggle .chev { font-size: 15px; }
.codelabel { font-size: 13px; font-weight: 600; }
.codemeta { margin-left: auto; font-size: 11.5px; color: var(--text-muted); }
.codemeta code { font-family: var(--mono); }
pre.code { margin: 8px 0 0; padding: 16px; background: var(--code-bg);
  border: 1px solid var(--border); border-radius: 7px; overflow-x: auto;
  max-height: 480px; }
pre.code code { font-family: var(--mono); font-size: 12px; color: var(--code-text);
  line-height: 1.55; white-space: pre; }
.note { font-size: 13px; color: var(--text-muted); margin: 28px 0 0;
  padding-top: 20px; border-top: 1px solid var(--border); }
"""

_SCRIPT = """
function toggleAddition(i) {
  var body = document.getElementById('addbody-' + i);
  var chev = document.getElementById('achev-' + i);
  var btn = chev.closest('.addhead');
  var open = !body.hidden;
  body.hidden = open;
  chev.classList.toggle('open', !open);
  btn.setAttribute('aria-expanded', String(!open));
}
function toggleCode(id) {
  var pre = document.getElementById(id);
  var chev = document.getElementById('chev-' + id);
  var btn = chev.closest('.codetoggle');
  var open = !pre.hidden;
  pre.hidden = open;
  chev.classList.toggle('open', !open);
  btn.setAttribute('aria-expanded', String(!open));
}
"""


def _code_block_html(rel_path: str | None, blocks: list[tuple], block_id_prefix: str) -> str:
    out = []
    for i, (lo, hi, code) in enumerate(blocks):
        block_id = f"{block_id_prefix}-{i}"
        range_txt = f"lines {lo}–{hi}" if lo is not None else "unresolved"
        path_txt = html.escape(rel_path) if rel_path else "?"
        out.append(f'''<div class="codewrap">
  <button class="codetoggle" onclick="toggleCode('{block_id}')" aria-expanded="false">
    <span class="chev" id="chev-{block_id}">&rsaquo;</span>
    <span class="codelabel">View embedded source</span>
    <span class="codemeta"><code>{path_txt}</code> &middot; {range_txt}</span>
  </button>
  <pre class="code" id="{block_id}" hidden><code>{html.escape(code)}</code></pre>
</div>''')
    return "".join(out)


def _addition_html(idx: int, addition: dict, resolved_entries: list[dict]) -> str:
    title = html.escape(addition.get("title", f"Addition {idx}"))
    desc = html.escape(addition.get("description", ""))
    bullets = []
    for j, entry in enumerate(resolved_entries):
        label = entry.get("label") or entry["process"]
        text = f"<code>{html.escape(entry['process'])}</code> &mdash; {html.escape(label)}" \
            if entry.get("label") else f"<code>{html.escape(entry['process'])}</code>"
        code_html = _code_block_html(entry["rel_path"], entry["blocks"], f"c{idx}-{j}")
        bullets.append(f'<div class="bullet"><div class="bullettext">{text}</div>{code_html}</div>')
    return f'''
<section class="addition">
  <button class="addhead" onclick="toggleAddition({idx})" aria-expanded="false">
    <span class="addnum">{idx}</span>
    <span class="addtitle">{title}</span>
    <span class="chev addchev" id="achev-{idx}">&rsaquo;</span>
  </button>
  <div class="addbody" id="addbody-{idx}" hidden>
    <p class="adddesc">{desc}</p>
    {"".join(bullets)}
  </div>
</section>'''


def build_report(fork: str, additions: list[dict], cand_prov: dict | None,
                 ref_prov: dict | None) -> str:
    sections = []
    for idx, addition in enumerate(additions, start=1):
        resolved = [_resolve_entry(fork, e) for e in addition.get("entries") or []]
        sections.append(_addition_html(idx, addition, resolved))

    cand_lbl = f"{cand_prov['name']}" if cand_prov else "the candidate"
    ref_lbl = f"{ref_prov['name']}" if ref_prov else "the reference"
    prov_bits = []
    if cand_prov:
        prov_bits.append(f"Candidate: <code>{html.escape(cand_prov['name'])}</code> "
                         f"@ <code>{html.escape(cand_prov.get('commit') or '?')}</code>")
    if ref_prov:
        prov_bits.append(f"Reference: <code>{html.escape(ref_prov['name'])}</code> "
                         f"@ <code>{html.escape(ref_prov.get('commit') or '?')}</code>")
    provenance = " &middot; ".join(prov_bits)

    return f'''<title>Private Repo Code Audit</title>
<style>{_STYLE}</style>
<div class="page">
  <p class="eyebrow">Code audit</p>
  <h1>{html.escape(cand_lbl)} &larr; {html.escape(ref_lbl)} &mdash; code audit</h1>
  <p class="subtitle">Everything transferred from the private reference repository into the candidate, with the full source embedded &mdash; not linked out.</p>
  <p class="provenance">{provenance}</p>
  <div class="banner">Our private repository has the following {len(additions)} addition{"s" if len(additions) != 1 else ""}:</div>
  {"".join(sections)}
  <p class="note">Generated deterministically from each study's private config (<code>code_audit</code> block) &mdash; addition titles/descriptions and excerpt patterns are config-declared, not hand-picked line numbers; re-running this script re-derives the same content from the current source.</p>
</div>
<script>{_SCRIPT}</script>
'''


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--investigation", required=True,
                   help="path or name of an investigation (study-YAML-only mode).")
    p.add_argument("--output-file", required=True, help="output HTML path.")
    args = p.parse_args(argv)

    from scripts._compare.study_spec import load_investigation
    _ctx, specs = load_investigation(args.investigation)

    fork = os.environ.get("V2E_VECOLI_DIR")
    if not fork:
        raise SystemExit("V2E_VECOLI_DIR must point at the private reference "
                         "repo checkout (the fork each study's config lives in).")

    cand = _git_provenance(str(Path(__file__).resolve().parents[1]))
    ref = _git_provenance(fork)

    additions: list[dict] = []
    per_study: dict[str, int] = {}
    for spec in specs:
        study_additions = _load_code_audit(fork, spec.config)
        per_study[spec.name] = len(study_additions)
        additions.extend(study_additions)

    html_doc = build_report(fork, additions, cand, ref)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    for name, n in per_study.items():
        print(f"{name}: {n} addition{'s' if n != 1 else ''}")
    print(f"total additions: {len(additions)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
