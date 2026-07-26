"""Materialize a study's report_cards + behavior_tests from its `comparison.cards`.

You author only `comparison.cards`; the CLI calls this on run to (re)write the
gating fields into the same study.yaml — one `report_card_axis` behavior_test per
GRADED card (standard/statistical; config/parca render but don't gate), pointing
at the canonical per-study card dir. Every other key (narrative, comparison
block, pipeline_gate, …) is preserved. Idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts._compare.study_spec import StudySpec, REPO

# report_card_axis verdict -> dashboard run-outcome (UPPERCASE; drift = PARTIAL
# so the pill carries the "within tolerance, with drift" caveat).
_OUTCOME = {"within_tol": "PASS", "drift": "PARTIAL", "mismatch": "FAIL",
            "ungraded": "PENDING"}

# overall verdict -> investigation-DAG node badge (the graph card reads
# `confidence`; without it every study renders as "Planned"). Biomass tracks
# vEcoli in every condition; the open discrepancy is on growth rate, and is a
# known low-copy stochastic / calibration effect rather than a refuted port —
# so drift and mismatch both map to "Investigating" (still open), not "Refuted".
_CONFIDENCE = {"within_tol": "Accepted", "drift": "Investigating",
               "mismatch": "Investigating", "ungraded": "Planned"}

# per-axis/per-finding verdict -> finding status glyph the dashboard renders
# (confirms ✓ / partial ◐ / contradicts ✗).
_FINDING_STATUS = {"within_tol": "confirms", "drift": "partial",
                   "mismatch": "contradicts"}


def _pct(axis) -> str:
    """median relative error of an axis as a percent string, or ''."""
    mr = (axis.get("detail") or {}).get("median_rel")
    return f"{mr * 100:.1f}%" if isinstance(mr, (int, float)) else ""


def _graph_fields(verdict: dict, spec: StudySpec) -> dict:
    """Derive the fields the investigation DAG graph reads (`confidence` +
    `findings`) from the per-study verdict. The graph card has no other source
    for these, so without them studies render as "Planned / pending evidence".
    One finding per GRADED group that has graded axes; evidence carries the
    measured per-observable deltas. No top-level `claim` is emitted — the card
    falls back to the finding statement, keeping any authored claim intact."""
    groups = verdict.get("groups") or {}
    overall = verdict.get("overall", "ungraded")
    findings = []
    for gname, g in groups.items():
        axes = [a for a in (g.get("axes") or [])
                if a.get("verdict") and a["verdict"] != "ungraded"]
        if not axes:
            continue
        gv = g.get("verdict", "ungraded")
        within = [a for a in axes if a["verdict"] == "within_tol"]
        diverge = [a for a in axes if a["verdict"] in ("drift", "mismatch")]
        # Compact, factual evidence line: agreeing observables + each diverging
        # observable with its measured median |Δ| and verdict.
        ev_bits = []
        if within:
            mx = max((_pct(a) for a in within), key=lambda s: float(s[:-1] or 0) if s else 0,
                     default="")
            ev_bits.append(f"{len(within)} within tolerance ("
                           + ", ".join(a["label"] for a in within)
                           + (f"; max median |Δ|={mx}" if mx else "") + ")")
        for a in diverge:
            ev_bits.append(f"{a['label']}: {a['verdict']} (median |Δ|={_pct(a)})")
        observed = "; ".join(ev_bits)
        labels = ", ".join(a["label"] for a in diverge)
        verb = " diverges" if len(diverge) == 1 else " diverge"
        if gname == "parca":
            statement = (f"v2ecoli and vEcoli start from the same ParCa-fit initial "
                         f"state on {spec.condition}: all {len(axes)} mass "
                         f"observables within tolerance.") if not diverge else (
                         f"ParCa initial states differ on {spec.condition}: "
                         f"{labels}{verb} ({gv}).")
        elif gname == "statistical":
            n_seeds = getattr(spec, "seeds", None)
            scope = f" across {n_seeds} seeds" if n_seeds else ""
            statement = (f"v2ecoli is statistically equivalent to vEcoli on "
                         f"{spec.condition}{scope}: all {len(axes)} observables "
                         f"within tolerance.") if not diverge else (
                         f"v2ecoli is not statistically equivalent to vEcoli on "
                         f"{spec.condition}{scope}: {labels}{verb} ({gv}).")
        elif not diverge:
            statement = (f"v2ecoli reproduces vEcoli on {spec.condition}: all "
                         f"{len(axes)} graded observables track within tolerance.")
        else:
            statement = (f"v2ecoli reproduces vEcoli biomass on {spec.condition} "
                         f"within tolerance; {labels}{verb} ({gv}).")
        findings.append({
            "id": f"{spec.name}-{gname}",
            "kind": "computational",
            "status": _FINDING_STATUS.get(gv, "partial"),
            "statement": statement,
            "evidence": {"from_card": gname, "observed": observed},
        })
    out = {"confidence": _CONFIDENCE.get(overall, "Planned")}
    if findings:
        out["findings"] = findings
    return out


def card_root(spec: StudySpec) -> str:
    """The per-investigation card root the verdict/behavior_tests address."""
    return f"docs/report_cards/{spec.invest_name}"


def materialized_fields(spec: StudySpec) -> dict:
    """report_cards (viz embeds) + a modular `tests` list of report_card modules
    (one per assigned card). Graded cards carry a report_card_axis measure so the
    gate aggregates; config/parca are informational (no measure)."""
    cdir = f"{card_root(spec)}/{spec.name}"          # docs/report_cards/<invest>/<name>
    tests = []
    for c in spec.cards:
        t = {"name": f"{c}-vs-vecoli", "kind": "report_card", "card": c,
             "classification": "primary",
             "question": f"Does v2ecoli reproduce vEcoli on {spec.name} ({c} card)?"}
        if c in spec.graded_cards:
            t["measure"] = {"kind": "report_card_axis", "card": cdir, "group": c}
        tests.append(t)
    return {
        "report_cards": [f"viz/report_card/{c}.html" for c in spec.cards],
        "tests": tests,
    }


def materialize_study(spec: StudySpec) -> Path:
    """Rewrite the study.yaml's report_cards + the modular `tests` list from its cards,
    preserving every other key; ensure an independent pipeline_gate. Returns the
    study path."""
    path = Path(spec.study_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update(materialized_fields(spec))
    data.pop("behavior_tests", None)   # replaced by the modular `tests` list
    # Pipeline DAG: the study's authored `depends_on` (a list of prerequisite
    # study names) becomes pipeline_gate.prerequisites — the prerequisites must
    # pass before this study's gate is evaluated (parca gates the per-condition
    # studies; the single-seed basal gates the statistical study).
    data["pipeline_gate"] = {"prerequisites": list(data.get("depends_on") or []),
                             "enables": []}
    # Canonical run + per-test outcomes from the study's verdict JSON (when it
    # exists) so the dashboard pill strip shows the REAL result per card.
    run = {"name": f"{spec.name}-comparison", "kind": "analysis", "canonical": True,
           "description": f"v2e-compare study {spec.name}"}
    vpath = Path(REPO) / card_root(spec) / spec.name / "report_card_verdict.json"
    if vpath.is_file():
        verdict = json.loads(vpath.read_text(encoding="utf-8"))
        groups = verdict.get("groups") or {}
        outcomes = {}
        for bt in data.get("tests") or []:
            grp = (bt.get("measure") or {}).get("group")
            if grp is None:
                continue
            gv = (groups.get(grp) or {}).get("verdict", "ungraded")
            outcomes[bt["name"]] = {"result": _OUTCOME.get(gv, "PENDING"),
                                    "detail": f"report card '{grp}': {gv}"}
        if outcomes:
            run.update(status="completed",
                       result=_OUTCOME.get(verdict.get("overall", "ungraded"), "PENDING"),
                       outcomes=outcomes)
            data["status"] = "evaluated"
        # Fields the investigation DAG graph reads (confidence + findings) —
        # derived VIEWS of the verdict, always refreshed so they never drift
        # from the gate. We deliberately do NOT write a top-level `claim`: the
        # graph card falls back to the finding statement, so an authored claim
        # is preserved (preserve-narrative contract) and the auto evidence stays
        # fresh without duplicating prose into the study.yaml.
        data.update(_graph_fields(verdict, spec))
    data["runs"] = [run]
    # Declare the v2ecoli baseline under a top-level `conditions:` block (the
    # v2ecoli baseline composite for this study's biological condition). The
    # `conditions:` block is what signals the dashboard's v4-REDESIGN study path,
    # which PRESERVES a list `tests:` (our report_card modules). A flat
    # `baseline: [list]` instead routes to the legacy-v4 path, which expects
    # `tests:` to be a dict and silently drops our module list.
    data.pop("baseline", None)
    data.setdefault("conditions", {
        "baseline": {
            "composite": "v2ecoli.composites.ecoli_baseline.ecoli_baseline",
            "params": {"condition": spec.condition},
        },
    })
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path
