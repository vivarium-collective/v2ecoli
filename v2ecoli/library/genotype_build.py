"""Build-integrity science core for a ParCa-level genotype perturbation.

Sibling of ``v2ecoli/library/overflow.py``: this module owns the measurement, the
report card's grading + rendering stay in ``v2ecoli/library/report_card.py``, and
``workflow/report_cards/genotype_build_integrity_card.py`` is only the Step.

The question is layer 2 of the genotype-perturbation plan: *does a genotype, once
built through the ecoli-sources generator and ParCa, produce a self-consistent
knowledge base?* That sits between "the coordinate arithmetic is correct" (a pytest
matrix over a pure function) and "the physiology is interesting" (the descriptive
studies), and neither of those reaches it.

WHAT NEEDS A FIT, AND WHAT DOES NOT
-----------------------------------
Four of the five declared readouts are properties of ``raw_data`` and cost seconds:

  * feature-sequence-round-trip   -- every surviving feature's sequence at its new
                                     coordinates
  * chromosome-length             -- genome length before/after
  * feature-inventory             -- gene / TU / DNA-site / cistron counts
  * deleted-gene-set-membership   -- functional absence

The last one is worth spelling out because it looks like it should need a fit. The
study names ``valid_gene_ids`` (getter_functions.py) and ``all_mRNA_cistrons``
(translation.py); both are *local variables* rather than attributes on a built
sim_data, and both are computed from ``raw_data`` -- gated on the gene's
``left_end_pos`` / ``right_end_pos`` being non-None. A knockout TOMBSTONES its gene
(row retained, coordinates nulled), so the tombstone is precisely the mechanism that
drops it from both sets. This module reconstructs both sets faithfully from
``raw_data`` rather than reading them, and says so, because they cannot be read.

Only the fitted amino-acid supply kcats need a ParCa run, and in ``fast`` mode they
must not be graded: the CLI's own ``--allow-partial-fit`` help flags step 9's
mechanistic fits as failing on "numerically-marginal kinetics (e.g. debug mode's
truncated TF set)", which is exactly the fast regime.

A KNOCKOUT THAT FAILS TO BUILD IS A RESULT
------------------------------------------
Plan decision D17: the generator writes exactly the coordinate-coupled keys and
prunes nothing else, leaving 7-8 dangling references by design. A build that fails on
those is recorded as the outcome, not treated as a broken study -- for an essential
gene it may be the correct answer.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

# DnaA boxes are supplied by v2ecoli's flat_overrides/dna_sites.tsv, not by the
# ecoli-sources base table. A variant bundle GENERATES dna_sites, and a generated key
# takes precedence over the whole-file override (#466/#468), so these rows are absent
# from every knockout build regardless of which gene is deleted. Excluding them keeps
# the inventory measuring the perturbation instead of a constant of the build path.
# Tracked as v2ecoli#490.
DNAA_PREFIX = "G0-DNAA"

# Mirrors getter_functions.py's exclusion set for valid_gene_ids.
EXCLUDED_RNA_TYPES = ("pseudogene", "phantom_gene")


# --------------------------------------------------------------------------
# raw_data set reconstructions (see module docstring -- these cannot be read)
# --------------------------------------------------------------------------

def valid_gene_ids(raw_data: Any) -> set[str]:
    """Reconstruct getter_functions.py's ``valid_gene_ids``.

    Genes with positions on the chromosome that are not pseudogenes or phantom genes.
    """
    gene_to_type = {r["gene_id"]: r["type"] for r in raw_data.rnas}
    return {
        g["id"] for g in raw_data.genes
        if g["left_end_pos"] is not None and g["right_end_pos"] is not None
        and gene_to_type.get(g["id"]) not in EXCLUDED_RNA_TYPES
    }


def all_mrna_cistron_genes(raw_data: Any) -> set[str]:
    """Genes owning an mRNA cistron, per translation.py's ``all_mRNA_cistrons``.

    translation.py builds a set of RNA ids; this returns the GENE ids behind them,
    which is the addressing the study's readout uses.
    """
    pos = {g["id"]: (g["left_end_pos"], g["right_end_pos"]) for g in raw_data.genes}
    out = set()
    for rna in raw_data.rnas:
        gid = rna.get("gene_id")
        if gid is None or rna.get("type") != "mRNA":
            continue
        left, right = pos.get(gid, (None, None))
        if left is not None and right is not None:
            out.add(gid)
    return out


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def _feature_seq(feature: dict, genome: str) -> "str | None":
    left, right = feature.get("left_end_pos"), feature.get("right_end_pos")
    if left is None or right is None:
        return None
    return genome[left - 1:right]


def _dna_site_ids(raw_data: Any, *, exclude_dnaa: bool = True) -> dict[str, dict]:
    return {
        d["id"]: d for d in raw_data.dna_sites
        if not (exclude_dnaa and str(d["id"]).startswith(DNAA_PREFIX))
    }


def measure_structure(wt: Any, ko: Any, gene_ids: Iterable[str],
                      spans: dict[str, tuple[int, int]]) -> dict:
    """The four fit-free readouts, as a card node tree.

    ``spans`` maps gene id -> (left_end_pos, right_end_pos) in WILD-TYPE coordinates,
    as recorded by the generator in ``genotype.json``.
    """
    gene_ids = list(gene_ids)
    deleted_bp = sum(r - l + 1 for l, r in spans.values())

    # -- chromosome-length ------------------------------------------------
    wt_len, ko_len = len(wt.genome_sequence), len(ko.genome_sequence)
    observed_delta = wt_len - ko_len

    # -- feature-sequence-round-trip --------------------------------------
    # Compared by CONTENT, not length: a length-preserving off-by-one would pass a
    # length check while corrupting every downstream feature.
    wt_genes = {g["id"]: g for g in wt.genes}
    identical = differing = tombstoned = 0
    mismatches: list[str] = []
    for gid, kg in ((g["id"], g) for g in ko.genes):
        wg = wt_genes.get(gid)
        if wg is None:
            continue
        ks = _feature_seq(kg, ko.genome_sequence)
        if ks is None:
            tombstoned += 1
            continue
        if ks == _feature_seq(wg, wt.genome_sequence):
            identical += 1
        else:
            differing += 1
            if len(mismatches) < 10:
                mismatches.append(gid)

    # -- tombstone accounting ---------------------------------------------
    # The wild type already carries genes with null coordinates; only the DELTA is
    # attributable to the knockout. Comparing against 0 would fail on correct data.
    wt_null = {g["id"] for g in wt.genes if g["left_end_pos"] is None}
    ko_null = {g["id"] for g in ko.genes if g["left_end_pos"] is None}
    newly_tombstoned = sorted(ko_null - wt_null)
    resurrected = sorted(wt_null - ko_null)

    # -- downstream coordinate shift --------------------------------------
    # Every DNA site above the cut shifts by exactly the deleted length; everything
    # below it must not move. Checked per-site rather than in aggregate.
    wt_sites, ko_sites = _dna_site_ids(wt), _dna_site_ids(ko)
    hi = max((r for _, r in spans.values()), default=0)
    lo = min((l for l, _ in spans.values()), default=0)
    shifted = unmoved = 0
    wrong: list[str] = []
    for sid, ws in wt_sites.items():
        ks = ko_sites.get(sid)
        if ks is None:
            continue
        a, b = ws["left_end_pos"], ks["left_end_pos"]
        if a is None or b is None:
            continue
        if a > hi:
            if a - b == deleted_bp:
                shifted += 1
            else:
                wrong.append(sid)
        elif a < lo:
            if a == b:
                unmoved += 1
            else:
                wrong.append(sid)

    # -- feature-inventory -------------------------------------------------
    inventory = {
        "genes": {"wt": len(wt.genes), "ko": len(ko.genes)},
        "transcription_units": {"wt": len(wt.transcription_units),
                                "ko": len(ko.transcription_units)},
        "dna_sites_excl_dnaa": {"wt": len(wt_sites), "ko": len(ko_sites)},
    }

    # -- deleted-gene-set-membership ---------------------------------------
    ko_valid, ko_mrna = valid_gene_ids(ko), all_mrna_cistron_genes(ko)
    wt_valid, wt_mrna = valid_gene_ids(wt), all_mrna_cistron_genes(wt)
    memberships = {
        g: {"wt_valid_gene_ids": g in wt_valid, "wt_all_mRNA_cistrons": g in wt_mrna,
            "ko_valid_gene_ids": g in ko_valid, "ko_all_mRNA_cistrons": g in ko_mrna,
            "row_retained_in_genes": g in {x["id"] for x in ko.genes}}
        for g in gene_ids
    }
    remaining = sum(int(m["ko_valid_gene_ids"]) + int(m["ko_all_mRNA_cistrons"])
                    for m in memberships.values())

    return {
        "chromosome_length": {
            "ok": observed_delta == deleted_bp,
            "wt": wt_len, "ko": ko_len,
            "observed_delta": observed_delta, "expected_delta": deleted_bp,
        },
        "round_trip": {
            "ok": differing == 0,
            "identical": identical, "differing": differing,
            "tombstoned": tombstoned, "examples": mismatches,
        },
        "tombstone": {
            "ok": newly_tombstoned == sorted(gene_ids) and not resurrected,
            "newly_tombstoned": newly_tombstoned,
            "resurrected": resurrected,
            "wt_pre_existing_nulls": len(wt_null),
        },
        "coordinate_shift": {
            "ok": not wrong,
            "shifted": shifted, "unmoved": unmoved,
            "wrong": len(wrong), "examples": wrong[:10],
            "expected_shift_bp": deleted_bp,
        },
        "inventory": inventory,
        "functional_absence": {
            "ok": remaining == 0,
            "memberships_remaining": remaining,
            "detail": memberships,
        },
    }


def measure_fit(state: dict | None) -> dict:
    """Fit-dependent readouts from a ParCa ``parca_state``. ``None`` -> ungraded."""
    if state is None:
        return {"completed": None, "conditions_fitted": None, "mode": None}
    status = state.get("mechanistic_fit_status") or {}
    return {
        "completed": bool(status) and all(v == "ok" for v in status.values()),
        "status": status,
        # len(cell_specs), NOT len(conditions): `conditions` is the full condition
        # LIST and reads 51 in BOTH fast and full mode, so grading against it would
        # pass on a fast build and never discriminate. cell_specs holds the
        # conditions actually fitted (7 in fast mode).
        "conditions_fitted": len(state.get("cell_specs") or {}),
        "conditions_declared": len(state.get("conditions") or []),
    }


def load_state(path: "str | Path") -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------
# bundle generation + the card's (card, reference) pair
# --------------------------------------------------------------------------

def make_knockout_bundle(gene_ids: Iterable[str], workdir: "str | Path") -> tuple:
    """Emit a KO variant bundle. Returns ``(manifest_path, genotype_id, spans)``.

    ``spans`` comes from the generator's own ``genotype.json`` provenance rather than
    being re-derived here, so the card grades against what the generator recorded.
    """
    import json
    from processing.genotypes import knockout, compose_variant_bundle, genotype_id

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    result = knockout(list(gene_ids), workdir / "gen")
    manifest = compose_variant_bundle([result], workdir / "bundle")
    sidecar = json.loads((Path(manifest).parent / "genotype.json").read_text())
    spans: dict[str, tuple[int, int]] = {}
    for gen in sidecar.get("generators") or []:
        for gid, (left, right) in (gen.get("params", {}).get("coordinates") or {}).items():
            spans[gid] = (int(left), int(right))
    return Path(manifest), genotype_id(manifest), spans


def resolve_raw_data(manifest: "str | Path | None"):
    """Build a KnowledgeBaseEcoli from a bundle manifest (None -> the default bundle).

    Routed through the ParCa step's own resolver so the card exercises the same path
    a study run does, rather than a parallel reimplementation of it (v2ecoli#482).
    """
    from v2ecoli.processes.parca.steps.step_01_initialize import _resolve_raw_data
    return _resolve_raw_data({"raw_data": None,
                              "bundle_manifest": str(manifest) if manifest else ""})


def _reference(gene_ids: list[str], mode: "str | None", deleted_bp: int) -> dict:
    """Axis declarations + their criteria. Kept separate from the measurement so the
    card states what it grades independently of what any one build produced."""
    graded_fit = mode == "full"
    return {
        "title": f"Genotype build integrity — knockout of {', '.join(gene_ids)}",
        "stimulus": {
            "measured_model": f"v2ecoli ParCa ({mode or 'no fit'})",
            "reference_model": "v2ecoli wild-type build",
        },
        "axes": {
            "chromosome_length.ok": {
                "group": "Genome", "label": "Chromosome length matches the deleted span",
                "criterion": {"type": "boolean"},
                "detail": {"expected_delta_bp": deleted_bp},
            },
            "round_trip.ok": {
                "group": "Genome", "label": "Every surviving feature's sequence is preserved",
                "criterion": {"type": "boolean"},
            },
            "coordinate_shift.ok": {
                "group": "Genome", "label": "Downstream features shift exactly; upstream unmoved",
                "criterion": {"type": "boolean"},
            },
            "tombstone.ok": {
                "group": "Knowledge base", "label": "Exactly the deleted genes are tombstoned",
                "criterion": {"type": "boolean"},
            },
            "functional_absence.ok": {
                "group": "Knowledge base",
                "label": "Deleted genes leave valid_gene_ids and all_mRNA_cistrons",
                "criterion": {"type": "boolean"},
            },
            "fit.completed": {
                "group": "ParCa fit", "label": "Fit completes on the perturbed genome",
                "criterion": {"type": "boolean"} if mode else {"type": "status"},
            },
            "fit.conditions_fitted": {
                "group": "ParCa fit", "label": "Conditions fitted (len(cell_specs))",
                # Deliberately ungraded outside full mode: a fast build fits 7 of 51
                # by design, so grading it would report a true fact as a failure.
                "criterion": ({"type": "rel_tol", "reference": 51, "tol_rel": 0.0}
                              if graded_fit else {"type": "status"}),
            },
        },
    }


def build(gene_ids: Iterable[str], *, workdir: "str | Path",
          parca_state: "str | Path | None" = None,
          mode: "str | None" = None) -> tuple[dict, dict]:
    """Measure a genotype's build integrity. Returns ``(card, reference)``.

    ``parca_state`` is an optional built ParCa state for the fit axes; without one
    those axes stay ungraded and the four structural readouts still grade, which is
    the cheap path (seconds, no fit).
    """
    gene_ids = list(gene_ids)
    manifest, gid, spans = make_knockout_bundle(gene_ids, workdir)
    wt = resolve_raw_data(None)
    ko = resolve_raw_data(manifest)

    card = measure_structure(wt, ko, gene_ids, spans)
    fit = measure_fit(load_state(parca_state) if parca_state else None)
    card["fit"] = fit
    card["genotype"] = {"gene_ids": gene_ids, "genotype_id": gid,
                        "manifest": str(manifest)}

    deleted_bp = sum(r - l + 1 for l, r in spans.values())
    return card, _reference(gene_ids, mode, deleted_bp)
