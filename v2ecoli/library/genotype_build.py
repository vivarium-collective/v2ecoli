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


def _coord(value: Any) -> "int | None":
    """Normalize a flat-file coordinate. The TSVs are JSON-typed but not uniformly:
    transcription-unit rows can carry string coordinates, and overlay-derived rows
    hold '' rather than null (D20). Both read as "no coordinate" here."""
    if value is None or value == "":
        return None
    return int(value)


def _expected_after_deletion(wt_seq: str, left: int, right: int,
                             spans: Iterable[tuple[int, int]]) -> str:
    """The sequence a wild-type feature at [left, right] (1-based inclusive) should
    carry after the deletion spans are excised. Spans are in wild-type coordinates;
    excising in descending order keeps the relative indices of lower spans valid."""
    out = wt_seq
    for l, r in sorted(spans, key=lambda s: s[0], reverse=True):
        lo, hi = max(left, l), min(right, r)
        if lo <= hi:
            out = out[:lo - left] + out[hi - left + 1:]
    return out


def _round_trip_features(wt_feats: dict, ko_feats: dict, wt_genome: str,
                         ko_genome: str, spans: dict) -> dict:
    """Two-branch round trip over one feature class.

    Unchanged length => the sequence must be byte-identical to the wild-type read;
    shrunk => it must equal the wild-type read with exactly the deleted span excised.
    The single-branch "everything identical" form is vacuous for an off-by-one
    splice (it preserves every length), which is why both branches exist and the
    shrunk count is reported: zero shrunk features means the second branch was
    never exercised.
    """
    identical = shrunk_ok = differing = no_coords = 0
    examples: list[str] = []
    for fid, kf in ko_feats.items():
        wf = wt_feats.get(fid)
        if wf is None:
            continue
        wl, wr = _coord(wf.get("left_end_pos")), _coord(wf.get("right_end_pos"))
        kl, kr = _coord(kf.get("left_end_pos")), _coord(kf.get("right_end_pos"))
        if None in (wl, wr, kl, kr):
            no_coords += 1
            continue
        ws, ks = wt_genome[wl - 1:wr], ko_genome[kl - 1:kr]
        expected = _expected_after_deletion(ws, wl, wr, spans.values())
        if ks == expected:
            if len(ks) == len(ws):
                identical += 1
            else:
                shrunk_ok += 1
        else:
            differing += 1
            if len(examples) < 10:
                examples.append(fid)
    return {"identical": identical, "shrunk_excised_ok": shrunk_ok,
            "differing": differing, "no_coords": no_coords, "examples": examples}


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

    # -- feature-sequence-round-trip (two-branch, three feature classes) ----
    # Compared by CONTENT, not length: a length-preserving off-by-one would pass a
    # length check while corrupting every downstream feature. Genes never shrink
    # (the target tombstones instead), so transcription units are what exercise the
    # shrunk branch — on the real KB a lacY deletion truncates lac-operon TUs.
    wt_genes = {g["id"]: g for g in wt.genes}
    wt_sites, ko_sites = _dna_site_ids(wt), _dna_site_ids(ko)
    rt_classes = {
        "genes": _round_trip_features(
            wt_genes, {g["id"]: g for g in ko.genes},
            wt.genome_sequence, ko.genome_sequence, spans),
        "transcription_units": _round_trip_features(
            {t["id"]: t for t in wt.transcription_units},
            {t["id"]: t for t in ko.transcription_units},
            wt.genome_sequence, ko.genome_sequence, spans),
        "dna_sites": _round_trip_features(
            wt_sites, ko_sites, wt.genome_sequence, ko.genome_sequence, spans),
    }

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

    # -- coordinate bounds --------------------------------------------------
    # Every surviving coordinate must satisfy 1 <= left <= right <= genome length.
    # Catches a shift applied in the wrong direction or past the shortened end.
    # A half-null pair counts as a violation: the transform's own guard requires
    # both coordinates null (tombstone) or both present.
    bounds_checked = 0
    bound_violations: list[str] = []
    for cls, rows in (("genes", ko.genes),
                      ("transcription_units", ko.transcription_units),
                      ("dna_sites", ko.dna_sites)):
        for row in rows:
            try:
                left = _coord(row.get("left_end_pos"))
                right = _coord(row.get("right_end_pos"))
            except (TypeError, ValueError):
                bounds_checked += 1
                bound_violations.append(f"{cls}:{row.get('id')}")
                continue
            if left is None and right is None:
                continue
            bounds_checked += 1
            if left is None or right is None or not (1 <= left <= right <= ko_len):
                bound_violations.append(f"{cls}:{row.get('id')}")

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
            "ok": all(c["differing"] == 0 for c in rt_classes.values()),
            "identical": sum(c["identical"] for c in rt_classes.values()),
            "shrunk_excised_ok": sum(c["shrunk_excised_ok"]
                                     for c in rt_classes.values()),
            "differing": sum(c["differing"] for c in rt_classes.values()),
            "tombstoned": rt_classes["genes"]["no_coords"],
            "by_class": rt_classes,
            "examples": [e for c in rt_classes.values() for e in c["examples"]][:10],
        },
        "coordinate_bounds": {
            "ok": not bound_violations,
            "checked": bounds_checked,
            "violations": len(bound_violations),
            "examples": bound_violations[:10],
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
    """Load a ParCa state, gzipped or not.

    Delegates to the ParCa data_loader rather than calling ``pickle.load`` here: that
    loader carries a remapping Unpickler for legacy module paths (``v2parca.*``,
    bare ``reconstruction.*``), so a state written by an older tree still loads.
    """
    from v2ecoli.processes.parca.data_loader import load_parca_state
    return load_parca_state(path)


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
                "group": "Genome",
                "label": "Two-branch round trip over genes, TUs and DNA sites: "
                         "unchanged features byte-identical, straddling features "
                         "exactly excised",
                "criterion": {"type": "boolean"},
            },
            "coordinate_bounds.ok": {
                "group": "Genome",
                "label": "Every coordinate lies within 1..genome_length",
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
    # The manifest is recorded relative to the (untracked, machine-local) workdir:
    # an absolute path would leak the local filesystem layout into a committed
    # verdict and break byte-identical re-rendering across machines. The
    # content-addressed genotype_id is the durable identity; the path is a hint.
    card["genotype"] = {"gene_ids": gene_ids, "genotype_id": gid,
                        "manifest": str(Path(manifest).resolve().relative_to(
                            Path(workdir).resolve()))}

    deleted_bp = sum(r - l + 1 for l, r in spans.values())
    return card, _reference(gene_ids, mode, deleted_bp)
