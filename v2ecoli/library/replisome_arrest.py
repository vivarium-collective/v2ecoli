"""Measure a mechanistic-vs-permissive replisome-gating lineage pair.

Science core for ``report_cards/replisome_arrest_card.py``. Reads the parquet
each arm emitted and returns the quantities the card grades. No grading and no
rendering here — that split matches ``genotype_build.py`` /
``report_card.py``.

What is measured, and why each is the observable it is:

``generations_completed`` / ``arrest_generation``
    Read from the runner's ``*_summary.json``, whose ``gens[]`` records a
    per-generation ``divided`` flag — the authoritative record. Counting
    parquet ``generation=`` partitions instead is wrong twice over: a completed
    lineage emits a trailing partition for the final daughters (12 generations
    -> 13 partitions), and the ARRESTED generation is the one that ran and
    failed to divide, not the one after it.

``subunit_margin``
    The initiation gate (``chromosome_replication.py``) fires only when every
    replisome trimer pool has ``6 * n_oriC`` copies and every monomer pool has
    ``2 * n_oriC``. For each pool this returns ``min(count - requirement)``
    over the arrested generation. A NEGATIVE margin means that pool was
    genuinely short; a positive margin across every pool means the gate blocked
    initiation for some reason OTHER than subunit availability.

    This is the discriminating observable. "The lineage arrested" is consistent
    with any number of failures; "the lineage arrested while every subunit pool
    was in surplus" is consistent with almost none.

The six pools are read from the cache config rather than hard-coded — an
earlier version of this study hard-coded five and silently omitted
``EG11500-MONOMER[c]`` (HolB), which would have made the margin look tighter
than it is.
"""
from __future__ import annotations

import glob
from pathlib import Path

TRIMER_MULT = 6      # per oriC, chromosome_replication.py initiation gate
MONOMER_MULT = 2

SUBUNIT_LABELS = {
    "CPLX0-2361[c]": "pol III core",
    "CPLX0-3761[c]": "beta clamp",
    "CPLX0-3621[c]": "DnaB hexamer",
    "EG10239-MONOMER[c]": "DnaG",
    "EG11500-MONOMER[c]": "HolB (delta')",
    "EG11412-MONOMER[c]": "HolA (delta)",
}


def subunit_groups(cache_dir: str | Path) -> tuple[list[str], list[str]]:
    """Return (trimer_ids, monomer_ids) as the initiation gate reads them."""
    from v2ecoli.core import load_cache_bundle
    cfg = load_cache_bundle(str(cache_dir))["configs"]["ecoli-chromosome-replication"]
    return (list(cfg["replisome_trimers_subunits"]),
            list(cfg["replisome_monomers_subunits"]))


def read_summary(arm_dir: str | Path) -> dict:
    """Load the runner's per-generation summary for one arm."""
    import json
    hits = sorted(Path(arm_dir).glob("*_summary.json"))
    if not hits:
        raise FileNotFoundError(f"no *_summary.json under {arm_dir}")
    return json.loads(hits[0].read_text())


def _history_files(arm_dir: str | Path, generation: int | None = None) -> list[str]:
    gen = f"generation={generation}" if generation is not None else "generation=*"
    return sorted(glob.glob(
        f"{arm_dir}/**/history/**/{gen}/**/*.pq", recursive=True))


def divided_generations(summary: dict) -> int:
    """Generations that actually divided (not merely ran)."""
    return sum(1 for g in summary.get("gens", []) if g.get("divided"))


def arrest_generation(summary: dict) -> "int | None":
    """First generation that ran but did not divide; None if none did."""
    for g in summary.get("gens", []):
        if not g.get("divided"):
            return int(g["gen"])
    return None



# ---------------------------------------------------------------------------
# Distilled-bundle fallback
# ---------------------------------------------------------------------------
#
# A finished study's bulk parquet is ~4 GB per run and gets deleted once the
# study is closed out; the seven columns every graded axis reads are distilled
# into workspace/studies/<study>/evidence/<label>.parquet first (see
# analyses/distill_evidence.py, whose --verify pass checks the distilled
# margins against the parquet before anything is removed).
#
# subunit_margins therefore prefers the parquet and falls back to that bundle,
# so a report card still rebuilds after the raw history is gone. Without this
# the deletion would silently turn every margin into an empty dict, and the card
# would report "0 pools graded" rather than failing.

_BUNDLE_LABELS = {
    "CPLX0-2361[c]": "pol_III_core",
    "CPLX0-3761[c]": "beta_clamp",
    "CPLX0-3621[c]": "DnaB_hexamer",
    "EG10239-MONOMER[c]": "DnaG",
    "EG11500-MONOMER[c]": "HolB",
    "EG11412-MONOMER[c]": "HolA",
}


def bundle_path(arm_dir: str | Path) -> "Path | None":
    """The distilled bundle for an arm dir, or None.

    ``out/<study>/<arm>[/<seed>]`` maps to
    ``workspace/studies/<study>/evidence/<arm>[__<seed>].parquet``.
    """
    arm_dir = Path(arm_dir).resolve()
    parts = list(arm_dir.parts)
    if "out" not in parts:
        return None
    i = parts.index("out")
    tail = parts[i + 1:]
    if len(tail) < 2:
        return None
    study, rest = tail[0], tail[1:]
    repo = Path(*parts[:i])
    cand = repo / "workspace" / "studies" / study / "evidence" / ("__".join(rest) + ".parquet")
    return cand if cand.is_file() else None


def _margins_from_bundle(path: Path, generation: int,
                         trimers: list[str], monomers: list[str]) -> dict:
    import polars as pl
    df = pl.read_parquet(path)
    df = df.filter(df["generation"] == generation)
    if df.height == 0:
        return {}
    oric = df["listeners__replication_data__number_of_oric"]
    out: dict[str, dict] = {}
    for mol, mult in [(m, TRIMER_MULT) for m in trimers] + \
                     [(m, MONOMER_MULT) for m in monomers]:
        col = _BUNDLE_LABELS.get(mol)
        if col is None or col not in df.columns:
            continue
        counts = df[col]
        out[mol] = {
            "label": SUBUNIT_LABELS.get(mol, mol),
            "requirement_per_oric": mult,
            "min_count": int(counts.min()),
            "margin": int((counts - oric * mult).min()),
            "source": "distilled bundle",
        }
    return out


def subunit_margins(arm_dir: str | Path, generation: int,
                    trimers: list[str], monomers: list[str]) -> dict:
    """Worst (count - requirement) per pool over one generation.

    Returns ``{molecule_id: {"label", "requirement_per_oric", "min_count",
    "margin"}}``. ``margin < 0`` for any pool means that pool was short of what
    the gate demanded at some point in the generation.
    """
    import polars as pl

    files = _history_files(arm_dir, generation)
    if not files:
        # Raw history deleted after close-out: fall back to the distilled bundle.
        bp = bundle_path(arm_dir)
        if bp is not None:
            return _margins_from_bundle(bp, generation, trimers, monomers)
        return {}
    df = pl.read_parquet(files)
    if "time" in df.columns:
        df = df.sort("time")

    oric = df["listeners__replication_data__number_of_oric"]
    ids = df["bulk__id"][0].to_list()
    out: dict[str, dict] = {}
    for mol, mult in [(m, TRIMER_MULT) for m in trimers] + \
                     [(m, MONOMER_MULT) for m in monomers]:
        if mol not in ids:
            continue
        counts = df["bulk__count"].list.get(ids.index(mol))
        margin = (counts - oric * mult).min()
        out[mol] = {
            "label": SUBUNIT_LABELS.get(mol, mol),
            "requirement_per_oric": mult,
            "min_count": int(counts.min()),
            "margin": int(margin),
        }
    return out


def measure(mechanistic_dir: str | Path, permissive_dir: str | Path,
            cache_dir: str | Path) -> dict:
    """Measure both arms. Returns the dict the card grades."""
    trimers, monomers = subunit_groups(cache_dir)
    mech = read_summary(mechanistic_dir)
    perm = read_summary(permissive_dir)
    mech_gens = divided_generations(mech)
    perm_gens = divided_generations(perm)

    # Grade the pools on the generation that RAN and failed to divide — that is
    # where the gate blocked. Falls back to the last generation present when the
    # arm never arrested.
    arrest_gen = arrest_generation(mech) or (mech.get("gens") or [{}])[-1].get("gen")
    margins = subunit_margins(mechanistic_dir, arrest_gen, trimers, monomers)
    worst = min((v["margin"] for v in margins.values()), default=None)
    limiting = None
    if margins:
        limiting = min(margins.items(), key=lambda kv: kv[1]["margin"])

    return {
        "mechanistic_generations": mech_gens,
        "permissive_generations": perm_gens,
        "generation_gap": perm_gens - mech_gens,
        "generations_requested": mech.get("generations_requested"),
        "arrest_dry_mass_fg": next(
            (g.get("final_dry_mass_fg") for g in mech.get("gens", [])
             if not g.get("divided")), None),
        "arrest_generation": arrest_gen,
        "subunit_margins": margins,
        "worst_subunit_margin": worst,
        "limiting_pool": limiting[1]["label"] if limiting else None,
        "limiting_pool_id": limiting[0] if limiting else None,
        "n_pools_graded": len(margins),
    }
