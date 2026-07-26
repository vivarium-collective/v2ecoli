"""Translate a v2ecoli molecular state into a 3D structural model.

The v2ecoli-specific half of the structural pipeline: pick which species to
place, map them to real structures (curated PDB assemblies + AlphaFold per
UniProt), label them with EcoCyc names + functional categories, and hand the
ingredient list to :func:`pbg_parsimony.build_pack` (the generic engine).

State source: a saved snapshot (``data/v2ecoli_state.npz``, the default — fast,
reproducible) or a live ``baseline`` composite run (``state_source="live"``).
"""
from __future__ import annotations
import ast
import importlib.util
import json
import logging
import os
from pathlib import Path

import numpy as np

from pbg_parsimony import Ingredient, Capsule, Chromosome, StructureRef, build_pack

log = logging.getLogger(__name__)

DATA = Path(__file__).parent / "data"

# ── Chromosome/replication geometry constants (E. coli K-12 MG1655) ─────────
# Genome length and the per-replichore length the replication forks travel
# (oriC → terC). Mirrors feat/3d-transcription-translation's build.py so
# fork_fraction is computed identically from live replisome coordinates.
GENOME_BP = 4_641_652
REPLICHORE_BP = GENOME_BP // 2  # 2,320,826

# Category → display colour (RGB 0–1).
CATEGORY_COLOR = {
    "Translation": (0.95, 0.55, 0.25), "Transcription": (0.35, 0.6, 0.95),
    "Nucleoid": (0.85, 0.75, 0.45), "Metabolism": (0.45, 0.8, 0.5),
    "Protein folding": (0.95, 0.85, 0.3), "Envelope": (0.8, 0.55, 0.85),
    "Regulation": (0.9, 0.4, 0.5),
}

# Large assemblies whose abundance is best taken as a representative count and
# whose structure is a curated PDB/mmCIF (AlphaFold gives only monomers).
CURATED = [
    # id,             gene,  category,           structure,         count_key,       region
    ("70S_ribosome",  None, "Translation",      ("cif", "4YBB"),   20000,           "interior"),
    ("rna_polymerase", None, "Transcription",   ("pdb", "4YG2"),   2000,            "fiber"),
    ("groel",         None, "Protein folding",  ("pdb", "1AON"),   1500,            "interior"),
    ("EG10367-MONOMER", None, "Metabolism",     "af",              "GAPDH-A-CPLX",  "interior"),  # GAPDH (complex abundance)
]
DISPLAY = {
    "70S_ribosome": "70S ribosome", "rna_polymerase": "RNA polymerase",
    "groel": "GroEL/ES chaperonin",
    "EG10367-MONOMER": "glyceraldehyde-3-phosphate dehydrogenase (GAPDH)",
}


def _flat_dir() -> Path:
    return Path(importlib.util.find_spec("reconstruction.ecoli.flat").submodule_search_locations[0])


def _load_tsv(name):
    rows, header = [], None
    for line in open(_flat_dir() / name):
        if line.startswith("#"):
            continue
        cells = [c.strip().strip('"') for c in line.rstrip("\n").split("\t")]
        if header is None:
            header = cells
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _proteins():
    return {r["id"]: r["common_name"] for r in _load_tsv("proteins.tsv")}


def _genes():
    return {r["id"]: r for r in _load_tsv("genes.tsv")}


def _uniprot_map():
    return json.load(open(DATA / "uniprot_map.json"))


def categorize(name: str) -> str:
    """Coarse functional category from an EcoCyc common name (ordered so e.g.
    'chaperone protein DnaK' lands in Protein folding, not Nucleoid)."""
    n = (name or "").lower()
    if any(k in n for k in ("chaperon", "heat shock", "foldase", "trigger factor",
                            "disulfide", "protease", "peptidase")) or ("prolyl" in n and "isomerase" in n):
        return "Protein folding"
    if any(k in n for k in ("ribosom", "elongation factor", "ef-", "trna", "aminoacyl",
                            "initiation factor", "translation")):
        return "Translation"
    if any(k in n for k in ("rna polymerase", "transcription termin", "transcription antitermin", "sigma factor")):
        return "Transcription"
    if any(k in n for k in ("regulator", "repressor", "activator", "transcriptional dual")):
        return "Regulation"
    if any(k in n for k in ("dna-binding", "dna gyrase", "dna polymerase", "topoisomerase",
                            "histone-like", "nucleoid", "hu-", "h-ns", "recombinase", "replicat")):
        return "Nucleoid"
    if any(k in n for k in ("outer membrane", "periplasm", "lipoprotein", "membrane", "porin",
                            "fimbri", "pilus", "flagell", "secret", "efflux", "transporter", " abc ")):
        return "Envelope"
    return "Metabolism"


def bulk_to_counts(bulk) -> dict:
    """A live ``['bulk']`` store (structured array with ``id``/``count`` fields)
    → ``{ecocyc_id: summed_count}``, compartment tags stripped."""
    ids = [str(x) for x in bulk["id"]]
    cnts = list(bulk["count"])
    counts = {}
    for idt, c in zip(ids, cnts):
        base = idt[:-1].rsplit("[", 1)[0] if idt.endswith("]") and "[" in idt else idt
        counts[base] = counts.get(base, 0) + int(c)
    return counts


# E. coli bulk-id compartment tag (the trailing ``[x]``) → parsimony envelope
# compartment. ``e``/``l``/``j``/``s`` and untagged ids fall through to
# "cytoplasm" via ``.get(tag, "cytoplasm")`` below.
_TAG_TO_COMPARTMENT = {
    "c": "cytoplasm", "i": "inner_membrane", "p": "periplasm",
    "o": "outer_membrane", "m": "inner_membrane",  # generic membrane → inner
}


def bulk_to_locations(bulk) -> dict:
    """{ecocyc_id: parsimony compartment} from the [x] compartment tag on each
    bulk id — the molecule's dominant location (by summed count across tags)."""
    ids = [str(x) for x in bulk["id"]]
    cnts = list(bulk["count"])
    # base_id -> {compartment: total count}
    agg = {}
    for idt, c in zip(ids, cnts):
        if idt.endswith("]") and "[" in idt:
            base, tag = idt[:-1].rsplit("[", 1)
        else:
            base, tag = idt, "c"
        comp = _TAG_TO_COMPARTMENT.get(tag, "cytoplasm")
        agg.setdefault(base, {}).setdefault(comp, 0)
        agg[base][comp] += int(c)
    # dominant compartment per base id
    return {base: max(comps.items(), key=lambda kv: kv[1])[0] for base, comps in agg.items()}


def load_state(state_source="snapshot", advance_s=2.0, seed=0):
    """Return ``(counts, volume_fl, locations)``: counts is ``{ecocyc_id: count}``
    (compartment tags stripped, summed); volume in fL; locations is
    ``{ecocyc_id: parsimony compartment}`` from the dominant ``[x]`` tag."""
    if state_source == "live":
        import v2ecoli
        comp = v2ecoli.build_composite("baseline", seed=seed, cache_dir="out/cache")
        comp.run(advance_s)
        cell = comp.state.get("agents", {}).get("0", comp.state)
        bulk = cell["bulk"]
        vol = cell["listeners"]["mass"]["volume"]
        volume_fl = float(getattr(vol, "magnitude", vol))
    else:
        st = np.load(DATA / "v2ecoli_state.npz")
        bulk = {"id": st["ids"], "count": st["counts"]}
        volume_fl = float(st["volume"])
    return bulk_to_counts(bulk), volume_fl, bulk_to_locations(bulk)


def _bnum(gene_id, genes):
    g = genes.get(gene_id)
    if not g:
        return None
    try:
        syn = ast.literal_eval(g["synonyms"])
    except Exception:
        syn = []
    for s in syn:
        if s.startswith("b") and s[1:].isdigit():
            return s
    return None


def _uniprot(ecocyc, genes, umap, gene_symbol=None):
    gene = ecocyc[:-len("-MONOMER")] if ecocyc.endswith("-MONOMER") else ecocyc
    b = _bnum(gene, genes)
    acc = umap["by_bnumber"].get(b) if b else None
    if not acc and gene_symbol:
        acc = umap["by_gene"].get(gene_symbol.lower())
    return acc


def select_ingredients(counts, locations=None, *, top_n=40, lipid_count=40000):
    """Curated assemblies + the top-N most-abundant protein monomers (AlphaFold,
    skipping individual ribosomal proteins) + a membrane lipid. Returns a list of
    :class:`pbg_parsimony.Ingredient` (counts are pre-scale; build_pack scales).

    ``locations`` (optional ``{ecocyc_id: parsimony compartment}``, from
    :func:`bulk_to_locations`) routes each ingredient to its real envelope
    compartment (cytoplasm/periplasm/inner_membrane/outer_membrane); a
    molecule's surface-vs-interior ``region`` follows its real compartment,
    not its functional category."""
    locations = locations or {}
    prot, genes, umap = _proteins(), _genes(), _uniprot_map()
    ingredients, already = [], set()

    for key, gene, cat, struct, ckey, region in CURATED:
        if isinstance(struct, tuple):
            ref = StructureRef(struct[0], struct[1])
            acc = None
        else:
            acc = _uniprot(key, genes, umap, gene)
            if not acc:
                continue
            ref = StructureRef("alphafold", acc)
        cnt = counts.get(ckey, 0) if isinstance(ckey, str) else int(ckey)
        # Keep CURATED's hand-authored region (e.g. rna_polymerase's "fiber",
        # which routes it to the fiber_pack stage seated along the DNA) —
        # only the compartment is derived from the real location tag.
        compartment = locations.get(ckey, "cytoplasm") if isinstance(ckey, str) else "cytoplasm"
        ingredients.append(Ingredient(
            id=key, count=max(1, cnt), structure=ref, region=region,
            display_name=DISPLAY.get(key, prot.get(key, key)), category=cat,
            color=CATEGORY_COLOR[cat], compartment=compartment,
            proxy_voxel_size=12.0 if isinstance(struct, tuple) else None))
        already.add(key)
        if isinstance(ckey, str):
            already.add(ckey)

    # Auto-expand: top-N abundant protein monomers, AlphaFold-modelled.
    monomers = sorted(((mid, c) for mid, c in counts.items()
                       if mid in prot and c > 0 and mid not in already),
                      key=lambda kv: -kv[1])
    added = 0
    for mid, c in monomers:
        if added >= top_n:
            break
        nm = prot.get(mid) or ""
        if not nm or nm == "null" or "ribosomal subunit protein" in nm.lower():
            continue
        acc = _uniprot(mid, genes, umap)
        if not acc:
            continue
        cat = categorize(nm)
        compartment = locations.get(mid, "cytoplasm")
        region = "surface" if compartment in ("inner_membrane", "outer_membrane") else "interior"
        ingredients.append(Ingredient(
            id=mid, count=c, structure=StructureRef("alphafold", acc), region=region,
            display_name=nm, category=cat, color=CATEGORY_COLOR[cat],
            compartment=compartment))
        added += 1

    ingredients.append(Ingredient(
        id="lipid", count=lipid_count, sphere_radius=12.0, region="surface",
        display_name="Membrane phospholipid", category="Envelope",
        color=(0.75, 0.78, 0.85), principal_vector=(0, 0, 1),
        compartment="inner_membrane"))
    return ingredients


def relax_ingredients(ingredients, *, cache_dir, relax_cfg):
    """Rewrite each AlphaFold ingredient's structure to its cached RELAXED
    structure (best-effort per ingredient: any failure keeps the raw ref so the
    pack still builds). Only ``alphafold`` structures are relaxed: they are the
    predicted models whose low-confidence disordered tails stick out and want
    compacting. Experimental ``pdb``/``cif`` assemblies (the curated 70S
    ribosome, RNA polymerase, GroEL, ...) are left untouched — they are already
    well-resolved and far too large to solvate all-atom. Ingredients without a
    structure (e.g. the lipid sphere) pass through untouched. Returns a NEW list."""
    from dataclasses import replace
    from pbg_parsimony.relax_cache import get_or_relax
    out = []
    for ing in ingredients:
        s = getattr(ing, "structure", None)
        if s is not None and s.kind == "alphafold":
            try:
                p = get_or_relax({"kind": s.kind, "ref": s.ref}, cache_dir,
                                 relax_cfg, obj_id=ing.id)
                ing = replace(ing, structure=StructureRef("file", str(p)))
            except Exception as exc:
                log.warning("relax skipped for %s (%s); using raw structure",
                            ing.id, exc)
        out.append(ing)
    return out


def _active_rows(arr):
    """Filter a live unique-molecule structured array down to its ACTIVE rows
    (``_entryState`` != 0). v2ecoli's unique-molecule stores are pre-allocated
    with spare capacity; inactive slots must be excluded from any count or
    per-row read — mirrors the ``arr[arr["_entryState"].view(bool)]`` pattern
    used throughout ``v2ecoli/processes/*`` and ``v2ecoli/bridge.py``.

    Arrays lacking an ``_entryState`` field (e.g. synthetic arrays built
    directly by unit tests) are returned unchanged — every row is treated as
    active, so tests can hand in bare structured arrays with just the fields
    they care about.
    """
    if arr is None:
        return None
    if getattr(arr, "dtype", None) is not None and arr.dtype.names and "_entryState" in arr.dtype.names:
        return arr[arr["_entryState"].astype(bool)]
    return arr


def _descendant_domains_set(domain_children: dict, root: int) -> set:
    """All transitive descendants of ``root`` (excluding root itself).

    Ported verbatim from feat/3d-transcription-translation's build.py (mirrors
    ``v2ecoli/visualizations/workflow.py::_descendant_domains``).
    """
    seen: set = set()
    stack = list((domain_children or {}).get(root, []))
    while stack:
        d = stack.pop()
        if d in seen:
            continue
        seen.add(d)
        stack.extend((domain_children or {}).get(d, []))
    return seen


def classify_domains(domain_children: dict, full_chromosome_domains: list,
                     query_domains: "np.ndarray"):
    """Classify per-RNAP domain indices by chromosome + daughter status.

    Ported verbatim from feat/3d-transcription-translation's build.py.

    Parameters
    ----------
    domain_children:
        Mapping ``parent_domain_index -> [child_domain_index, ...]`` (as
        produced by reading ``chromosome_domain`` unique-molecule fields).
    full_chromosome_domains:
        Ordered list of root ``domain_index`` values for each full chromosome
        (from the ``full_chromosome`` unique molecule). The *k*-th entry in
        this list defines chromosome index *k*.
    query_domains:
        1-D int array of per-RNAP ``domain_index`` values to classify.

    Returns
    -------
    chromosome_index : np.ndarray[int32]
        Per-RNAP chromosome index (0-based). Unmatched entries -> 0.
    is_daughter : np.ndarray[bool]
        True when the RNAP is on a replicated (daughter) copy of chromosome
        *k* (i.e. its domain_index != the root domain of the matched
        chromosome). Unmatched entries -> False.
    """
    n = len(query_domains)
    chromosome_index = np.zeros(n, dtype=np.int32)
    is_daughter = np.zeros(n, dtype=bool)

    lineages: list = []
    for root in full_chromosome_domains:
        lineage = {root} | _descendant_domains_set(domain_children, root)
        lineages.append((root, lineage))

    for i, dom in enumerate(query_domains):
        d = int(dom)
        for k, (root, lineage) in enumerate(lineages):
            if d in lineage:
                chromosome_index[i] = k
                is_daughter[i] = (d != root)
                break
        # If not matched: defaults remain (chromosome_index=0, is_daughter=False).

    return chromosome_index, is_daughter


def chromosome_state_from_live(full_chromosome, active_replisome=None) -> "tuple[int, float]":
    """``(n_chromosomes, fork_fraction)`` from LIVE unique-molecule arrays.

    ``full_chromosome`` is the live ``['unique']['full_chromosome']`` store
    (one row per chromosome copy; ``n_chromosomes`` = its active row count).
    ``fork_fraction`` is the mean replication-fork position — read from the
    live ``active_replisome`` store's ``coordinates`` field, expressed as a
    fraction of :data:`REPLICHORE_BP` (0 = unreplicated) — computed exactly
    the way ``scripts/capture_structural_snapshot.py`` derives it on
    ``feat/3d-transcription-translation``.

    When ``active_replisome`` isn't wired/available, ``fork_fraction`` falls
    back to ``0.0`` (an unreplicated chromosome) rather than fabricating a
    value, and a message is logged.
    """
    fc = _active_rows(full_chromosome)
    n_chromosomes = int(len(fc)) if fc is not None else 0

    fork_fraction = 0.0
    if active_replisome is None:
        log.info("chromosome_state_from_live: no active_replisome data available; "
                 "fork_fraction defaults to 0.0 (unreplicated)")
    else:
        rep = _active_rows(active_replisome)
        if rep is not None and len(rep) > 0 and "coordinates" in rep.dtype.names:
            # Mean fork position as a fraction of the replichore. Clamp to
            # [0, 0.95]: forks travel oriC→terC (coordinate ≤ REPLICHORE_BP),
            # so near division a fork can reach ~terC; cap just under 1.0 so the
            # theta-bubble mapping never over-stretches to a degenerate terminus.
            fork_fraction = min(0.95, float(np.mean(np.abs(rep["coordinates"]))) / REPLICHORE_BP)
        else:
            log.info("chromosome_state_from_live: active_replisome present but empty "
                     "(or missing 'coordinates'); fork_fraction defaults to 0.0")
    return n_chromosomes, fork_fraction


def rnaps_from_live(active_rnap, full_chromosome=None, chromosome_domain=None) -> list:
    """``rnaps`` list for :class:`pbg_parsimony.Chromosome` from LIVE arrays.

    Each entry is ``{"coordinates": int, "domain_index": int, "is_forward":
    bool, "chromosome_index": int, "is_daughter": bool}`` — the first three
    are what ``pbg_parsimony.Chromosome.rnaps`` documents; the trailing two
    are additive (the parsimony recipe/engine accept and pass them through;
    see ``parsimony`` ``RawRnap`` — ``#[serde(default)]`` on both) and route
    each RNAP onto the correct chromosome/daughter copy when replication data
    is available.

    ``active_rnap`` is the live ``['unique']['active_RNAP']`` store.
    ``full_chromosome``/``chromosome_domain`` (optional) supply the domain
    lineage :func:`classify_domains` needs to resolve ``chromosome_index``/
    ``is_daughter``; without them every RNAP defaults to chromosome 0,
    is_daughter=False (i.e. treated as living on the (sole) primary
    chromosome — never fabricated as a daughter).
    """
    rn = _active_rows(active_rnap)
    if rn is None or len(rn) == 0:
        return []

    coords = rn["coordinates"]
    domains = rn["domain_index"]
    is_forward = (rn["is_forward"] if "is_forward" in rn.dtype.names
                 else np.ones(len(rn), dtype=bool))

    chrom_idx = np.zeros(len(rn), dtype=np.int32)
    is_dau = np.zeros(len(rn), dtype=bool)
    fc = _active_rows(full_chromosome) if full_chromosome is not None else None
    if fc is not None and len(fc) > 0 and "domain_index" in fc.dtype.names:
        fc_domains = [int(x) for x in fc["domain_index"]]
        domain_children: dict = {}
        cd = _active_rows(chromosome_domain) if chromosome_domain is not None else None
        if cd is not None and {"domain_index", "child_domains"}.issubset(set(cd.dtype.names)):
            for entry in cd:
                parent = int(entry["domain_index"])
                kids = [int(k) for k in entry["child_domains"] if int(k) >= 0]
                if kids:
                    domain_children[parent] = kids
        chrom_idx, is_dau = classify_domains(domain_children, fc_domains, domains)

    return [
        {"coordinates": int(c), "domain_index": int(d), "is_forward": bool(f),
         "chromosome_index": int(ci), "is_daughter": bool(isd)}
        for c, d, f, ci, isd in zip(coords, domains, is_forward, chrom_idx, is_dau)
    ]


def pack_from_state(out_dir, name, counts, volume_fl, locations=None, *, top_n=40, scale=0.3,
                    proxy_lod=2, relax=False, cache_dir="out/cache", relax_params=None,
                    envelope=True, periplasm_gap_A=250.0,
                    rnaps=None, n_chromosomes=1, fork_fraction=0.0):
    """Pack a 3D structural model directly from in-memory ``counts``/``volume_fl``
    (no ``load_state`` round-trip). When ``envelope`` is true (default), builds
    a gram-negative two-membrane envelope (inner membrane + periplasm + outer
    membrane) sized from ``volume_fl``, and routes each ingredient to its real
    compartment via ``locations`` (``{ecocyc_id: parsimony compartment}``, from
    :func:`bulk_to_locations`). ``rnaps``/``n_chromosomes``/``fork_fraction``
    carry the live RNAP loci + replication state (see
    :func:`rnaps_from_live`/:func:`chromosome_state_from_live`) into the
    Chromosome recipe. Returns build_pack's result dict."""
    ingredients = select_ingredients(counts, locations, top_n=top_n)
    if relax:
        ingredients = relax_ingredients(ingredients, cache_dir=cache_dir,
                                        relax_cfg=relax_params or {})
    outer = Capsule.from_volume_fl(volume_fl)
    env = None
    if envelope:
        gap = min(periplasm_gap_A, outer.radius * 0.4, outer.half_len * 0.4)
        inner = Capsule(half_len=max(1.0, outer.half_len - gap),
                        radius=max(1.0, outer.radius - gap))
        env = {"inner": inner, "outer": outer}
    # NOTE: n_chromosomes>=2 or fork_fraction>0 routes the chromosome through
    # parsimony's theta/replicated (structured) layout instead of the diffuse
    # self-avoiding-walk nucleoid (see parsimony-core/src/placer.rs
    # place_chromosome: the diffuse walk is used only when
    # n_chromosomes==1 and fork_fraction<=0.0). That's a known, accepted
    # limitation for now — the unreplicated case (the common case right after
    # birth) stays diffuse; a replicating/dividing cell's nucleoid renders as
    # the structured theta form.
    chromosome = Chromosome(
        beads=90000, spacing=22.0, bead_radius=10.0,
        genome_csv=str(DATA / "ecoli_k12_genes.csv"),
        segment=StructureRef("pdb", "1BNA"),
        # No supercoil → a diffuse, space-filling self-avoiding walk (generate_fiber)
        # confined to the cytoplasm: the nucleoid spreads through the cell interior
        # rather than folding into a coherent plectoneme rosette.
        supercoil=None,
        n_chromosomes=n_chromosomes, fork_fraction=fork_fraction,
        fork_marker="replisome", oric_marker="oriC", ter_marker="terminus",
        rnaps=rnaps or [], rnap_marker="rna_polymerase")
    return build_pack(ingredients, outer, chromosome,
                      out_dir=out_dir, name=name, scale=scale, proxy_lod=proxy_lod,
                      envelope=env)


def build_model(out_dir="out/ecoli3d", *, name="ecoli_3d", top_n=40, scale=0.3,
                state_source="snapshot", proxy_lod=2) -> dict:
    """Build the 3D E. coli pack from a v2ecoli state. Returns build_pack's result."""
    return pack_from_state(out_dir, name, *load_state(state_source),
                           top_n=top_n, scale=scale, proxy_lod=proxy_lod)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the 3D E. coli structural model.")
    ap.add_argument("--out", default="out/ecoli3d")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--state", choices=["snapshot", "live"], default="snapshot")
    a = ap.parse_args()
    res = build_model(a.out, top_n=a.top_n, scale=a.scale, state_source=a.state)
    print(f"packed {res['n_placed']} placements · {res['ingredients']} ingredients → {res['pack_path']}")
