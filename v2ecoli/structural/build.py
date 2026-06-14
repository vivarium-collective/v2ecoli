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
import os
from pathlib import Path

import numpy as np

from pbg_parsimony import Ingredient, Capsule, Chromosome, StructureRef, build_pack
from pbg_parsimony.structures import fetch

DATA = Path(__file__).parent / "data"

# Category → display colour (RGB 0–1).
CATEGORY_COLOR = {
    "Translation": (0.95, 0.55, 0.25), "Transcription": (0.35, 0.6, 0.95),
    "Nucleoid": (0.85, 0.75, 0.45), "Metabolism": (0.45, 0.8, 0.5),
    "Protein folding": (0.95, 0.85, 0.3), "Envelope": (0.8, 0.55, 0.85),
    "Regulation": (0.9, 0.4, 0.5), "Motility": (0.25, 0.78, 0.72),
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

# Large interior assemblies packed in an early stage so they reach true abundance
# (packed alongside the small-molecule flood they saturate at a few % of count).
BIG_ASSEMBLIES = {"70S_ribosome", "groel"}

# ── assembled complexes from the bulk ───────────────────────────────────────
# v2ecoli tracks assembled complexes (CPLX*) in the bulk, but AlphaFold only
# models single chains, so a complex needs a real assembled structure. Each
# catalog entry maps a bulk complex id → an "arrangement" that resolves to a
# structure, placed at the complex's bulk count (like the curated assemblies).
#   arrangement "motor+filament": composite of a basal-body/motor PDB at the
#   base + a flagellin filament PDB repeated into a whip (built at run time).
# (Future complexes without an assembled PDB can use a stoichiometry-driven
#  blob: read subunits from complexation_reactions.tsv + pack their AlphaFolds.)
COMPLEX_CATALOG = [
    # complex_id,   display_name,                 category,    arrangement,      region
    ("CPLX0-7452", "flagellum (motor + filament)", "Motility", "motor+filament", "surface"),
]


def _parse_pdb_atoms(path):
    out = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            out.append((x, y, z, (ln[76:78].strip() or ln[12:14].strip()[:1] or "C")))
    return out


def _parse_cif_atoms(path):
    """Minimal mmCIF ``_atom_site`` loop reader (Cartn_x/y/z + type_symbol)."""
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            hdr, j = [], i + 1
            while j < len(lines) and lines[j].lstrip().startswith("_"):
                hdr.append(lines[j].strip()); j += 1
            if any(h.startswith("_atom_site.") for h in hdr):
                cx, cy, cz = (hdr.index("_atom_site.Cartn_x"),
                              hdr.index("_atom_site.Cartn_y"),
                              hdr.index("_atom_site.Cartn_z"))
                ce = hdr.index("_atom_site.type_symbol") if "_atom_site.type_symbol" in hdr else None
                out, k = [], j
                while k < len(lines) and lines[k].strip() and lines[k].strip() not in ("#", "loop_"):
                    p = lines[k].split()
                    if len(p) >= len(hdr):
                        try:
                            out.append((float(p[cx]), float(p[cy]), float(p[cz]),
                                        (p[ce] if ce is not None else "C")))
                        except ValueError:
                            pass
                    k += 1
                return out
            i = j
        else:
            i += 1
    return []


def _align_to_z(coords):
    """Rotate the structure's longest axis (PCA) onto +z, centred at origin."""
    c = coords - coords.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z); s = float(np.linalg.norm(v)); cth = float(np.dot(axis, z))
    if s < 1e-8:
        R = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - cth) / (s * s))
    return c @ R.T


def _build_flagellum_pdb(struct_cache, *, motor=("cif", "6SD5"), filament=("pdb", "1UCU"),
                         whip_len=2200.0) -> Path:
    """Assemble a composite flagellum PDB: motor/basal-body at the base (−z,
    membrane side) + a flagellin filament repeated into a whip along +z. Returns
    the composite path; the whip points along local +z so a ``surface`` ingredient
    with principal_vector (0,0,1) anchors the motor in the envelope, whip outward."""
    struct_cache = Path(struct_cache); struct_cache.mkdir(parents=True, exist_ok=True)
    out = struct_cache / "flagellum.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    fil = _parse_pdb_atoms(fetch(StructureRef(*filament), struct_cache))
    mot = (_parse_cif_atoms if motor[0] == "cif" else _parse_pdb_atoms)(
        fetch(StructureRef(*motor), struct_cache))
    fz = _align_to_z(np.array([(x, y, z) for x, y, z, _ in fil])); fe = [e for *_, e in fil]
    mz = _align_to_z(np.array([(x, y, z) for x, y, z, _ in mot])); me = [e for *_, e in mot]
    seg = float(fz[:, 2].max() - fz[:, 2].min()) or 1.0
    nrep = max(3, int(round(whip_len / seg)))
    atoms = []
    mb = mz.copy(); mb[:, 2] -= mz[:, 2].max()                 # motor top at z=0 (base below)
    atoms += [(x, y, z, e) for (x, y, z), e in zip(mb, me)]
    f0 = fz.copy(); f0[:, 2] -= fz[:, 2].min()                  # filament base at z=0
    for r in range(nrep):
        atoms += [(x, y, z + r * seg, e) for (x, y, z), e in zip(f0, fe)]
    with open(out, "w") as f:
        for i, (x, y, z, e) in enumerate(atoms, 1):
            f.write(f"ATOM  {i % 100000:5d}  CA  ALA A{i % 9999:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2}\n")
        f.write("END\n")
    return out


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
    if any(k in n for k in ("flagell", "motil", "chemotax", "flagellar hook", "flagellar motor")):
        return "Motility"
    if any(k in n for k in ("regulator", "repressor", "activator", "transcriptional dual")):
        return "Regulation"
    if any(k in n for k in ("dna-binding", "dna gyrase", "dna polymerase", "topoisomerase",
                            "histone-like", "nucleoid", "hu-", "h-ns", "recombinase", "replicat")):
        return "Nucleoid"
    if any(k in n for k in ("outer membrane", "periplasm", "lipoprotein", "membrane", "porin",
                            "fimbri", "pilus", "flagell", "secret", "efflux", "transporter", " abc ")):
        return "Envelope"
    return "Metabolism"


def load_state(state_source="snapshot", advance_s=2.0, seed=0):
    """Return ``(counts, volume_fl)``: counts is ``{ecocyc_id: count}`` (compartment
    tags stripped, summed); volume in fL."""
    if state_source == "live":
        import v2ecoli
        comp = v2ecoli.build_composite("baseline", seed=seed, cache_dir="out/cache")
        comp.run(advance_s)
        cell = comp.state.get("agents", {}).get("0", comp.state)
        bulk = cell["bulk"]
        vol = cell["listeners"]["mass"]["volume"]
        volume_fl = float(getattr(vol, "magnitude", vol))
        ids = [str(x) for x in bulk["id"]]
        cnts = list(bulk["count"])
    else:
        st = np.load(DATA / "v2ecoli_state.npz")
        ids = [str(x) for x in st["ids"]]
        cnts = list(st["counts"])
        volume_fl = float(st["volume"])
    counts = {}
    for idt, c in zip(ids, cnts):
        base = idt[:-1].rsplit("[", 1)[0] if idt.endswith("]") and "[" in idt else idt
        counts[base] = counts.get(base, 0) + int(c)
    return counts, volume_fl


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


# ── generic complexes: assemble from subunit stoichiometry + AlphaFold monomers ─
def _complexation():
    """Return ``({product_id: {subunit_id: count}}, {product_id: name})`` parsed
    from complexation_reactions.tsv. The stoichiometry dict has the product at
    +1 and each subunit at a negative coeff (abs = copies) or ``null`` (= 1)."""
    rxn, names = {}, {}
    for r in _load_tsv("complexation_reactions.tsv"):
        vals = list(r.values())
        try:
            stoich = json.loads(vals[1])
        except Exception:
            continue
        prod = next((k for k, v in stoich.items() if isinstance(v, (int, float)) and v > 0), None)
        if not prod:
            continue
        subs = {}
        for k, v in stoich.items():
            if k == prod:
                continue
            subs[k] = 1 if v is None else int(abs(v))
        rxn[prod] = subs
        nm = vals[2] if len(vals) > 2 else ""
        names[prod] = "" if nm in ("null", None) else nm
    return rxn, names


def _expand_monomers(cid, rxn, prot, depth=0, acc=None):
    """Recursively expand a complex to its monomer composition {monomer: count}."""
    acc = acc if acc is not None else {}
    if depth > 8:
        return acc
    for sub, n in rxn.get(cid, {}).items():
        if sub in prot:                         # a monomer
            acc[sub] = acc.get(sub, 0) + n
        elif sub in rxn:                        # a sub-complex → recurse n times
            for _ in range(n):
                _expand_monomers(sub, rxn, prot, depth + 1, acc)
        # else: tRNA/RNA/ion subunit with no monomer structure → skipped
    return acc


def _cluster_offsets(n, spacing):
    """n compact 3D-grid offsets (Å), nearest-origin-first, for clustering subunits."""
    pts, r = [], 0
    while len(pts) < n:
        r += 1
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                for z in range(-r, r + 1):
                    if max(abs(x), abs(y), abs(z)) == r:
                        pts.append((x, y, z))
    pts.sort(key=lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    return [(x * spacing, y * spacing, z * spacing) for x, y, z in pts[:n]]


def _build_complex_blob(cid, monomers, struct_cache, genes, umap):
    """Assemble a composite PDB clustering each subunit's AlphaFold by stoichiometry.
    Returns the path, or None if no subunit structures resolve."""
    struct_cache = Path(struct_cache)
    out = struct_cache / f"cplx_{cid.replace('/', '_')}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    units = []  # (centered_atoms[N,3], elems, radius)
    for mono, n in monomers.items():
        acc = _uniprot(mono, genes, umap)
        if not acc:
            continue
        try:
            atoms = _parse_pdb_atoms(fetch(StructureRef("alphafold", acc), struct_cache))
        except Exception:
            continue
        if not atoms:
            continue
        xyz = np.array([(x, y, z) for x, y, z, _ in atoms]); xyz -= xyz.mean(0)
        rad = float(np.sqrt((xyz ** 2).sum(1)).max())
        elems = [e for *_, e in atoms]
        for _ in range(n):
            units.append((xyz, elems, rad))
    if not units:
        return None
    spacing = 2.0 * max(u[2] for u in units)
    offsets = _cluster_offsets(len(units), spacing)
    lines = []
    serial = 1
    for (xyz, elems, _), off in zip(units, offsets):
        ox, oy, oz = off
        for (x, y, z), e in zip(xyz, elems):
            lines.append(f"ATOM  {serial % 100000:5d}  CA  ALA A{serial % 9999:4d}    "
                         f"{x + ox:8.3f}{y + oy:8.3f}{z + oz:8.3f}  1.00  0.00          {e:>2}")
            serial += 1
    out.write_text("\n".join(lines) + "\nEND\n")
    return out


def select_ingredients(counts, *, top_n=40, lipid_count=40000, struct_cache=None,
                       top_complexes=0):
    """Curated assemblies + assembled complexes from the bulk + the top-N
    most-abundant protein monomers (AlphaFold, skipping individual ribosomal
    proteins) + a membrane lipid. Returns a list of :class:`pbg_parsimony.Ingredient`
    (counts are pre-scale; build_pack scales). ``struct_cache`` is where composite
    complex structures are assembled (defaults to a temp dir)."""
    import tempfile
    prot, genes, umap = _proteins(), _genes(), _uniprot_map()
    struct_cache = Path(struct_cache) if struct_cache else Path(tempfile.mkdtemp())
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
        ingredients.append(Ingredient(
            id=key, count=max(1, cnt), structure=ref, region=region,
            display_name=DISPLAY.get(key, prot.get(key, key)), category=cat,
            color=CATEGORY_COLOR[cat],
            proxy_voxel_size=12.0 if isinstance(struct, tuple) else None,
            # Big interior assemblies pack first (before small molecules fragment
            # the space) so they reach true abundance, not ~3% of it.
            pack_first=(key in BIG_ASSEMBLIES)))
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
        # compartment → region via the proteins.tsv computational compartment
        region = "interior"  # refined below if membrane-y by category
        if cat in ("Envelope", "Motility"):
            region = "surface"
        ingredients.append(Ingredient(
            id=mid, count=c, structure=StructureRef("alphafold", acc), region=region,
            display_name=nm, category=cat, color=CATEGORY_COLOR[cat]))
        added += 1

    # Assembled complexes from the bulk (placed at their bulk count).
    for cid, disp, cat, arrangement, region in COMPLEX_CATALOG:
        cnt = counts.get(cid, 0)
        if cnt <= 0:
            continue
        if arrangement == "motor+filament":
            pdb = _build_flagellum_pdb(struct_cache)
            ingredients.append(Ingredient(
                id=cid, count=cnt, structure=StructureRef("file", str(pdb)),
                region=region, display_name=disp, category=cat,
                color=CATEGORY_COLOR[cat], principal_vector=(0, 0, 1),
                proxy_voxel_size=14.0))
            already.add(cid)
        # other arrangements (single PDB, stoichiometry blob) added here later

    # Generic complexes from the bulk: the top-N most abundant CPLX* assembled
    # from their subunit stoichiometry (complexation_reactions.tsv) + AlphaFold
    # monomers, clustered into a blob. Free monomers stay too (the bulk count is
    # the free pool — assembled and free coexist, no double-count).
    if top_complexes > 0:
        rxn, cnames = _complexation()
        cand = sorted(((cid, c) for cid, c in counts.items()
                       if c > 0 and cid in rxn and cid not in already
                       and cid not in prot),
                      key=lambda kv: -kv[1])
        n_added = n_skipped = 0
        for cid, c in cand:
            if n_added >= top_complexes:
                break
            monos = _expand_monomers(cid, rxn, prot)
            if not monos:
                n_skipped += 1; continue
            blob = _build_complex_blob(cid, monos, struct_cache, genes, umap)
            if blob is None:
                n_skipped += 1; continue
            nm = cnames.get(cid) or cid
            cat = categorize(nm)
            region = "surface" if cat in ("Envelope", "Motility") else "interior"
            ingredients.append(Ingredient(
                id=cid, count=c, structure=StructureRef("file", str(blob)),
                region=region, display_name=nm, category=cat,
                color=CATEGORY_COLOR[cat], proxy_voxel_size=12.0))
            already.add(cid)
            n_added += 1
        print(f"  complexes: added {n_added}, skipped {n_skipped} (no resolvable subunit structures)")

    ingredients.append(Ingredient(
        id="lipid", count=lipid_count, sphere_radius=12.0, region="surface",
        display_name="Membrane phospholipid", category="Envelope",
        color=(0.75, 0.78, 0.85), principal_vector=(0, 0, 1)))
    return ingredients


def build_model(out_dir="out/ecoli3d", *, name="ecoli_3d", top_n=40, scale=1.0,
                state_source="snapshot", proxy_lod=2, top_complexes=150) -> dict:
    """Build the 3D E. coli pack from a v2ecoli state. Returns build_pack's result.

    ``scale`` defaults to 1.0 (true abundance from the state — every molecule is
    placed once per real copy; large interior assemblies pack first so they reach
    their count). The committed/published pack is additionally compacted to the
    array8 placement format to stay under the 100 MB file limit."""
    counts, volume_fl = load_state(state_source)
    struct_cache = Path(out_dir) / "structures"
    ingredients = select_ingredients(counts, top_n=top_n, struct_cache=struct_cache,
                                     top_complexes=top_complexes)
    capsule = Capsule.from_volume_fl(volume_fl)
    chromosome = Chromosome(
        beads=34000, spacing=135.0, bead_radius=12.0,
        genome_csv=str(DATA / "ecoli_k12_genes.csv"),
        segment=StructureRef("pdb", "1BNA"),
        supercoil={"radius": 90.0, "pitch": 130.0, "domains": 200})
    return build_pack(ingredients, capsule, chromosome,
                      out_dir=out_dir, name=name, scale=scale, proxy_lod=proxy_lod)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the 3D E. coli structural model.")
    ap.add_argument("--out", default="out/ecoli3d")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--state", choices=["snapshot", "live"], default="snapshot")
    a = ap.parse_args()
    res = build_model(a.out, top_n=a.top_n, scale=a.scale, state_source=a.state)
    print(f"packed {res['n_placed']} placements · {res['ingredients']} ingredients → {res['pack_path']}")
