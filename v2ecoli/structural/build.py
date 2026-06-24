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

# ── Chromosome geometry constants (E. coli K-12 MG1655) ──────────────────────
# Genome length and the per-replichore length the replication forks travel
# (oriC→terC), from v2ecoli's replication reconstruction.
GENOME_BP = 4_641_652
REPLICHORE_BP = GENOME_BP // 2  # 2,320,826
# Coarse-grained bead count representing one full (unreplicated) genome. The
# theta builder adds ~fork_fraction×GENOME_BEADS more via the sister strand, so
# total DNA contour scales as n_chromosomes×(1+fork_fraction) — i.e. with the
# real total bp / dna_mass of the state.
GENOME_BEADS = 34_000


def chromosome_state(state_source="snapshot"):
    """Return ``(n_chromosomes, fork_fraction)`` for the given state.

    ``n_chromosomes`` is the ``full_chromosome`` count; ``fork_fraction`` is the
    mean replication-fork position as a fraction of the replichore length
    (0 = unreplicated, 0.5 = forks halfway to terC). Read from the saved state
    npz (keys ``n_chromosomes`` / ``fork_fraction``), with sane defaults when a
    state predates the chromosome fields.
    """
    if state_source == "division":
        npz = DATA / "v2ecoli_state_division.npz"
        default = (2, 728_151 / REPLICHORE_BP)
    else:
        npz = DATA / "v2ecoli_state.npz"
        default = (1, 1_040_161 / REPLICHORE_BP)
    try:
        st = np.load(npz)
        n = int(st["n_chromosomes"]) if "n_chromosomes" in st else default[0]
        f = float(st["fork_fraction"]) if "fork_fraction" in st else default[1]
        return n, f
    except Exception:
        return default


def division_progress(state_source="snapshot"):
    """Fraction through the D-period (replication termination → division), 0..1.

    The cell's progress toward cytokinesis, read from the state npz key
    ``division_progress`` (computed from the source run: a newborn cell ≈ 0; the
    last frame before division ≈ 1). Drives the septum constriction depth.
    """
    npz = (DATA / "v2ecoli_state_division.npz") if state_source == "division" \
        else (DATA / "v2ecoli_state.npz")
    default = 1.0 if state_source == "division" else 0.0
    try:
        st = np.load(npz)
        return float(st["division_progress"]) if "division_progress" in st else default
    except Exception:
        return default


def septum_from_progress(progress, onset=0.4, max_depth=0.7):
    """Septum constriction depth (0..max_depth) from division progress.

    The FtsZ-ring constriction begins partway through the D-period (``onset``)
    and deepens to ``max_depth`` at division — capped below 1.0 so the envelope
    keeps a visible neck rather than fully pinching into two cells.
    """
    p = max(0.0, min(1.0, float(progress)))
    if p <= onset:
        return 0.0
    return max_depth * (p - onset) / (1.0 - onset)


# Category → display colour (RGB 0–1).
CATEGORY_COLOR = {
    "Translation": (0.95, 0.55, 0.25), "Transcription": (0.35, 0.6, 0.95),
    "Nucleoid": (0.85, 0.75, 0.45), "Metabolism": (0.45, 0.8, 0.5),
    "Protein folding": (0.95, 0.85, 0.3), "Envelope": (0.8, 0.55, 0.85),
    "Regulation": (0.9, 0.4, 0.5), "Motility": (0.25, 0.78, 0.72),
    "Replication": (1.0, 0.35, 0.1), "Division": (0.2, 0.85, 0.9),
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

# FtsZ Z-ring: how many FtsZ to lay around the septum circle (a visible cyan band
# at midcell; the real ring is denser but this reads cleanly at whole-cell scale).
FTSZ_RING_COUNT = 160

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
    # complex_id,   display_name,   category,    arrangement,      region
    ("CPLX0-7452", "flagellum", "Motility", "motor+filament", "surface"),
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


def _flagellum_tube_obj(out_path, *, contour_len=22000.0, helix_radius=1500.0,
                        helix_pitch=12000.0, base_radius=95.0, tip_radius=55.0,
                        n_ring=12, ds=70.0):
    """Write a flagellum DIRECTLY as a watertight OBJ tube swept along a helix (a
    generalised cylinder), centred at its centroid. No PDB/mesher round-trip — that
    path fragmented into thousands of pieces AND the PDB writer overflowed its
    8-col coordinate fields past 9999 Å, mis-scaling the mesh to ~900 µm. Here the
    coordinates are exact. Radius tapers base→tip; helix axis = +z (so a placement
    that rotates +z onto the outward normal trails the corkscrew off the pole)."""
    Lturn = float(np.hypot(2 * np.pi * helix_radius, helix_pitch))
    rings, s = [], 0.0
    while s <= contour_len:
        f = s / contour_len
        th = 2 * np.pi * s / Lturn
        Reff = helix_radius * min(1.0, s / Lturn)              # ramp out over the 1st turn (hook)
        c = np.array([Reff * np.cos(th), Reff * np.sin(th), helix_pitch * th / (2 * np.pi)])
        T = np.array([-2 * np.pi * helix_radius * np.sin(th),
                      2 * np.pi * helix_radius * np.cos(th), helix_pitch]); T /= np.linalg.norm(T)
        N = np.array([np.cos(th), np.sin(th), 0.0]); N -= T * np.dot(N, T); N /= (np.linalg.norm(N) or 1.0)
        B = np.cross(T, N)
        rings.append((c, N, B, base_radius * (1 - f) + tip_radius * f))
        s += ds
    verts = []
    for (c, N, B, r) in rings:
        for k in range(n_ring):
            a = 2 * np.pi * k / n_ring
            verts.append(c + r * (np.cos(a) * N + np.sin(a) * B))
    verts = np.array(verts); verts -= verts.mean(0)            # centre → A-offset logic works
    faces = []
    for i in range(len(rings) - 1):
        for k in range(n_ring):
            a = i * n_ring + k + 1; b = i * n_ring + (k + 1) % n_ring + 1
            cc = (i + 1) * n_ring + (k + 1) % n_ring + 1; dd = (i + 1) * n_ring + k + 1
            faces.append((a, b, cc)); faces.append((a, cc, dd))
    with open(out_path, "w") as fo:
        for p in verts:
            fo.write(f"v {p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\n")
        for (a, b, c) in faces:
            fo.write(f"f {a} {b} {c}\n")


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
        # "division" = pre-division (max-mass) state extracted from a cached
        # two-generation run (end of generation 1); "snapshot" = birth state.
        fname = ("v2ecoli_state_division.npz" if state_source == "division"
                 else "v2ecoli_state.npz")
        st = np.load(DATA / fname)
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


def _pack_cluster(radii):
    """Greedy compact sphere-packing: place each subunit (largest first) touching
    the growing cluster as close to the centre as possible → a globular assembly
    (subunits in contact), not subunits floating on a sparse grid. ``radii`` are
    the subunit bounding radii; returns centred [N,3] centre positions (Å)."""
    n = len(radii)
    order = sorted(range(n), key=lambda i: -radii[i])
    pos = np.zeros((n, 3))
    placed = []
    ga = np.pi * (3 - np.sqrt(5))
    C = 128
    dirs = []
    for c in range(C):
        y = 1 - 2 * (c + 0.5) / C; rxy = max(0.0, 1 - y * y) ** 0.5; phi = c * ga
        dirs.append(np.array([rxy * np.cos(phi), y, rxy * np.sin(phi)]))
    for i in order:
        ri = radii[i]
        if not placed:
            placed.append(i); continue                  # first subunit at origin
        best_p, best_score = None, 1e30
        for dvec in dirs:
            t = 0.0
            for j in placed:
                pj = pos[j]; R = ri + radii[j]
                dp = float(np.dot(dvec, pj))
                disc = dp * dp - float(np.dot(pj, pj)) + R * R
                if disc <= 0:
                    continue                              # this ray misses subunit j
                tplus = dp + np.sqrt(disc)                # exit point past subunit j
                if tplus > t:
                    t = tplus
            if t * t < best_score:                        # closest to centre = most compact
                best_score = t * t; best_p = t * dvec
        pos[i] = best_p; placed.append(i)
    pos -= pos.mean(0)
    return pos


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
    centers = _pack_cluster([u[2] for u in units])     # compact globular packing
    lines = []
    serial = 1
    for (xyz, elems, _), off in zip(units, centers):
        ox, oy, oz = float(off[0]), float(off[1]), float(off[2])
        for (x, y, z), e in zip(xyz, elems):
            # %8.3f overflows past ±9999 Å (mis-parses on read); compact clusters
            # stay well inside that, but clamp the format defensively.
            lines.append(f"ATOM  {serial % 100000:5d}  CA  ALA A{serial % 9999:4d}    "
                         f"{x + ox:8.2f}{y + oy:8.2f}{z + oz:8.2f}  1.00  0.00          {e:>2}")
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
            # The flagellum is NOT handed to the packer at all: its 19000 Å tube
            # mesh makes the octree proxy voxeliser explode (it hung for hours).
            # It's meshed + injected entirely post-pack (_inject_flagellum) as a
            # rear-pole tuft at the true bulk count. Just reserve the id here so
            # the generic stoichiometry-blob path below skips it.
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


def _cluster_complex_at_pole(pack_path, mesh_dir, name, mesh_stem, cone_deg=55.0):
    """Gather a lopsided surface complex into a tuft on one capsule cap.

    The flagellum (compact motor + long whip) packed across the whole surface
    reads as a pincushion of spikes, and anchoring by the mesh centroid buries the
    motor. Instead place all copies on the rear (−x) cap, each with its motor end
    on the envelope and its whip pointing outward — a flagellar tuft 'out the
    back'. Format-agnostic (object or array8 placements)."""
    pack_path = Path(pack_path)
    d = json.loads(pack_path.read_text())
    ing = next((g for g in d["ingredients"] if g["name"] == name), None)
    if ing is None:
        return
    fid = ing["id"]
    finest = sorted(Path(mesh_dir).glob(f"{mesh_stem}.lod*.obj"))[-1]
    vs = np.array([[float(x) for x in line.split()[1:4]]
                   for line in open(finest) if line.startswith("v ")])
    zc = vs[:, 2] - vs[:, 2].mean()
    # The composite is built motor-at-−z, filament-along-+z, so the motor end is
    # the most-negative-z point; A is its distance from the centroid.
    A = float(-zc.min())
    cap = next(c for c in d["compartments"] if c.get("kind") == "capsule")
    a = np.array(cap["a"], float); radius = float(cap["radius"])
    pole = a / np.linalg.norm(a)                    # outward axis of the −x cap
    arr8 = d.get("placement_format") == "array8"
    fl = [p for p in d["placements"] if (p[0] if arr8 else p["ingredient"]) == fid]
    n = len(fl); ga = np.pi * (3 - np.sqrt(5))
    cos_half = np.cos(np.deg2rad(cone_deg))
    u, v = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])  # ⟂ to pole (−x)
    for i, p in enumerate(fl):
        ct = 1 - (i + 0.5) / n * (1 - cos_half); st = np.sqrt(max(0.0, 1 - ct * ct))
        phi = i * ga
        d_ = ct * pole + st * (np.cos(phi) * u + np.sin(phi) * v)   # dir in cone around pole
        d_ /= np.linalg.norm(d_)
        pos = a + (radius + A) * d_                                 # centroid; motor end → envelope
        # quaternion rotating mesh +z (filament) onto d_
        zc_ = float(d_[2])
        if zc_ > 0.999999:
            q = [1.0, 0.0, 0.0, 0.0]
        elif zc_ < -0.999999:
            q = [0.0, 1.0, 0.0, 0.0]
        else:
            ax = np.array([-d_[1], d_[0], 0.0]); ax /= np.linalg.norm(ax)
            half = np.arccos(zc_) / 2; s = np.sin(half)
            q = [float(np.cos(half)), float(ax[0] * s), float(ax[1] * s), float(ax[2] * s)]
        if arr8:
            p[1], p[2], p[3] = round(float(pos[0]), 1), round(float(pos[1]), 1), round(float(pos[2]), 1)
            p[4], p[5], p[6], p[7] = [round(c, 4) for c in q]
        else:
            p["position"] = [float(pos[0]), float(pos[1]), float(pos[2])]; p["rotation"] = q
    pack_path.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
    return {"placed": n, "offset": A}


def _set_sidecar_count(sidecar_path, name, count):
    side = json.loads(Path(sidecar_path).read_text())
    ings = side.get("ingredients", side)
    if name in ings:
        ings[name]["count"] = int(count)
    Path(sidecar_path).write_text(json.dumps(side, indent=1))


def _quat_z_to(dvec):
    """Quaternion [w,x,y,z] rotating the mesh +z axis onto unit vector ``dvec``."""
    z = np.array([0.0, 0.0, 1.0]); c = float(np.dot(z, dvec))
    if c > 0.999999:
        return [1.0, 0.0, 0.0, 0.0]
    if c < -0.999999:
        return [0.0, 1.0, 0.0, 0.0]
    ax = np.cross(z, dvec); ax /= np.linalg.norm(ax)
    half = np.arccos(c) / 2.0; s = np.sin(half)
    return [float(np.cos(half)), float(ax[0] * s), float(ax[1] * s), float(ax[2] * s)]


def _emit(fid, pos, q, arr8):
    if arr8:
        return [fid, round(float(pos[0])), round(float(pos[1])), round(float(pos[2])),
                round(q[0], 3), round(q[1], 3), round(q[2], 3), round(q[3], 3)]
    return {"ingredient": fid, "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "rotation": [float(v) for v in q]}


def _place_flagella_tuft(pack_path, sidecar_path, mesh_dir, name, mesh_stem,
                         count, half_len, radius, cone_deg=50.0):
    """Place EXACTLY ``count`` flagella as a tuft on the rear (−x) pole — motors on
    the envelope, helical whips fanning outward. REPLACES whatever the packer
    placed (flagella are a surface ingredient, so the area-limited packer
    under-places them) and sets the sidecar count to ``count``, so the rendered
    number matches the v2ecoli bulk count."""
    pack_path = Path(pack_path); d = json.loads(pack_path.read_text())
    ing = next((g for g in d["ingredients"] if g["name"] == name), None)
    if ing is None or count <= 0:
        return
    fid = ing["id"]; arr8 = d.get("placement_format") == "array8"
    finest = sorted(Path(mesh_dir).glob(f"{mesh_stem}.lod*.obj"))[-1]
    vs = np.array([[float(x) for x in line.split()[1:4]]
                   for line in open(finest) if line.startswith("v ")])
    A = float(-(vs[:, 2] - vs[:, 2].mean()).min())   # motor-end offset from centroid
    a = np.array([-float(half_len), 0.0, 0.0])        # rear cap centre, axis = x
    pole = np.array([-1.0, 0.0, 0.0])
    u, v = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    ga = np.pi * (3 - np.sqrt(5)); cos_half = np.cos(np.deg2rad(cone_deg))
    newpl = []
    for i in range(int(count)):
        ct = 1 - (i + 0.5) / count * (1 - cos_half); st = np.sqrt(max(0.0, 1 - ct * ct))
        phi = i * ga
        d_ = ct * pole + st * (np.cos(phi) * u + np.sin(phi) * v); d_ /= np.linalg.norm(d_)
        pos = a + (radius + A) * d_
        newpl.append(_emit(fid, pos, _quat_z_to(d_), arr8))
    d["placements"] = [p for p in d["placements"]
                       if (p[0] if arr8 else p["ingredient"]) != fid] + newpl
    pack_path.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
    _set_sidecar_count(sidecar_path, name, count)


def _place_flagella_peritrichous(pack_path, sidecar_path, mesh_dir, name, mesh_stem,
                                 count, half_len, radius, sweep=0.5):
    """Distribute ``count`` flagella over the WHOLE capsule surface (cylinder body +
    both caps) — peritrichous, not a polar tuft — each emerging from its surface
    point and trailing outward with a mild rearward ``sweep`` (the swimming-bundle
    look). Replaces existing placements + sets the sidecar count."""
    pack_path = Path(pack_path); d = json.loads(pack_path.read_text())
    ing = next((g for g in d["ingredients"] if g["name"] == name), None)
    if ing is None or count <= 0:
        return
    fid = ing["id"]; arr8 = d.get("placement_format") == "array8"
    finest = sorted(Path(mesh_dir).glob(f"{mesh_stem}.lod*.obj"))[-1]
    vs = np.array([[float(x) for x in ln.split()[1:4]]
                   for ln in open(finest) if ln.startswith("v ")])
    A = float(-(vs[:, 2] - vs[:, 2].mean()).min())   # base→centroid offset (centred mesh)
    L = float(half_len); R = float(radius)
    cyl_frac = (2 * np.pi * R * 2 * L) / (2 * np.pi * R * 2 * L + 4 * np.pi * R * R)
    ga = np.pi * (3 - np.sqrt(5))
    newpl = []
    for i in range(int(count)):
        u = (i + 0.5) / count; phi = i * ga
        if u < cyl_frac:                                   # cylindrical body
            x = -L + 2 * L * (u / cyl_frac)
            n = np.array([0.0, np.cos(phi), np.sin(phi)])
            surf = np.array([x, R * n[1], R * n[2]])
        else:                                              # the two hemispherical caps
            uc = (u - cyl_frac) / (1 - cyl_frac)
            cap = -1.0 if uc < 0.5 else 1.0
            cz = (uc % 0.5) / 0.5; sr = np.sqrt(max(0.0, 1 - cz * cz))
            n = np.array([cap * cz, sr * np.cos(phi), sr * np.sin(phi)])
            surf = np.array([cap * L, 0.0, 0.0]) + R * n
        # Sweep toward the NEAREST pole (not a single global direction) — so on a
        # near-dividing dumbbell each daughter's flagella trail off ITS own end,
        # splitting the bundle evenly between the two daughters.
        pole = np.array([1.0 if surf[0] >= 0 else -1.0, 0.0, 0.0])
        wd = n + sweep * pole; wd /= (np.linalg.norm(wd) or 1.0)
        pos = surf + A * wd                                # base sits on the surface
        newpl.append(_emit(fid, pos, _quat_z_to(wd), arr8))
    d["placements"] = [p for p in d["placements"]
                       if (p[0] if arr8 else p["ingredient"]) != fid] + newpl
    pack_path.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
    _set_sidecar_count(sidecar_path, name, count)


def _place_septum_ring(pack_path, sidecar_path, name, ring_radius, count, x=0.0, band=420.0):
    """Place EXACTLY ``count`` copies of ``name`` in a ring at midcell (x≈0, the
    septum), radius ``ring_radius`` — the FtsZ Z-ring constricting the division
    site, oriented tangentially in a 2-row band. Replaces the packer's placements
    + sets the sidecar count."""
    pack_path = Path(pack_path); d = json.loads(pack_path.read_text())
    ing = next((g for g in d["ingredients"] if g["name"] == name), None)
    if ing is None or count <= 0:
        return
    fid = ing["id"]; arr8 = d.get("placement_format") == "array8"
    newpl = []
    count = int(count)
    ga = np.pi * (3 - np.sqrt(5))      # golden angle → even coverage of the band
    for i in range(count):
        th = i * ga
        xx = x + band * ((i + 0.5) / count - 0.5)         # spread across the band width
        pos = np.array([xx, ring_radius * np.cos(th), ring_radius * np.sin(th)])
        t = np.array([0.0, -np.sin(th), np.cos(th)])      # tangent (protofilament direction)
        newpl.append(_emit(fid, pos, _quat_z_to(t), arr8))
    d["placements"] = [p for p in d["placements"]
                       if (p[0] if arr8 else p["ingredient"]) != fid] + newpl
    pack_path.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
    _set_sidecar_count(sidecar_path, name, count)


def _mat_to_quat(R):
    """3×3 rotation matrix (orthonormal, right-handed cols) → quaternion [w,x,y,z]."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return [float(0.25 * s), float((R[2, 1] - R[1, 2]) / s),
                float((R[0, 2] - R[2, 0]) / s), float((R[1, 0] - R[0, 1]) / s)]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return [float((R[2, 1] - R[1, 2]) / s), float(0.25 * s),
                float((R[0, 1] + R[1, 0]) / s), float((R[0, 2] + R[2, 0]) / s)]
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return [float((R[0, 2] - R[2, 0]) / s), float((R[0, 1] + R[1, 0]) / s),
                float(0.25 * s), float((R[1, 2] + R[2, 1]) / s)]
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return [float((R[1, 0] - R[0, 1]) / s), float((R[0, 2] + R[2, 0]) / s),
            float((R[1, 2] + R[2, 1]) / s), float(0.25 * s)]


def _mesh_ellipsoid(verts):
    """Principal-axis ellipsoid (enclosing_radius, semi_axes, rotation[w,x,y,z])
    fitted to a vertex cloud — the anisotropic fallback proxy the viewer draws
    before a mesh's OBJ LODs stream in (so a flagellum reads as a cigar, not a
    huge ball)."""
    c = verts.mean(0); X = verts - c
    _, V = np.linalg.eigh(X.T @ X / max(1, len(X)))
    V = V[:, ::-1]                                  # eigh ascending → descending
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    proj = X @ V
    semi = [float(max(1.0, (proj[:, k].max() - proj[:, k].min()) / 2.0)) for k in range(3)]
    enclosing = float(np.sqrt((X ** 2).sum(1)).max())
    return enclosing, semi, _mat_to_quat(V)


def _inject_flagellum(pack_path, sidecar_path, out_dir, name, count, half_len, radius,
                      *, color, category, display_name):
    """Write the flagellum tube OBJ directly + inject it into the finished pack as a
    rear-pole tuft of ``count`` placements (kept entirely outside the octree packer,
    whose proxy voxeliser explodes on the long tube). Idempotent: re-running updates
    the existing flagellum ingredient + replaces its placements."""
    out_dir = Path(out_dir); mesh_dir = out_dir / "meshes"; mesh_dir.mkdir(parents=True, exist_ok=True)
    stem = "flagellum"
    _flagellum_tube_obj(mesh_dir / f"{stem}.lod0.obj", n_ring=14, ds=70.0)
    _flagellum_tube_obj(mesh_dir / f"{stem}.lod1.obj", n_ring=9, ds=150.0)
    verts = np.array([[float(x) for x in ln.split()[1:4]]
                      for ln in open(mesh_dir / f"{stem}.lod0.obj") if ln.startswith("v ")])
    enclosing, semi, ell_q = _mesh_ellipsoid(verts)
    shape = {"kind": "mesh", "enclosing_radius": enclosing,
             "ellipsoid": {"rotation": ell_q, "semi_axes": semi},
             "lods": [{"url": f"meshes/{stem}.lod0.obj", "voxel_size": 16.0},
                      {"url": f"meshes/{stem}.lod1.obj", "voxel_size": 8.0}]}
    d = json.loads(Path(pack_path).read_text())
    existing = next((g for g in d["ingredients"] if g["name"] == name), None)
    if existing is not None:
        existing["shape"] = shape; existing["color"] = [float(c) for c in color]
    else:
        new_id = max((g["id"] for g in d["ingredients"]), default=-1) + 1
        d["ingredients"].append({"id": new_id, "name": name,
                                 "color": [float(c) for c in color], "shape": shape})
    Path(pack_path).write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
    side = json.loads(Path(sidecar_path).read_text())
    ings = side.get("ingredients", side)
    ings[name] = {"display_name": display_name, "category": category, "count": int(count)}
    Path(sidecar_path).write_text(json.dumps(side, indent=1))
    _place_flagella_peritrichous(pack_path, sidecar_path, mesh_dir, name, stem, count,
                                 half_len, radius)


def _constricted_capsule_mesh(half_len, radius, depth, width=None,
                              n_axial=160, n_theta=56):
    """Triangle mesh (verts, faces) for a spherocylinder (``half_len`` cyl
    half-length, ``radius``, axis = x) with a Gaussian radius dip of fractional
    ``depth`` at midcell — a dividing cell's septum constriction. ``depth`` 0 =
    smooth rod, ~0.5 = a deep waist. Used as the cell compartment so the membrane
    pinches and the interior leaves a midcell gap for the division site."""
    L, R = float(half_len), float(radius)
    w = float(width) if width is not None else 0.5 * R
    # Drop the exact tips (radius 0) — add explicit apex points + fans instead,
    # so there are no degenerate rings.
    xs = np.linspace(-(L + R), (L + R), n_axial)[1:-1]
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    def rad(x):
        ax = abs(x)
        if ax <= L:
            return R * (1.0 - depth * np.exp(-((x / w) ** 2)))
        dx = ax - L
        return float(np.sqrt(max(0.0, R * R - dx * dx)))

    verts = []
    for x in xs:
        r = rad(float(x))
        for th in thetas:
            verts.append((float(x), float(r * np.cos(th)), float(r * np.sin(th))))
    faces = []
    n_rings = len(xs)
    for i in range(n_rings - 1):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = (i + 1) * n_theta + (j + 1) % n_theta
            d = (i + 1) * n_theta + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    # Cap the two ends with apex points fanned to the first / last rings.
    apex_lo = len(verts); verts.append((float(-(L + R)), 0.0, 0.0))
    apex_hi = len(verts); verts.append((float(L + R), 0.0, 0.0))
    base = (n_rings - 1) * n_theta
    for j in range(n_theta):
        faces.append((apex_lo, j, (j + 1) % n_theta))
        faces.append((apex_hi, base + (j + 1) % n_theta, base + j))
    return verts, faces


def compact_to_array8(pack_path):
    """Rewrite an object-format pack in place as compact ``array8`` placements.

    Each placement becomes ``[ingredient_id, x, y, z (int Å), w, qx, qy, qz
    (3 dp)]`` and ``placement_format`` is set to ``"array8"`` — the format the
    R2-hosted viewer loads (smaller files, faster load). Idempotent.
    """
    p = Path(pack_path)
    d = json.loads(p.read_text())
    if d.get("placement_format") == "array8":
        return d
    # Pack-relative mesh URLs (meshes/X.lodN.obj) so they resolve next to the
    # pack wherever it's hosted (R2), not at an absolute local path.
    for ing in d.get("ingredients", []):
        for lod in ing.get("shape", {}).get("lods", []):
            if "url" in lod:
                lod["url"] = "meshes/" + os.path.basename(lod["url"])
    out = []
    for pl in d["placements"]:
        x, y, z = pl["position"]
        w, qx, qy, qz = pl["rotation"]
        out.append([pl["ingredient"],
                    round(float(x)), round(float(y)), round(float(z)),
                    round(float(w), 3), round(float(qx), 3),
                    round(float(qy), 3), round(float(qz), 3)])
    d["placements"] = out
    d["placement_format"] = "array8"
    p.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
    return d


def build_model(out_dir="out/ecoli3d", *, name="ecoli_3d", top_n=40, scale=1.0,
                state_source="snapshot", proxy_lod=2, top_complexes=150,
                width_um=1.0, density_g_per_ml=1.1, septum_fraction=None) -> dict:
    """Build the 3D E. coli pack from a v2ecoli state. Returns build_pack's result.

    ``scale`` defaults to 1.0 (true abundance from the state — every molecule is
    placed once per real copy; large interior assemblies pack first so they reach
    their count). The committed/published pack is additionally compacted to the
    array8 placement format to stay under the 100 MB file limit."""
    counts, volume_fl = load_state(state_source)
    struct_cache = Path(out_dir) / "structures"
    ingredients = select_ingredients(counts, top_n=top_n, struct_cache=struct_cache,
                                     top_complexes=top_complexes)
    # Chromosome landmark molecules, seated by the chromosome stage at their real
    # loci (count=0 → not placed randomly, only at the forks/origins/terminus).
    # The replisome and oriC are genuine unique molecules in the cell state (their
    # counts = active_replisome / oriC counts); terC is the terminus locus.
    ingredients.append(Ingredient(
        id="replisome", count=0, structure=StructureRef("pdb", "2HPI"),
        color=(1.0, 0.35, 0.1), category="Replication", proxy_voxel_size=14.0,
        display_name="Replisome — DNA polymerase III (active_replisome, at fork)"))
    ingredients.append(Ingredient(
        id="oriC", count=0, sphere_radius=70.0,
        color=(0.2, 0.9, 0.4), category="Replication",
        display_name="oriC (origin of replication)"))
    ingredients.append(Ingredient(
        id="terminus", count=0, sphere_radius=70.0,
        color=(0.35, 0.5, 1.0), category="Replication",
        display_name="terC (replication terminus)"))
    # Cell envelope from the Shape step (Skalnik et al. 2023): fixed width +
    # density, length derived from volume — so a pre-division state yields the
    # elongated, about-to-divide capsule.
    from v2ecoli.structural.shape import shape_from_mass
    mass_fg = volume_fl * density_g_per_ml * 1000.0
    capsule = shape_from_mass(mass_fg, width_um=width_um,
                              density_g_per_ml=density_g_per_ml)["capsule"]
    # Chromosome state from the model: number of chromosomes + how far the
    # replication forks have travelled. Each chromosome is laid out as a theta
    # structure with a replication bubble pinched at two forks; DNA contour (and
    # so size/mass) scales as n_chromosomes×(1+fork_fraction) — matching the
    # state's real total DNA bp.
    n_chrom, fork_fraction = chromosome_state(state_source)
    chromosome = Chromosome(
        beads=GENOME_BEADS, spacing=135.0, bead_radius=12.0,
        genome_csv=str(DATA / "ecoli_k12_genes.csv"),
        segment=StructureRef("pdb", "1BNA"),
        supercoil={"radius": 90.0, "pitch": 130.0, "domains": 200},
        n_chromosomes=n_chrom, fork_fraction=fork_fraction,
        fork_marker="replisome", oric_marker="oriC", ter_marker="terminus")
    # Septum: a constricting pre-division cell gets a pinched-capsule envelope (the
    # membrane + interior follow it). Depth is state-driven — it tracks the cell's
    # division progress (D-period), so a newborn is a smooth rod and a near-division
    # cell has a deep waist — but capped at a ~50% medial neck (a constricted
    # dumbbell: two full-radius lobes joined by a defined septum, not a sharp pinch).
    if septum_fraction is None:
        septum_fraction = septum_from_progress(division_progress(state_source), max_depth=0.5)
    dividing = n_chrom >= 2
    # FtsZ Z-ring constricting the septum — a dividing-cell feature only. Added as a
    # curated ingredient (so it's meshed + in the sidecar); placements arranged into
    # the midcell ring post-pack.
    if dividing:
        ingredients.append(Ingredient(
            id="ftsz_ring", count=FTSZ_RING_COUNT, structure=StructureRef("alphafold", "P0A9A6"),
            region="interior", display_name="FtsZ (Z-ring at septum)", category="Division",
            color=CATEGORY_COLOR["Division"], proxy_voxel_size=8.0))
    cell_mesh = (_constricted_capsule_mesh(capsule.half_len, capsule.radius,
                                           depth=septum_fraction,
                                           width=0.28 * capsule.radius)
                 if septum_fraction > 0.0 else None)
    res = build_pack(ingredients, capsule, chromosome,
                     out_dir=out_dir, name=name, scale=scale, proxy_lod=proxy_lod,
                     cell_mesh=cell_mesh)
    # Flagella: meshed + injected entirely post-pack (kept out of the packer,
    # whose proxy voxeliser explodes on the 19000 Å tube) as a rear-pole tuft at
    # the true v2ecoli bulk count.
    fcount = int(counts.get("CPLX0-7452", 0))
    if fcount > 0:
        _inject_flagellum(res["pack_path"], res["sidecar_path"], out_dir,
                          "CPLX0-7452", fcount, capsule.half_len, capsule.radius,
                          color=CATEGORY_COLOR["Motility"], category="Motility",
                          display_name="flagellum")
    # FtsZ Z-ring at the constricted waist (radius ≈ body·(1−septum_fraction)),
    # placed at the cell's REAL FtsZ count (EG10347-MONOMER) — during division
    # most FtsZ localises to the septal ring.
    if dividing:
        _place_septum_ring(res["pack_path"], res["sidecar_path"], "ftsz_ring",
                           ring_radius=capsule.radius * (1.0 - septum_fraction),
                           count=int(counts.get("EG10347-MONOMER", FTSZ_RING_COUNT)))
    # Counts enforcement: rewrite EVERY ingredient's sidecar count to the number
    # actually placed (markers seeded at loci, flagella/FtsZ injected, surface/
    # fiber species under-placed by area/length limits) so the viewer's "copies
    # placed" is always truthful. Also reports the under-placed.
    _backfill_all_counts(res["pack_path"], res["sidecar_path"])
    return res


def _backfill_all_counts(pack_path, sidecar_path):
    """Set every sidecar ingredient's count to its ACTUAL placement count in the
    finished pack, and warn about species the packer under-placed vs requested."""
    from collections import Counter
    pack = json.loads(Path(pack_path).read_text())
    arr8 = pack.get("placement_format") == "array8"
    id_by_name = {ing["name"]: ing["id"] for ing in pack["ingredients"]}
    placed = Counter(p[0] if arr8 else p["ingredient"] for p in pack["placements"])
    side = json.loads(Path(sidecar_path).read_text())
    ings = side.get("ingredients", side)
    under = []
    for name, meta in ings.items():
        iid = id_by_name.get(name)
        actual = int(placed.get(iid, 0))
        requested = int(meta.get("count", 0))
        if requested and actual < 0.9 * requested:
            under.append((name, requested, actual))
        meta["count"] = actual
    Path(sidecar_path).write_text(json.dumps(side, indent=1))
    if under:
        under.sort(key=lambda r: r[2] - r[1])
        print(f"  counts: {len(under)} species under-placed (area/length-limited); worst:")
        for name, req, act in under[:8]:
            print(f"    {name}: requested {req}, placed {act} ({100 * act / req:.0f}%)")


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
