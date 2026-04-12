"""
Carbon-atom counts for the external exchange molecules E. coli's FBA
can move across the cell boundary.

Used by:
  * the carbon-budget listener to weight per-step exchange counts into
    mmol-of-carbon in / mmol-of-carbon out;
  * the nutrient-growth report to visualise whether the cell is actually
    carbon-balanced (imports == biomass + secreted + decarboxylated).

Counts are from the compound's molecular formula. Keys are the
BioCyc / EcoCyc identifiers used by the wholecell model, without the
compartment suffix ("GLC[p]" → "GLC").

Unknowns (CPD- codes without obvious formulas, trace pathway
intermediates) default to 0 via a ``.get(mol, 0)`` lookup — the
budget is an overestimate of what's known, not a claim about
everything.
"""

# Amino acids (C counts from canonical side-chains).
_AA_CARBONS = {
    "L-ALPHA-ALANINE": 3,   # ALA
    "ARG": 6,
    "ASN": 4,
    "L-ASPARTATE": 4,       # ASP
    "CYS": 3,
    "GLN": 5,
    "GLT": 5,               # GLU
    "GLY": 2,
    "HIS": 6,
    "ILE": 6,
    "LEU": 6,
    "LYS": 6,
    "MET": 5,
    "PHE": 9,
    "PRO": 5,
    "SER": 3,
    "THR": 4,
    "TRP": 11,
    "TYR": 9,
    "VAL": 5,
    "L-SELENOCYSTEINE": 3,
    "D-ALANINE": 3,
}

_CORE = {
    # Carbon sources
    "GLC": 6,               # glucose
    "GLC-D-LACTONE": 6,
    "GLYCEROL": 3,
    "ACET": 2,              # acetate
    "FORMATE": 1,
    "ETOH": 2,              # ethanol
    "D-LACTATE": 3,
    "SUC": 4,               # succinate
    "MAL": 4,               # malate
    "FUM": 4,               # fumarate
    "ARABINOSE": 5,
    "BUTANAL": 4,
    "BETAINE": 5,
    # One-carbon
    "CARBON-DIOXIDE": 1,
    "CARBON-MONOXIDE": 1,
    "UREA": 1,
    "METOH": 1,             # methanol
    # Nucleobases / nucleosides
    "CYTIDINE": 9,
    "CYTOSINE": 4,
    "THYMINE": 5,
    "URACIL": 4,
    "HYPOXANTHINE": 5,
    "XANTHINE": 5,
    # Other aromatic / ring
    "INDOLE": 8,
    "IMIDAZOLE-PYRUVATE": 6,
    # Small N / inorganic (zero carbon)
    "AMMONIUM": 0,
    "NITRATE": 0,
    "NITRITE": 0,
    "SULFATE": 0,
    "Pi": 0,
    "WATER": 0,
    "OXYGEN-MOLECULE": 0,
    "HYDROGEN-MOLECULE": 0,
    "PROTON": 0,
    # Ions (no carbon)
    "K+": 0, "MG+2": 0, "NA+": 0, "CA+2": 0, "CL-": 0,
    "FE+2": 0, "FE+3": 0, "MN+2": 0, "ZN+2": 0, "NI+2": 0, "CO+2": 0,
    "TUNGSTATE": 0,
    "4FE-4S": 0,
    # Selected cofactors / pathway intermediates
    "S-ADENOSYLMETHIONINE": 15,
    "S-ADENOSYL-4-METHYLTHIO-2-OXOBUTANOATE": 14,
    "CH33ADO": 11,          # 5'-methylthioadenosine
    "5-Deoxy-D-Ribofuranose": 5,
    "GLYCOLALDEHYDE": 2,
    "GLYCOLLATE": 2,
    "MI-PENTAKISPHOSPHATE": 6,  # inositol pentaphosphate
    "UNDECAPRENYL-DIPHOSPHATE": 55,
}


CARBON_COUNTS: dict[str, int] = {**_CORE, **_AA_CARBONS}


def carbon_of(molecule: str) -> int:
    """Carbon count for a boundary molecule id, with or without
    ``[compartment]`` suffix. Returns 0 for unknowns."""
    if "[" in molecule:
        molecule = molecule.split("[", 1)[0]
    return CARBON_COUNTS.get(molecule, 0)
