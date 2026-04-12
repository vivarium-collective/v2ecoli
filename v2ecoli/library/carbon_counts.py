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


# Molecular weights (g/mol) for mass-balance accounting. Covers the
# molecules that actually carry mass through boundary exchanges in a
# minimal+glucose sim. Missing entries default to 0 — the balance check
# becomes an under-estimate but never NaN.
MOLECULAR_WEIGHTS: dict[str, float] = {
    # Carbon sources
    "GLC": 180.16, "GLC-D-LACTONE": 178.14, "GLYCEROL": 92.09,
    "ACET": 60.05, "FORMATE": 46.03, "ETOH": 46.07,
    "D-LACTATE": 90.08, "SUC": 118.09, "MAL": 134.09, "FUM": 116.07,
    "ARABINOSE": 150.13, "BETAINE": 117.15, "BUTANAL": 72.11,
    # One-carbon
    "CARBON-DIOXIDE": 44.01, "CARBON-MONOXIDE": 28.01,
    "UREA": 60.06, "METOH": 32.04,
    # Nitrogen / phosphorus / sulfur
    "AMMONIUM": 18.04, "NITRATE": 62.00, "NITRITE": 46.01,
    "Pi": 95.98, "SULFATE": 96.06,
    # Gases, water, proton
    "OXYGEN-MOLECULE": 32.00, "HYDROGEN-MOLECULE": 2.02,
    "WATER": 18.02, "PROTON": 1.01,
    # Amino acids (free acid MWs)
    "L-ALPHA-ALANINE": 89.09, "ARG": 174.20, "ASN": 132.12,
    "L-ASPARTATE": 133.10, "CYS": 121.16, "GLN": 146.15,
    "GLT": 147.13, "GLY": 75.07, "HIS": 155.16, "ILE": 131.17,
    "LEU": 131.17, "LYS": 146.19, "MET": 149.21, "PHE": 165.19,
    "PRO": 115.13, "SER": 105.09, "THR": 119.12, "TRP": 204.23,
    "TYR": 181.19, "VAL": 117.15, "D-ALANINE": 89.09,
    "L-SELENOCYSTEINE": 168.05,
    # Ions
    "K+": 39.10, "MG+2": 24.31, "NA+": 22.99, "CA+2": 40.08,
    "CL-": 35.45, "FE+2": 55.85, "FE+3": 55.85, "MN+2": 54.94,
    "ZN+2": 65.38, "NI+2": 58.69, "CO+2": 58.93,
    "TUNGSTATE": 247.86, "4FE-4S": 351.65,
    # Nucleobases
    "CYTOSINE": 111.10, "CYTIDINE": 243.22,
    "THYMINE": 126.11, "URACIL": 112.09,
    "HYPOXANTHINE": 136.11, "XANTHINE": 152.11,
    "INDOLE": 117.15, "IMIDAZOLE-PYRUVATE": 154.13,
    # Large cofactors
    "S-ADENOSYLMETHIONINE": 398.44,
    "S-ADENOSYL-4-METHYLTHIO-2-OXOBUTANOATE": 384.42,
    "CH33ADO": 297.33, "5-Deoxy-D-Ribofuranose": 134.13,
    "GLYCOLALDEHYDE": 60.05, "GLYCOLLATE": 76.05,
    "MI-PENTAKISPHOSPHATE": 579.98,
    "UNDECAPRENYL-DIPHOSPHATE": 926.45,
}


def mw_of(molecule: str) -> float:
    """Molecular weight (g/mol) for a boundary molecule id. Returns 0
    for unknowns so mass-balance partial coverage is visible as an
    under-count rather than NaN propagation."""
    if "[" in molecule:
        molecule = molecule.split("[", 1)[0]
    return MOLECULAR_WEIGHTS.get(molecule, 0.0)
