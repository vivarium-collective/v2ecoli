"""
Generate a BioNetGen model for E. coli flagella complexation.

Reads the complexation stoichiometry (see below) and generates a BNGL file
with sequential bimolecular binding rules for each assembly step.

Each multi-subunit complex is modeled as a scaffold molecule with counter
states that track how many of each subunit have been incorporated. Monomers
bind one at a time via bimolecular reactions. When all subunits are bound,
a final rule converts the scaffold into the completed complex.
"""
import os

# ---------------------------------------------------------------------------
# CITATIONS USED IN THIS FILE:
#   Stoichiometry: fliN -111 (PMC10128058); fliG/fliF -34 (C-ring 34-fold
#     symmetry, cryo-EM); fliP:fliQ:fliR -5:-4:-1 (Kuhlen et al. 2018, Nat
#     Struct Mol Biol, building on Fukumura et al. 2017, PLOS Biol
#     15:e2002281); flhA -9 (PLOS One, doi:10.1371/journal.pone.0252800);
#     flgH/flgI -26 each, flgK/flgL -11 each (C26 symmetry, cryo-EM).
#   Hierarchy: MS-ring -> C-ring -> export apparatus -> motor complex
#     (Minamino & Namba 2008, Nature; Chevance & Hughes 2008, Nat Rev
#     Microbiol); rod -> P-ring -> L-ring -> hook (Cohen & Hughes 2014, J
#     Bacteriol 196:2387).
#   Rates: see "Rate constants" section below (McMurry et al. 2015;
#     Matsunami et al. 2016; Sim et al. 2017).
#
# THREE SPECIES KEPT AS DESCRIPTIVE NAMES, NOT RENAMED -- no real WCM bulk
# molecule corresponds to them:
#   'flagellar export apparatus subunit' -- purely this generator's own
#     internal bookkeeping (splits the 9-reactant export-apparatus assembly
#     into two sub-reactions; the real WCM Step does this in one shot).
#   'flagellar hook' -- the real WCM has NO discrete "hook complete" bulk
#     molecule at all; flagella_filament_nucleation.py merges hook
#     completion and nucleation into a single Step/event (consumes motor
#     complex + 120x FlgE + 11x FlgK + 11x FlgL directly, creating a
#     nascent_flagellum UNIQUE molecule, not a bulk one).
#   'flagella' (final observable) -- corresponds to creating a
#     nascent_flagellum UNIQUE molecule in the real pipeline (via
#     flagella_filament_nucleation.py today), not a bulk-store molecule --
#     bridging this is the wrapper Step's job, not a rename.
#
# FLIC REMOVAL (2026-08-12): fliC at -5000 (real target_length) caused a
# combinatorial explosion (237 -> 5,588 rules, 486 KB) -- same problem
# v2ecoli already solved by pulling filament elongation OUT of the
# reaction network entirely (flagella_filament_elongation.py: dL/dt =
# a/(b+L), Renault et al. 2017). fliC and filament growth are EXCLUDED
# from this model; the 'flagellum reaction' below represents assembly
# complete through the HOOK-BASAL-BODY stage only. Filament elongation is
# out of scope here -- v2ecoli's existing flagella_filament_elongation.py
# should be reused directly when this couples to the real WCM, not
# reimplemented in BNGL.
# ---------------------------------------------------------------------------
COMPLEXATION_STOICHIOMETRY = {
    'flhDC': {
        'EG10320-MONOMER[c]': -4.0,
        'MONOMER0-2488[c]': -2.0,
        'CPLX0-3930[c]': 1.0,
    },
    'flagellar motor switch reaction': {
        'CPLX0-7450[i]': 1.0,
        'FLIF-FLAGELLAR-MS-RING[i]': -34.0,
        'FLIG-FLAGELLAR-SWITCH-PROTEIN[i]': -34.0,
        'FLIM-FLAGELLAR-C-RING-SWITCH[i]': -34.0,
        'FLIN-FLAGELLAR-C-RING-SWITCH[m]': -111.0,
    },
    'flagellar export apparatus reaction 1': {
        'flagellar export apparatus subunit': 1.0,  # no real bulk ID -- generator-internal, see note above
        'CPLX0-7450[i]': -1.0,        # was 'flagellar motor switch'
        'G370-MONOMER[i]': -9.0,
        'G7028-MONOMER[i]': -1.0,
        'EG11224-MONOMER[j]': -1.0,
        'EG11975-MONOMER[i]': -5.0,
        'EG11976-MONOMER[j]': -4.0,
        'EG11977-MONOMER[i]': -1.0,
        'G378-MONOMER[c]': -1.0,
        'G377-MONOMER[c]': -6.0,
    },
    'flagellar export apparatus reaction 2': {
        'CPLX0-7451[j]': 1.0,                     # was 'flagellar export apparatus'
        'flagellar export apparatus subunit': -1.0,  # no real bulk ID -- generator-internal
        'EG11656-MONOMER[c]': -12.0,  # was fliH
    },
    'flagellar rod reaction': {
        'flagellar rod': 1.0,   # no real bulk ID -- new intermediate
        'CPLX0-7451[j]': -1.0,               # was 'flagellar export apparatus'
        'EG10322-MONOMER[j]': -2.0,
        'FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -5.0,
        'FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -6.0,
        'FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]': -5.0,
        'FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]': -24.0,
        'EG11346-MONOMER[p]': -6.0,
    },
    'flagellar p-ring reaction': {
        'flagellar rod with p-ring': 1.0,   # no real bulk ID -- new intermediate
        'flagellar rod': -1.0,
        'FLGI-FLAGELLAR-P-RING[j]': -26.0,
    },
    'flagellar l-ring reaction': {
        'FLAGELLAR-MOTOR-COMPLEX[j]': 1.0,   # real bulk ID -- now represents
        # the finished rod+P-ring+L-ring base, ready for the hook
        'flagellar rod with p-ring': -1.0,
        'FLGH-FLAGELLAR-L-RING[j]': -26.0,
    },
    'flagellar hook reaction': {
        'flagellar hook': 1,           # no real bulk ID -- see note above
        # NEW 2026-08-27: hook now requires the finished base
        'FLAGELLAR-MOTOR-COMPLEX[j]': -1.0,
        'G361-MONOMER[c]': -120.0,
    },
    'flagellum reaction': {
        'flagella': 1.0,   # no real bulk ID (maps to nascent_flagellum creation)
        'EG11545-MONOMER[e]': -11.0,
        'EG11967-MONOMER[e]': -11.0,
        'EG10841-MONOMER[e]': -5.0,    # FliD, correct timing -- see note below
        'flagellar hook': -1,          # no real bulk ID -- see note above,
        # now carries the whole rod+P-ring+L-ring+hook history forward.
    },
}

# FLID TIMING FIX (2026-09-17): FliD caps the hook-filament junction BEFORE
# flagellin polymerization, not after (Song et al. 2017, J Mol Biol
# 429:847; Postel et al. 2020, Nat Commun 11:1965). Charged here, at
# hook-basal-body-cap completion, instead of at filament-elongation
# completion -- fliD_per_completion set to 0 in
# flagella_filament_elongation.py's config (sim_data.py) to avoid a
# double-count.

# ---------------------------------------------------------------------------
# Rate constants -- UPDATED 2026-08-12. K_BIND was previously an arbitrary
# placeholder (5e-1, no citation). Replaced with a real, literature-measured
# association rate constant, converted to the per-molecule stochastic units
# BNGL/NFsim expects.
#
# FlhA<->FlhB binding (flagellar export apparatus): k_on = 8.5e4 M^-1 s^-1,
# k_off = 0.09 s^-1, biosensor/SPR measurement, Salmonella enterica.
# McMurry, Sampson, Case & Hughes (2015), "Weak Interactions between
# Salmonella enterica FlhB and Other Flagellar Export Apparatus Proteins
# Govern Type III Secretion Dynamics," PLOS One,
# doi:10.1371/journal.pone.0134884 (PMC4526367).
# This SAME value is used as default k_bind for every OTHER binding
# reaction in this model lacking its own measured rate. A real
# same-assembly proxy (FlhA/FlhB are both export-apparatus subunits), not
# a generic guess -- but still a proxy, not a per-interaction measurement,
# for every reaction except the one it was actually measured for.

# Deterministic (M^-1 s^-1) -> stochastic (molecules^-1 s^-1) conversion:
#   k_stoch = k_on / (N_A * V)
# using V = 1 femtoliter (1e-15 L), the standard assumed E. coli cell volume
# in stochastic bacterial modeling.

# FlgA<->FlgI binding (flagellar P-ring assembly, periplasmic chaperone):
# k_on = 3.41e5 M^-1 s^-1, k_off = 5.89e-3 s^-1, KD = 0.126 uM. SPR,
# FlgI immobilized on the sensor chip / FlgA as the analyte in solution,
# Salmonella enterica. Matsunami, Yoon, Meshcheryakov, Namba & Samatey
# (2016), "Structural flexibility of the periplasmic protein, FlgA,
# regulates flagellar P-ring assembly in Salmonella enterica," Scientific
# Reports 6:27399, doi:10.1038/srep27399 (PMC4895218), Table 2.
# Applied to 'flagellar p-ring reaction' only.

N_AVOGADRO = 6.022e23         # /mol
CELL_VOLUME_L = 1e-15         # 1 fL, standard assumed E. coli cell volume
K_ON_FLHA_FLHB_MOLAR = 8.5e4  # M^-1 s^-1, McMurry et al. 2015 (real, cited)
K_OFF_FLHA_FLHB = 0.09        # s^-1, McMurry et al. 2015
K_BIND = K_ON_FLHA_FLHB_MOLAR / (N_AVOGADRO * CELL_VOLUME_L)  # ~1.412e-4 /molecule/s

K_ON_FLGA_FLGI_MOLAR = 3.41e5  # M^-1 s^-1, Matsunami et al. 2016, Table 2
K_OFF_FLGA_FLGI = 5.89e-3      # s^-1, same source
K_FLGI_ELONGATION = K_ON_FLGA_FLGI_MOLAR / (N_AVOGADRO * CELL_VOLUME_L)  # ~5.663e-4 /molecule/s
K_COMPLETION = 10.0   # still an unconverted placeholder -- no literature search done for this rate

# NUCLEATION FIX (2026-08-12): the original nucleation rate formula was
# independent of k_bind, letting new scaffolds nucleate far faster than
# any existing one could finish (4,530 real reaction events over a 2400s
# test run, zero complexes ever completed). Real E. coli flagellar
# assembly does the opposite -- existing structures preferentially absorb
# material over nucleating new ones (Chang, Sung & Hong 2025, Biochem
# Biophys Reports 42:102051, same citation flagella_filament_nucleation.py
# uses for the same principle). Fixed by tying nucleation rate to a small
# fraction of k_bind, so an existing scaffold's elongation always
# outcompetes starting a new one.

# REAL NUCLEATION RATE (2026-08-17): for the C-ring/MS-ring reaction
# ('flagellar motor switch reaction'), Sim et al. 2017 (Sci Rep 7:41189)
# measured 7.8 flagella/cell at 1.2hr doubling time -> ~1.81e-3/s per-cell
# rate, matching (within 8%) the 0.00167/s flagella_filament_nucleation.py
# already uses from the same paper -- used here for consistency.
#
# CAVEAT: 0.00167/s is a concentration-independent per-cell event rate;
# NFsim's nucleation is bimolecular mass-action instead (propensity =
# k_nuc * [FliF]^2). Resolved by calibrating k_nuc so propensity matches
# 0.00167/s AT the real ambient FliF count used throughout (657):
#   k_nuc_cring = 0.00167 / (657 * 656 / 2) ~= 7.75e-9 /molecule/s
# This only matches the real rate at that reference concentration -- as
# FliF rises or falls, propensity scales with [FliF]^2, not pinned to
# 0.00167/s. A real, accepted limitation of this rate-law mismatch.
REAL_NUCLEATION_RATE_PER_S = 0.00167  # Sim et al. 2017, same citation as
                                        # flagella_filament_nucleation.py

REAL_AMBIENT_MONOMER_COUNTS = {
    'FLIF-FLAGELLAR-MS-RING[i]': 657,   # C-ring nucleating species
    'MONOMER0-2488[c]': 649,            # FlhC, flhDC nucleating species
}

# NOT FIXED HERE: this module now accepts an optional
# `real_ambient_monomer_counts` override (see generate_bngl/write_bngl/
# get_model_path below) so a caller building a condition-specific model CAN
# pass the cell's actual live counts instead of this hardcoded snapshot --
# but nothing currently DOES that automatically. The real fix -- having
# flagella_nfsim_complexation.py's initialize() read the live/cached ambient
# counts for whatever condition it's actually running under and pass them
# through before calling get_model_path() -- is not yet built. Revisit
# before trusting any non-basal-condition NFsim result quantitatively.
_DEFAULT_REAL_AMBIENT_MONOMER_COUNTS = REAL_AMBIENT_MONOMER_COUNTS


def _nucleating_species(consumed):
    """Return (nuc_species_1, nuc_species_2) for a reaction's consumed dict
    -- the exact same species-selection logic the rule-writing loop below
    uses, factored out so the parameter-writing loop can compute a
    per-reaction nucleation rate using the SAME nucleating pair."""
    species_by_count = sorted(consumed.keys(), key=lambda s: consumed[s])
    nuc_species_1 = species_by_count[0]
    if consumed[nuc_species_1] >= 2:
        nuc_species_2 = nuc_species_1
    else:
        nuc_species_2 = species_by_count[1]
    return nuc_species_1, nuc_species_2


def _calibrated_nucleation_rate(consumed, k_bind, real_ambient_monomer_counts=None):
    """Real, per-reaction nucleation rate: if the nucleating species has a
    known real ambient count, calibrate so propensity matches
    REAL_NUCLEATION_RATE_PER_S at that real count. Otherwise (nucleating
    from a scarce, dynamically-produced precursor), plain k_bind -- no
    artificial suppression needed.

    `real_ambient_monomer_counts` defaults to the module-level (basal-only)
    REAL_AMBIENT_MONOMER_COUNTS snapshot -- see that dict's own "NOT
    CONDITION-AWARE" comment. Pass a condition-specific dict here (same keys,
    real live counts) to calibrate correctly for a non-basal cell; nothing
    does this automatically yet.
    """
    if real_ambient_monomer_counts is None:
        real_ambient_monomer_counts = _DEFAULT_REAL_AMBIENT_MONOMER_COUNTS
    nuc_species_1, _ = _nucleating_species(consumed)
    real_count = real_ambient_monomer_counts.get(nuc_species_1)
    if real_count is None:
        return k_bind
    return REAL_NUCLEATION_RATE_PER_S / (real_count * (real_count - 1) / 2)


def _dynamic_nuc_species(consumed, real_ambient_monomer_counts=None):
    """Return the nucleating species name if this reaction's rate should
    track that species' LIVE count via a BNGL function, or None to use a
    static parameter (plain k_bind fallback)."""
    if real_ambient_monomer_counts is None:
        real_ambient_monomer_counts = _DEFAULT_REAL_AMBIENT_MONOMER_COUNTS
    nuc_species_1, _ = _nucleating_species(consumed)
    return nuc_species_1 if nuc_species_1 in real_ambient_monomer_counts else None

# Number of flagella worth of monomers to provide
N_FLAGELLA = 5


def _safe_name(name):
    """Convert a name to a valid BNG identifier.

    Extended 2026-08-12 to also strip '[' / ']' -- species keys are now real
    v2ecoli bulk molecule IDs (e.g. 'FLIF-FLAGELLAR-MS-RING[i]'), which carry
    a compartment suffix BNGL identifiers can't contain. This mapping is
    deterministic and one-way by design: a future wrapper Step that needs to
    go from a safe name back to the real bulk ID should keep its OWN
    {_safe_name(real_id): real_id} lookup built from the known real IDs
    (e.g. via real_bulk_ids() below), not try to invert this string
    transform, since '-' and '[' both collapse to the same '_' here.
    """
    return name.replace(' ', '_').replace('-', '_').replace('[', '_').replace(']', '')


# Species keys in COMPLEXATION_STOICHIOMETRY that ARE real v2ecoli bulk
# molecule IDs (i.e. everything except the three generator-internal-only
# names documented above: 'flagellar export apparatus subunit',
# 'flagellar hook', 'flagella'). Added 2026-08-12 for the future wrapper
# Step to build its safe-name -> real-ID lookup from, rather than
# guessing/inverting _safe_name()'s output.
_NON_BULK_SPECIES = {
    'flagellar export apparatus subunit', 'flagellar hook', 'flagella',
    # New 2026-08-27 (HOOK DEPENDENCY FIX): the two new in-between items on
    # the way to a finished base -- rod alone, then rod with the P-ring.
    # Neither has a real bulk ID, same as the three above.
    'flagellar rod', 'flagellar rod with p-ring',
}


def real_bulk_ids():
    """Return the set of species names in COMPLEXATION_STOICHIOMETRY that are
    real v2ecoli bulk molecule IDs (excludes the 3 generator-internal-only
    names -- see _NON_BULK_SPECIES)."""
    names = set()
    for stoich in COMPLEXATION_STOICHIOMETRY.values():
        names.update(stoich.keys())
    return names - _NON_BULK_SPECIES


def _parse_reaction(rxn_name, stoich):
    """Parse a reaction into consumed monomers and produced complex."""
    consumed = {}
    product = None
    for species, count in stoich.items():
        if count < 0:
            consumed[species] = int(abs(count))
        elif count > 0:
            product = species
    return consumed, product


def default_production_rates():
    """Compute default rates: produce enough monomers for 1 flagellum per 100s."""
    duration = 100.0

    demand = {}
    complex_names = set()
    for stoich in COMPLEXATION_STOICHIOMETRY.values():
        for species, count in stoich.items():
            if count > 0:
                complex_names.add(species)
            else:
                demand[species] = demand.get(species, 0) + abs(count)

    rates = {}
    for species, count in demand.items():
        if species not in complex_names:
            rates[f'Free_{_safe_name(species)}'] = count / duration

    return rates


# Ordered reactions (assembly hierarchy). Extracted to a module constant
# 2026-08-12 so both generate_bngl() and bulk_id_to_observable_name() share
# the exact same reaction/species derivation instead of duplicating it.
REACTION_ORDER = [
    'flhDC',
    'flagellar motor switch reaction',
    'flagellar export apparatus reaction 1',
    'flagellar export apparatus reaction 2',
    'flagellar rod reaction',
    'flagellar p-ring reaction',
    'flagellar l-ring reaction',
    'flagellar hook reaction',
    'flagellum reaction',
]


def _parse_all_reactions():
    """Parse every reaction in REACTION_ORDER; return (reactions,
    monomer_names, complex_names_ordered, complex_names, all_consumed) --
    the same derivation generate_bngl() needs, factored out so
    bulk_id_to_observable_name() can reuse it exactly rather than
    duplicating the monomer-vs-complex split."""
    reactions = {}
    for rxn_name in REACTION_ORDER:
        consumed, product = _parse_reaction(rxn_name, COMPLEXATION_STOICHIOMETRY[rxn_name])
        reactions[rxn_name] = {'consumed': consumed, 'product': product}

    complex_names = set()
    for rxn in reactions.values():
        complex_names.add(rxn['product'])

    all_consumed = set()
    for rxn in reactions.values():
        all_consumed.update(rxn['consumed'].keys())

    monomer_names = sorted(all_consumed - complex_names)
    complex_names_ordered = [reactions[r]['product'] for r in REACTION_ORDER]
    return reactions, monomer_names, complex_names_ordered, complex_names, all_consumed


def bulk_id_to_observable_name():
    """Return {real_bulk_id: NFsim_observable_name} for every real v2ecoli
    bulk molecule ID this model uses (real_bulk_ids()) -- monomers are
    observed as 'Free_{safe_name}', complex products as bare '{safe_name}'
    (matches generate_bngl()'s own observables-block emission exactly, see
    the 'begin observables' section below). Added 2026-08-12 so the wrapper
    Step doesn't need to re-derive which real IDs are monomers vs. products -- reuses
    _parse_all_reactions() directly rather than guessing."""
    _, monomer_names, complex_names_ordered, _, _ = _parse_all_reactions()
    real_ids = real_bulk_ids()
    mapping = {}
    for name in monomer_names:
        if name in real_ids:
            mapping[name] = f'Free_{_safe_name(name)}'
    for name in complex_names_ordered:
        if name in real_ids:
            mapping[name] = _safe_name(name)
    return mapping


def generate_bngl(n_flagella=N_FLAGELLA, k_bind=K_BIND,
                   k_completion=K_COMPLETION, real_ambient_monomer_counts=None,
                   k_flgI_elongation=K_FLGI_ELONGATION):
    """Generate the complete BNGL model string.

    `real_ambient_monomer_counts`: optional override for
    REAL_AMBIENT_MONOMER_COUNTS (same keys: real bulk IDs -> real ambient
    count), forwarded to _calibrated_nucleation_rate. Defaults to the
    basal-only hardcoded snapshot -- see that dict's "NOT CONDITION-AWARE"
    comment. Pass the cell's actual live counts here to get a correctly
    calibrated nucleation rate for a non-basal condition.

    `k_flgI_elongation`: real, Matsunami et al. 2016-derived rate (see
    K_FLGI_ELONGATION module docstring) used for the 'flagellar p-ring
    reaction' specifically (both its nucleation step and every per-subunit
    FlgI elongation step), in place of the generic k_bind used everywhere
    else.
    """

    reaction_order = REACTION_ORDER
    reactions, monomer_names, complex_names_ordered, complex_names, all_consumed = _parse_all_reactions()

    # Calculate initial monomer counts
    monomer_counts = {}
    for rxn in reactions.values():
        for species, count in rxn['consumed'].items():
            if species in monomer_names:
                needed = count * n_flagella
                monomer_counts[species] = max(monomer_counts.get(species, 0), needed)

    # ---- Build BNGL ----
    lines = []
    lines.append('begin model')
    lines.append('')

    # -- Parameters --
    lines.append('begin parameters')
    lines.append(f'    n_flagella  {n_flagella}')
    lines.append(f'    k_bind      {k_bind}')
    lines.append(f'    k_completion {k_completion}')
    lines.append(f'    k_flgI_elongation  {k_flgI_elongation}')  # Matsunami et al.
    # 2016, FlgA-FlgI SPR rate -- p-ring reaction only, see K_FLGI_ELONGATION
    # module docstring above for the full derivation/orientation rationale.
    lines.append('')
    # Nucleation rate per reaction -- see _calibrated_nucleation_rate()
    # docstring for the real-rate vs. plain-k_bind logic. Species with a
    # real ambient count get a live BNGL function instead (see "begin
    # functions" below) -- skipped here.
    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        total = sum(consumed.values())
        if total > 2:
            safe_product = _safe_name(product)
            if rxn_name == 'flagellar p-ring reaction':
                # Real FlgA-FlgI rate (Matsunami et al. 2016) applies to
                # the p-ring's nucleation step too -- it's just the FIRST
                # FlgI joining the rod, mechanistically identical to every
                # later FlgI-joining (elongation) event. See
                # K_FLGI_ELONGATION module docstring.
                this_rate = k_flgI_elongation
            elif _dynamic_nuc_species(consumed, real_ambient_monomer_counts):
                continue
            else:
                this_rate = _calibrated_nucleation_rate(
                    consumed, k_bind, real_ambient_monomer_counts)
            lines.append(f'    k_nuc_{safe_product}  {this_rate:.6e}')
    lines.append('')

    for monomer in sorted(monomer_counts.keys()):
        safe = _safe_name(monomer)
        lines.append(f'    {safe}_0  {monomer_counts[monomer]}')
    lines.append('end parameters')
    lines.append('')

    # -- Molecule Types --
    lines.append('begin molecule types')

    for monomer in monomer_names:
        lines.append(f'    {_safe_name(monomer)}()')

    intermediate_complexes = sorted(complex_names & all_consumed)
    for cx in intermediate_complexes:
        lines.append(f'    {_safe_name(cx)}()')

    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        safe_product = _safe_name(product)

        total_subunits = sum(consumed.values())

        if total_subunits <= 2:
            continue

        state_parts = []
        for species in sorted(consumed.keys()):
            count = consumed[species]
            safe_species = _safe_name(species)
            states = '~'.join(str(i) for i in range(count + 1))
            state_parts.append(f'{safe_species}~{states}')

        scaffold_name = f'Growing_{safe_product}'
        lines.append(f'    {scaffold_name}({",".join(state_parts)})')

    final_complexes = sorted(complex_names - all_consumed)
    for cx in final_complexes:
        lines.append(f'    {_safe_name(cx)}()')

    lines.append('end molecule types')
    lines.append('')

    # -- Seed Species --
    lines.append('begin seed species')
    for monomer in sorted(monomer_counts.keys()):
        safe = _safe_name(monomer)
        lines.append(f'    {safe}()  {safe}_0')
    lines.append('end seed species')
    lines.append('')

    # -- Observables --
    lines.append('begin observables')

    for monomer in monomer_names:
        safe = _safe_name(monomer)
        lines.append(f'    Molecules  Free_{safe}  {safe}()')

    for cx_name in complex_names_ordered:
        safe = _safe_name(cx_name)
        lines.append(f'    Molecules  {safe}  {safe}()')

    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        safe_product = _safe_name(product)
        total_subunits = sum(consumed.values())
        if total_subunits > 2:
            scaffold_name = f'Growing_{safe_product}'
            lines.append(f'    Molecules  {scaffold_name}_total  {scaffold_name}()')

    lines.append('end observables')
    lines.append('')

    # -- Functions -- nucleation rate tracks the LIVE ambient count (via
    # Free_X, defined above), not a fixed snapshot. See
    # _dynamic_nuc_species() docstring.
    lines.append('begin functions')
    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        if sum(consumed.values()) <= 2:
            continue
        dyn_species = _dynamic_nuc_species(consumed, real_ambient_monomer_counts)
        if dyn_species is None:
            continue
        safe_product = _safe_name(product)
        safe_species = _safe_name(dyn_species)
        lines.append(
            f'    k_nuc_{safe_product}() = {REAL_NUCLEATION_RATE_PER_S} / '
            f'max(Free_{safe_species}*(Free_{safe_species}-1)/2, 1)')
    lines.append('end functions')
    lines.append('')

    # -- Reaction Rules --
    lines.append('begin reaction rules')

    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        safe_product = _safe_name(product)
        total_subunits = sum(consumed.values())

        lines.append(f'')
        lines.append(f'    # === {rxn_name} ===')
        lines.append(f'    # Product: {product}')
        lines.append(f'    # Subunits: {", ".join(f"{c}x {s}" for s, c in consumed.items())}')

        if total_subunits == 1:
            species = list(consumed.keys())[0]
            safe_species = _safe_name(species)
            lines.append(f'    {safe_species}() -> {safe_product}()  k_bind')

        elif total_subunits == 2:
            species_list = []
            for species, count in consumed.items():
                for _ in range(count):
                    species_list.append(species)

            if len(species_list) == 2:
                s1, s2 = species_list
                lines.append(f'    {_safe_name(s1)}() + {_safe_name(s2)}() -> {safe_product}()  k_bind')

        else:
            scaffold_name = f'Growing_{safe_product}'
            sorted_species = sorted(consumed.keys())

            species_by_count = sorted(consumed.keys(), key=lambda s: consumed[s])
            nuc_species_1 = species_by_count[0]
            if consumed[nuc_species_1] >= 2:
                nuc_species_2 = nuc_species_1
            else:
                nuc_species_2 = species_by_count[1]

            init_states = []
            nuc_counts = {}
            for species in sorted_species:
                if species == nuc_species_1:
                    nuc_counts[species] = nuc_counts.get(species, 0) + 1
                if species == nuc_species_2:
                    nuc_counts[species] = nuc_counts.get(species, 0) + 1

            for species in sorted_species:
                safe_sp = _safe_name(species)
                c = nuc_counts.get(species, 0)
                init_states.append(f'{safe_sp}~{c}')

            safe_nuc1 = _safe_name(nuc_species_1)
            safe_nuc2 = _safe_name(nuc_species_2)

            nuc_rate_name = f'k_nuc_{safe_product}'
            if rxn_name != 'flagellar p-ring reaction' and _dynamic_nuc_species(
                    consumed, real_ambient_monomer_counts):
                nuc_rate_name += '()'  # live BNGL function, not a static parameter
            lines.append(f'    # Nucleation (rate scaled by 1/total_subunits)')
            lines.append(f'    {safe_nuc1}() + {safe_nuc2}() -> '
                         f'{scaffold_name}({",".join(init_states)})  {nuc_rate_name}')

            # Real FlgA-FlgI rate (Matsunami et al. 2016) for every
            # per-subunit FlgI elongation step in the p-ring reaction --
            # see K_FLGI_ELONGATION module docstring. Every other
            # reaction's elongation steps keep plain k_bind.
            elong_rate_name = ('k_flgI_elongation'
                                if rxn_name == 'flagellar p-ring reaction'
                                else 'k_bind')

            for species in sorted_species:
                safe_sp = _safe_name(species)
                count = consumed[species]

                start = nuc_counts.get(species, 0)

                for i in range(start, count):
                    lines.append(
                        f'    {scaffold_name}({safe_sp}~{i}) + {_safe_name(species)}() -> '
                        f'{scaffold_name}({safe_sp}~{i + 1})  {elong_rate_name}')

            complete_states = []
            for species in sorted_species:
                safe_sp = _safe_name(species)
                complete_states.append(f'{safe_sp}~{consumed[species]}')

            lines.append(f'    # Completion')
            lines.append(f'    {scaffold_name}({",".join(complete_states)}) -> '
                         f'{safe_product}()  k_completion')

    lines.append('')
    lines.append('end reaction rules')
    lines.append('')
    lines.append('end model')

    return '\n'.join(lines)


def write_bngl(output_path=None, **kwargs):
    """Generate and write the BNGL model file."""
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), 'flagella_complexation.bngl')

    bngl_text = generate_bngl(**kwargs)

    with open(output_path, 'w') as f:
        f.write(bngl_text)

    return output_path


def get_model_path(real_ambient_monomer_counts=None):
    """Return path to the generated BNGL model, generating it if needed.

    `real_ambient_monomer_counts`: forwarded to generate_bngl -- see that
    function's docstring and REAL_AMBIENT_MONOMER_COUNTS's "NOT
    CONDITION-AWARE" comment. NOTE: if the file already exists on disk (the
    common case -- flagella_nfsim_complexation.py calls this with no
    arguments), it is reused AS-IS regardless of this argument -- passing a
    condition-specific override here only takes effect when the file doesn't
    exist yet, or the caller deletes it first / calls write_bngl directly.
    Nothing currently regenerates a condition-specific model automatically;
    this is the wiring gap flagged in REAL_AMBIENT_MONOMER_COUNTS's comment.
    """
    path = os.path.join(
        os.path.dirname(__file__), 'flagella_complexation.bngl')
    if not os.path.exists(path):
        write_bngl(path, real_ambient_monomer_counts=real_ambient_monomer_counts)
    return path


def make_production_document(
    model_file=None,
    n_steps=100,
    complexation_interval=50.0,
    production_interval=1.0,
    production_rate_scale=1.0,
):
    """v2ecoli-local equivalent of pbg_nfsim.composites.make_production_document.

    Added 2026-08-12. pbg_nfsim's own version hardcodes get_model_path() and
    default_production_rates() from ITS OWN bundled (generic/uncorrected)
    generate_flagella_bngl.py -- no way to pass a custom model_file and get
    matching production rates. This sources BOTH from THIS module (the
    v2ecoli-owned, corrected stoichiometry) so they stay consistent.
    NFSimProcess/MonomerProduction (generic engine code, via
    flagella_nfsim_assembly.py) still come from pbg_nfsim.
    """
    if model_file is None:
        model_file = get_model_path()

    rates = default_production_rates()
    scaled_rates = {
        name: rate * production_rate_scale
        for name, rate in rates.items()
    }

    return {
        'production': {
            '_type': 'process',
            'address': 'local:monomer-production',
            'config': {
                'production_rates': scaled_rates,
            },
            'outputs': {
                'monomers': ['species'],
            },
            'interval': production_interval,
        },
        'complexation': {
            '_type': 'process',
            'address': 'local:nfsim',
            'config': {
                'model_file': model_file,
                'n_steps': n_steps,
            },
            'inputs': {
                'observables': ['species'],
                # Added 2026-08-12: without this, NFSimProcess's scaffold
                # persistence fix (pbg_nfsim/processes.py) has nowhere to
                # round-trip through -- the port existed on the process but
                # was never connected to a store, so Growing_X scaffold
                # state was still silently dropped every interval despite
                # the wrapper itself now being capable of carrying it.
                'scaffold_species': ['scaffold'],
            },
            'outputs': {
                'observables': ['species'],
                'scaffold_species': ['scaffold'],
            },
            'interval': complexation_interval,
        },
        'species': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'species': 'map[float]',
                    'time': 'float',
                },
            },
            'inputs': {
                'species': ['species'],
                'time': ['global_time'],
            },
        },
    }


if __name__ == '__main__':
    path = write_bngl()
    print(f'Model written to: {path}')

    with open(path) as f:
        text = f.read()
    n_rules = text.count(' -> ')
    print(f'Reaction rules: {n_rules}')
