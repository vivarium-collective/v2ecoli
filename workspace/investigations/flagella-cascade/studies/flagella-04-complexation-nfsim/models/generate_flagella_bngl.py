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
# Stoichiometry -- UPDATED 2026-08-12 to match v2ecoli's corrected reaction
# network (see v2ecoli's flagella-cascade investigation,
# workspace/investigations/flagella-cascade/studies/
# flagella-02-transcription-regulation/study.yaml, findings
# flagella-02-export-apparatus-flip-flop-stoichiometry-fixed,
# flagella-02-cring-export-apparatus-hierarchy-fix,
# flagella-02-ms-ring-ordering-fix). Original values here were sourced from
# an older, unrelated codebase (vivarium-chemotaxis flagella_chromosome.py)
# and had NOT been checked against real structural literature -- most
# multi-subunit coefficients were simple placeholders (mostly -1), and the
# assembly ORDER (which complex depends on which) didn't match real biology
# either. Both are fixed here. Negative values = consumed, positive =
# produced.
#
# STOICHIOMETRY FIXES (real cryo-EM literature, same citations as v2ecoli):
#   fliN: -1 -> -111 (was off by two orders of magnitude; PMC10128058,
#     "Precise Measurement of the Stoichiometry of the Adaptive Bacterial
#     Flagellar Switch")
#   fliG: -26 -> -34 (C-ring 34-fold rotational symmetry; two independent
#     literature searches this session both gave 34, not 26)
#   fliP:fliQ:fliR: -1:-1:-1 -> -5:-4:-1 (real cryo-EM-determined
#     stoichiometry; Kuhlen et al. 2018 Nat Struct Mol Biol, building on
#     Fukumura et al. 2017 PLOS Biol 15:e2002281/PMC5542437)
#   flhA: -1 -> -9 (homo-nonameric ring in the export gate; PLOS One
#     10.1371/journal.pone.0252800 + corroborating literature)
#   flgH, flgI (L-ring, P-ring): -1 -> -26 each (C26 symmetry, cryo-EM)
#   motA, motB (stator): -1 -> -55, -22 (derived estimate, medium
#     confidence -- 5:2 per-unit ratio is solid, x~11 stator units/motor is
#     the estimated part, flagged in v2ecoli's own docs too)
#   flgB, flgC, flgF (proximal rod): -1 -> -5, -6, -5
#   flgG (distal rod): -1 -> -24
#   fliE: -1 -> -6
#   flgK, flgL (hook-filament junction): -1 -> -11 each (cryo-EM)
#   fliF (MS-ring): was MISSING from the motor-switch reaction entirely (only
#     consumed, wrongly, at -1 in the final motor reaction) -> -34, moved to
#     the motor-switch reaction (see HIERARCHY FIX below)
#   fliC: REMOVED from this model entirely (2026-08-12) -- see FLIC REMOVAL
#     note below.
#
# HIERARCHY FIXES (real assembly order, Minamino & Namba 2008 Nature;
# Chevance & Hughes 2008 Nat Rev Microbiol -- MS-ring forms first, C-ring
# assembles onto it, THEN the export apparatus inserts into the C-ring's
# pore, THEN the export apparatus is incorporated into the motor complex):
#   'flagellar motor switch reaction' (C-ring) now also consumes fliF
#     (MS-ring) -- represents the merged MS-ring + C-ring structure, same
#     simplification v2ecoli uses (single-stage merge, not a third complex).
#   'flagellar export apparatus reaction 1' now ALSO consumes
#     'flagellar motor switch' (-1) -- the export apparatus can't form until
#     the C-ring already exists (was previously two independent branches
#     only merging at the motor reaction).
#   'flagellar motor reaction' no longer consumes 'flagellar motor switch'
#     or fliF directly (both moved upstream, see above) -- avoids double-
#     counting the C-ring/MS-ring for one motor complex.
#   'flagellar motor reaction' now consumes 'flagellar export apparatus'
#     (-1) instead of the final 'flagellum reaction' consuming it -- matches
#     v2ecoli's flagella_motor_complex_assembly.py, which consumes CPLX0-7451
#     to build FLAGELLAR-MOTOR-COMPLEX, not at filament completion.
#
# NOT changed here (logged as an open question in v2ecoli, not resolved):
#   fliO stays at -1, still a genuine consumed reactant in the export-
#   apparatus reaction, even though real literature describes it as a
#   transient assembly scaffold not necessarily part of the final mature
#   complex. Maya's explicit direction (2026-08-12): keep it at -1 for now;
#   this is exactly the kind of role-vs-copy-number question NFsim rules
#   should eventually constrain properly, not something to hand-fix here.
#
# FLIC REMOVAL (2026-08-12): fliC was briefly set to -5000 (matching
# v2ecoli's current target_length) and immediately caused the SAME
# combinatorial/file-size explosion v2ecoli itself hit and already solved --
# this generator represents "N copies bound" as N explicit numbered scaffold
# states, so a single coefficient of 5000 expanded into 5000 individual BNGL
# rules (237 -> 5,588 total rules, 486 KB file) for FliC alone. v2ecoli's own
# fix for the identical problem was to pull filament elongation OUT of the
# combinatorial engine entirely (Gillespie SSA there, this rule-matching
# network here) and run it as a separate incremental process
# (flagella_filament_elongation.py: dL/dt = a/(b+L), Renault et al. 2017,
# adding FliC one subunit at a time outside the reaction network). Rather
# than reimplementing that incremental logic a second time inside this
# standalone BNGL/NFsim example -- untested against the real WCM, and
# redundant with code that already works -- fliC and filament growth are
# EXCLUDED from this model entirely. The 'flagellum reaction' below now
# represents assembly complete through the HOOK-BASAL-BODY stage only (motor
# + hook + hook-filament junction + cap machinery) -- the real endpoint of
# what rule-based complexation was ever meant to validate here. Filament
# elongation (FliC incorporation) is explicitly out of scope for this
# standalone example; when this couples to the real WCM (task: assess NFsim
# coupling vs. this session's custom v2ecoli assembly Steps), v2ecoli's
# existing, working flagella_filament_elongation.py should be reused
# directly rather than reimplemented in BNGL. The final species is still
# named 'flagella' for backward compatibility with run_nfsim_assembly.py's
# existing observable name -- see that script's own updated comment.
# ---------------------------------------------------------------------------
COMPLEXATION_STOICHIOMETRY = {
    'flhDC': {
        'flhD': -4.0,
        'flhC': -2.0,
        'flhDC': 1.0,
    },
    'flagellar motor switch reaction': {
        'flagellar motor switch': 1.0,
        'fliF': -34.0,   # MS-ring -- MISSING before, added (MS-ring ordering fix)
        'fliG': -34.0,   # was -26
        'fliM': -34.0,   # unchanged
        'fliN': -111.0,  # was -1
    },
    'flagellar export apparatus reaction 1': {
        'flagellar export apparatus subunit': 1.0,
        'flagellar motor switch': -1.0,  # NEW -- C-ring dependency (hierarchy fix)
        'flhA': -9.0,    # was -1
        'flhB': -1.0,    # unchanged
        'fliO': -1.0,    # unchanged -- open question, see note above
        'fliP': -5.0,    # was -1
        'fliQ': -4.0,    # was -1
        'fliR': -1.0,    # unchanged (already the "1" in 5:4:1)
        'fliJ': -1.0,    # unchanged
        'fliI': -6.0,    # unchanged
    },
    'flagellar export apparatus reaction 2': {
        'flagellar export apparatus': 1.0,
        'flagellar export apparatus subunit': -1.0,
        'fliH': -12.0,   # unchanged
    },
    'flagellar motor reaction': {
        'flagellar motor': 1.0,
        'flagellar export apparatus': -1.0,  # NEW -- moved here from flagellum reaction (hierarchy fix)
        'fliL': -2.0,    # unchanged
        'flgH': -26.0,   # was -1
        'motA': -55.0,   # was -1
        'motB': -22.0,   # was -1
        'flgB': -5.0,    # was -1
        'flgC': -6.0,    # was -1
        'flgF': -5.0,    # was -1
        'flgG': -24.0,   # was -1
        'flgI': -26.0,   # was -1
        'fliE': -6.0,    # was -1
        # 'flagellar motor switch': -1.0,  REMOVED -- consumed upstream now (hierarchy fix)
        # 'fliF': -1.0,  REMOVED -- consumed upstream now (MS-ring ordering fix)
    },
    'flagellar hook reaction': {
        'flagellar hook': 1,
        'flgE': -120.0,  # unchanged, already correct
    },
    'flagellum reaction': {
        'flagella': 1.0,   # hook-basal-body complete -- NOT the full filament-bearing organelle, see FLIC REMOVAL note above
        'flagellar motor': -1.0,  # unchanged
        # 'fliC': -5000.0,  REMOVED -- see FLIC REMOVAL note above (kept per standing preserve-old-code rule)
        'flgL': -11.0,   # was -1
        'flgK': -11.0,   # was -1
        'fliD': -5.0,    # unchanged -- cap, still installed here even though the filament it caps isn't modeled
        'flagellar hook': -1,  # unchanged
        # 'flagellar export apparatus': -1.0,  REMOVED -- moved to flagellar motor reaction (hierarchy fix)
    },
}

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
#
# Deterministic (M^-1 s^-1) -> stochastic (molecules^-1 s^-1) conversion:
#   k_stoch = k_on / (N_A * V)
# using V = 1 femtoliter (1e-15 L), the standard assumed E. coli cell volume
# in stochastic bacterial modeling (v2ecoli itself derives volume dynamically
# from dry_mass rather than a fixed constant, so no repo-internal value to
# match here -- this is the field's usual convention instead).
#
# This SAME converted value is also used as the default k_bind for every
# OTHER binding reaction in this model that lacks its own measured rate
# constant (the C-ring: FliG/FliM/FliN/FliF -- searched, only structural
# characterization exists in the literature, no kinetics; and the rest of
# the export apparatus and motor-complex components). This is a real,
# same-organelle, same-protein-family adjacent proxy -- not a generic guess
# -- since it's measured from a direct physical neighbor within this exact
# assembly (FlhA/FlhB are both export-apparatus subunits), rather than an
# unrelated system's "typical" protein-protein rate. Flagged explicitly:
# this is a proxy, not a per-interaction measurement, for every reaction
# except the one it was actually measured for.
N_AVOGADRO = 6.022e23         # /mol
CELL_VOLUME_L = 1e-15         # 1 fL, standard assumed E. coli cell volume
K_ON_FLHA_FLHB_MOLAR = 8.5e4  # M^-1 s^-1, McMurry et al. 2015 (real, cited)
K_OFF_FLHA_FLHB = 0.09        # s^-1, McMurry et al. 2015 -- NOTE: not
# currently used anywhere; this generator only emits irreversible ("->")
# binding rules, so there is no reverse reaction to attach this to yet. Real
# value, kept here for whenever reversibility is added.
K_BIND = K_ON_FLHA_FLHB_MOLAR / (N_AVOGADRO * CELL_VOLUME_L)  # ~1.412e-4 /molecule/s
K_NUCLEATION = 5e-2   # unused now -- see NUCLEATION_SUPPRESSION_FACTOR below
K_COMPLETION = 10.0   # still an unconverted placeholder -- no literature search done for this rate

# NUCLEATION FIX (2026-08-12): the original per-reaction nucleation rate was
# computed from a "target_propensity" formula (see generate_bngl below,
# original: nuc_rate = (n_flagella/50) / combinatorial) that was INDEPENDENT
# of k_bind -- with the corrected, much larger real stoichiometry, this let
# new scaffolds nucleate far faster than any existing scaffold could finish
# (4,530 real reaction events fired over a 2400s test run, species diversity
# growing throughout, yet ZERO complexes ever completed -- confirmed by
# actually running the model, not assumed). Real E. coli flagellar assembly
# does the opposite: existing structures preferentially absorb material over
# nucleating new ones (Chang, Sung & Hong 2025, Biochem Biophys Reports
# 42:102051, "Intrinsic clustering of flagellar basal body proteins in E.
# coli" -- the SAME citation v2ecoli's own flagella_filament_nucleation.py
# uses for the identical principle at the whole-flagellum level).
#
# Fixed by tying nucleation rate to a small FRACTION of k_bind instead of an
# independent formula, so that once any scaffold already exists, adding to
# it (propensity = k_bind * scaffold_count * free_monomer) overwhelmingly
# outcompetes starting a new one (propensity = k_nuc * free_monA * free_monB)
# -- the qualitative mechanism real biology uses, not a specific literature
# rate (no such per-sub-assembly nucleation rate was found in literature;
# this is a design-level ratio choice, analogous in spirit to v2ecoli's own
# ~1000x-scale gap between its measured nucleation_rate=0.00167/s and its
# elongation rate, not a directly measured value).
NUCLEATION_SUPPRESSION_FACTOR = 1000.0

# Number of flagella worth of monomers to provide
N_FLAGELLA = 5


def _safe_name(name):
    """Convert a name to a valid BNG identifier."""
    return name.replace(' ', '_').replace('-', '_')


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
            safe = species.replace(' ', '_').replace('-', '_')
            rates[f'Free_{safe}'] = count / duration

    return rates


def generate_bngl(n_flagella=N_FLAGELLA, k_bind=K_BIND, k_nucleation=K_NUCLEATION, k_completion=K_COMPLETION):
    """Generate the complete BNGL model string."""

    # Ordered reactions (assembly hierarchy)
    reaction_order = [
        'flhDC',
        'flagellar motor switch reaction',
        'flagellar export apparatus reaction 1',
        'flagellar export apparatus reaction 2',
        'flagellar motor reaction',
        'flagellar hook reaction',
        'flagellum reaction',
    ]

    # Parse all reactions
    reactions = {}
    for rxn_name in reaction_order:
        consumed, product = _parse_reaction(rxn_name, COMPLEXATION_STOICHIOMETRY[rxn_name])
        reactions[rxn_name] = {
            'consumed': consumed,
            'product': product,
        }

    # Collect all monomer species (those that are never products of a reaction)
    complex_names = set()
    for rxn in reactions.values():
        complex_names.add(rxn['product'])

    all_consumed = set()
    for rxn in reactions.values():
        all_consumed.update(rxn['consumed'].keys())

    monomer_names = sorted(all_consumed - complex_names)
    complex_names_ordered = [reactions[r]['product'] for r in reaction_order]

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
    lines.append(f'    k_nucleation {k_nucleation}')
    lines.append(f'    k_completion {k_completion}')
    lines.append('')
    # NUCLEATION FIX (2026-08-12): nuc_rate is now a small, fixed fraction of
    # k_bind (see NUCLEATION_SUPPRESSION_FACTOR module docstring for the full
    # reasoning) -- the same rate for every reaction, deliberately much
    # slower than ordinary binding, so existing scaffolds outcompete new ones
    # for the finite monomer supply. Previous target_propensity-based formula
    # (independent of k_bind, and per-reaction-specific) kept per standing
    # preserve-old-code rule:
    #   target_propensity = n_flagella / 50.0
    #   ...
    #   if combinatorial > 0:
    #       nuc_rate = target_propensity / combinatorial
    #   else:
    #       nuc_rate = k_nucleation
    nuc_rate = k_bind / NUCLEATION_SUPPRESSION_FACTOR
    for rxn_name in reaction_order:
        rxn = reactions[rxn_name]
        consumed = rxn['consumed']
        product = rxn['product']
        total = sum(consumed.values())
        if total > 2:
            safe_product = _safe_name(product)
            lines.append(f'    k_nuc_{safe_product}  {nuc_rate:.6e}')
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
            lines.append(f'    # Nucleation (rate scaled by 1/total_subunits)')
            lines.append(f'    {safe_nuc1}() + {safe_nuc2}() -> '
                         f'{scaffold_name}({",".join(init_states)})  {nuc_rate_name}')

            for species in sorted_species:
                safe_sp = _safe_name(species)
                count = consumed[species]

                start = nuc_counts.get(species, 0)

                for i in range(start, count):
                    lines.append(
                        f'    {scaffold_name}({safe_sp}~{i}) + {_safe_name(species)}() -> '
                        f'{scaffold_name}({safe_sp}~{i + 1})  k_bind')

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


def get_model_path():
    """Return path to the generated BNGL model, generating it if needed."""
    path = os.path.join(
        os.path.dirname(__file__), 'flagella_complexation.bngl')
    if not os.path.exists(path):
        write_bngl(path)
    return path


def make_production_document(
    model_file=None,
    n_steps=100,
    complexation_interval=50.0,
    production_interval=1.0,
    production_rate_scale=1.0,
):
    """v2ecoli-local equivalent of pbg_nfsim.composites.make_production_document.

    Added 2026-08-12, part of Maya Abdalla's flagella-cascade investigation.
    pbg_nfsim's own make_production_document hardcodes get_model_path() and
    default_production_rates() from ITS OWN bundled (deliberately left
    generic/uncorrected) generate_flagella_bngl.py -- there's no way to pass
    a custom model_file through it and get matching production rates, since
    the rates always come from pbg_nfsim's own stock stoichiometry
    regardless of which model_file is supplied. This function is otherwise
    identical in structure, but sources BOTH the model file and the
    production rates from THIS module -- the v2ecoli-owned, corrected
    stoichiometry (see COMPLEXATION_STOICHIOMETRY above for the full
    provenance/citations) -- so the two stay consistent. pbg_nfsim is still
    used for the actual runtime process classes (NFSimProcess,
    MonomerProduction) via flagella_nfsim_assembly.py's core_extensions --
    those are genuinely generic engine code, not investigation-specific
    science, and stay as an external dependency.
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
