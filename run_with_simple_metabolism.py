"""
Run the workflow report with SimplifiedMetabolism instead of the full Metabolism.

Approach:
  1. Build a translated config that maps the existing Metabolism config
     fields to what SimplifiedMetabolism expects.
  2. Monkey-patch generate.py's Metabolism import to SimplifiedMetabolism
     and inject the translated config.
  3. Run workflow.py to generate the report.

SimplifiedMetabolism uses direct scipy LP instead of the full wholecell
FluxBalanceAnalysis framework. Expected to be faster but less accurate.
"""

import sys
import os


def translate_config(full_config):
    """Translate standard Metabolism config to SimplifiedMetabolism format."""
    simple = {
        # Direct mappings (same field names)
        'catalyst_ids': list(full_config.get('catalyst_ids', [])),
        'exchange_molecules': list(full_config.get('exchange_molecules', [])),
        'avogadro': full_config.get('avogadro', 6.022e23),
        'cell_density': full_config.get('cell_density', 1100.0),
        'ngam': full_config.get('ngam', 8.39),
        'seed': full_config.get('seed', 0),
        'time_step': full_config.get('time_step', 1),
        'maintenance_reaction': full_config.get('maintenance_reaction', {}),
        'exchange_data_from_media': full_config.get('exchange_data_from_media'),
        'media_id': full_config.get('media_id', 'minimal'),
        'nutrientToDoublingTime': full_config.get('nutrientToDoublingTime', {}),
        'get_biomass_as_concentrations': full_config.get(
            'get_biomass_as_concentrations'),
        'dark_atp': 33.565,
        'cell_dry_mass_fraction': 0.3,

        # Rename: stoichiometry → reaction_stoich
        'reaction_stoich': full_config.get('stoichiometry', {}),

        # Empty defaults for fields not in standard config
        'homeostatic_targets': {},
        'reactions_with_catalyst': [],
        'catalysis_matrix_I': [],
        'catalysis_matrix_J': [],
        'catalysis_matrix_V': [],
        'output_molecule_ids': [],
        'molecule_masses': {},
    }
    return simple


def patch_generate():
    """Monkey-patch generate.py to use SimplifiedMetabolism."""
    import v2ecoli.generate as gen
    from v2ecoli.processes.metabolism_simple import SimplifiedMetabolism

    # Wrap _instantiate_step to swap Metabolism with SimplifiedMetabolism
    original_instantiate = gen._instantiate_step

    def patched(step_name, config, loader, core, process_cache=None):
        if step_name == 'ecoli-metabolism':
            from v2ecoli.generate import _make_instance
            simple_config = translate_config(config)
            instance = _make_instance(SimplifiedMetabolism, simple_config, core)
            topology = {
                'bulk': ('bulk',),
                'bulk_total': ('bulk',),
                'listeners': ('listeners',),
                'environment': ('environment',),
                'polypeptide_elongation': ('process_state', 'polypeptide_elongation'),
                'global_time': ('global_time',),
                'timestep': ('timestep',),
                'next_update_time': ('next_update_time', 'metabolism'),
            }
            return instance, topology, 'step'
        return original_instantiate(step_name, config, loader, core, process_cache)

    gen._instantiate_step = patched
    print('[patch] Metabolism -> SimplifiedMetabolism')


if __name__ == '__main__':
    patch_generate()
    import workflow
    workflow.run_workflow()
