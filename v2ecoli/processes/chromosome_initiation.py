"""
======================
Chromosome Initiation
======================

Lightweight process-bigraph stubs for chromosome-replication initiation and
partitioning.  These classes inherit directly from ``process_bigraph.Process``
(not from the heavier ``EcoliStep``/``EcoliProcess`` adapters) so they import
cleanly even without the vEcoli parameter-calculator or the vendored Cython
extensions.  They are therefore the entry-points that bigraph-schema's
``discover_packages()`` registers under the ``v2ecoli`` distribution.

Real DnaA-oriC binding and chromosome partitioning logic lives in
``chromosome_replication.py`` (the ``ChromosomeReplication`` Step that wraps the
full vEcoli implementation).  These stubs are here for the discovery hook and
for consumers that need a lightweight handle to the process graph shape without
pulling in the full simulation stack.
"""
from process_bigraph import Process


class DnaABinder(Process):
    """DnaA-oriC binding dynamics.

    Models the cooperative binding of DnaA-ATP to oriC that triggers
    chromosome replication initiation.  The full implementation accumulates
    free DnaA and flips ``oriC_state`` from ``'free'`` to ``'bound'`` once
    the occupancy threshold is reached.

    (Skeleton — full binding kinetics TBD.)
    """

    config_schema = {
        'binding_rate': 'float',
        'unbinding_rate': 'float',
    }

    def inputs(self):
        return {
            'free_DnaA': 'float',
            'oriC_state': 'string',
        }

    def outputs(self):
        return {
            'free_DnaA': 'float',
            'oriC_state': 'string',
        }

    def update(self, state, interval):
        # TODO: real DnaA-oriC binding dynamics
        return {
            'free_DnaA': 0.0,
            'oriC_state': state.get('oriC_state', 'free'),
        }


class ChromosomePartition(Process):
    """Chromosome partitioning at cell division.

    Ensures that each daughter cell receives exactly one complete chromosome
    copy after replication terminates.  Coordinates with the MukBEF
    condensin machinery and the MatP/ter-linkage anchoring system.

    (Skeleton — partitioning logic TBD.)
    """

    config_schema = {
        'partition_method': 'string',
    }

    def inputs(self):
        return {'chromosome': 'map[float]'}

    def outputs(self):
        return {'chromosome': 'map[float]'}

    def update(self, state, interval):
        # TODO: real partitioning logic
        return {'chromosome': state.get('chromosome', {})}
