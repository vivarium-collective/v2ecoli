"""Tests for the __init__ → initialize() and port_defaults → _default migrations.

Validates that:
1. EcoliStep/EcoliProcess base classes call initialize()
2. Subclass initialize() receives parameters and sets instance attrs
3. port_defaults() auto-extracts _default from inputs()
4. _extract_defaults works for nested schemas
"""

import pytest


# ---------------------------------------------------------------------------
# 1. Base class calls initialize()
# ---------------------------------------------------------------------------

def test_ecoli_step_calls_initialize():
    """EcoliStep.__init__ calls initialize(self.parameters)."""
    from v2ecoli.library.ecoli_step import EcoliStep

    class MyStep(EcoliStep):
        initialized = False
        received_config = None

        def initialize(self, config):
            self.initialized = True
            self.received_config = config

    step = MyStep(parameters={"foo": 1})
    assert step.initialized
    assert step.received_config is step.parameters
    assert step.parameters["foo"] == 1


def test_ecoli_process_calls_initialize():
    """EcoliProcess.__init__ calls initialize(self.parameters)."""
    from v2ecoli.library.ecoli_step import EcoliProcess

    class MyProc(EcoliProcess):
        initialized = False
        received_config = None

        def initialize(self, config):
            self.initialized = True
            self.received_config = config

    proc = MyProc(parameters={"bar": 2})
    assert proc.initialized
    assert proc.received_config is proc.parameters
    assert proc.parameters["bar"] == 2


def test_base_initialize_is_noop():
    """Base EcoliStep/EcoliProcess with no override doesn't crash."""
    from v2ecoli.library.ecoli_step import EcoliStep, EcoliProcess

    step = EcoliStep(parameters={"x": 1})
    assert step.parameters["x"] == 1

    proc = EcoliProcess(parameters={"y": 2})
    assert proc.parameters["y"] == 2


# ---------------------------------------------------------------------------
# 2. Concrete process subclass: initialize sets attrs from self.parameters
# ---------------------------------------------------------------------------

def test_unique_update_initialize():
    """UniqueUpdate.initialize() sets self.unique_topo from parameters."""
    from v2ecoli.steps.unique_update import UniqueUpdate

    topo = {"active_ribosome": "unique/active_ribosome"}
    step = UniqueUpdate(parameters={"unique_topo": topo})
    assert step.unique_topo is topo


def test_allocator_initialize():
    """Allocator.initialize() sets molecule/process name dicts."""
    from v2ecoli.steps.allocator import Allocator

    params = {
        "molecule_names": ["ATP[c]", "GTP[c]"],
        "process_names": ["metabolism", "translation"],
        "custom_priorities": {},
        "seed": 42,
    }
    step = Allocator(parameters=params)
    assert step.moleculeNames == ["ATP[c]", "GTP[c]"]
    assert step.n_molecules == 2
    assert step.processNames == ["metabolism", "translation"]
    assert step.n_processes == 2
    assert step.seed == 42


# ---------------------------------------------------------------------------
# 3. PartitionedProcess subclass
# ---------------------------------------------------------------------------

def test_partitioned_process_initialize():
    """PartitionedProcess.initialize sets evolve_only, request_only, etc."""
    from v2ecoli.steps.partition import PartitionedProcess
    import abc

    # Create a concrete subclass to test (PartitionedProcess is abstract)
    class ConcretePartitioned(PartitionedProcess):
        name = "test_partitioned"
        topology = {"bulk": ("bulk",)}
        config_schema = {}

        def inputs(self):
            return {"bulk": "bulk_array"}

        def outputs(self):
            return {"bulk": "bulk_array"}

        def calculate_request(self, timestep, states):
            return {}

        def evolve_state(self, timestep, states):
            return {}

    proc = ConcretePartitioned(parameters={"evolve_only": True})
    assert proc.evolve_only is True
    assert proc.request_only is False
    assert proc.request_set is False


# ---------------------------------------------------------------------------
# 4. Listener subclass
# ---------------------------------------------------------------------------

def test_dna_supercoiling_initialize():
    """DnaSupercoiling.initialize() sets relaxed_DNA_base_pairs_per_turn."""
    from v2ecoli.steps.listeners.dna_supercoiling import DnaSupercoiling

    step = DnaSupercoiling(parameters={
        "relaxed_DNA_base_pairs_per_turn": 10.5
    })
    assert step.relaxed_DNA_base_pairs_per_turn == 10.5


def test_unique_molecule_counts_initialize():
    """UniqueMoleculeCounts.initialize() sets unique_ids."""
    from v2ecoli.steps.listeners.unique_molecule_counts import UniqueMoleculeCounts

    ids = ["active_RNAP", "RNA"]
    step = UniqueMoleculeCounts(parameters={"unique_ids": ids})
    assert step.unique_ids is ids


# ---------------------------------------------------------------------------
# 5. Process subclass with complex init
# ---------------------------------------------------------------------------

def test_protein_degradation_initialize():
    """ProteinDegradation.initialize() builds degradation_matrix."""
    import numpy as np
    from v2ecoli.processes.protein_degradation import ProteinDegradation

    params = {
        "raw_degradation_rate": 0.01,
        "water_id": "WATER[c]",
        "amino_acid_ids": ["ALA[c]", "GLY[c]"],
        "amino_acid_counts": np.array([[2, 3], [1, 4]]),
        "protein_ids": ["PROT1", "PROT2"],
        "protein_lengths": np.array([10, 15]),
        "seed": 42,
    }
    proc = ProteinDegradation(parameters=params)
    assert proc.raw_degradation_rate == 0.01
    assert proc.degradation_matrix.shape == (3, 2)  # 2 AAs + water x 2 proteins
    assert proc.seed == 42


# ---------------------------------------------------------------------------
# 6. GlobalClock (no __init__ or initialize - just test it works)
# ---------------------------------------------------------------------------

def test_global_clock_no_init():
    """GlobalClock has no initialize, should construct without error."""
    from v2ecoli.steps.global_clock import GlobalClock
    clock = GlobalClock()
    assert clock.name == "global_clock"


# ---------------------------------------------------------------------------
# 7. Inheritance chain: initialize is NOT called twice
# ---------------------------------------------------------------------------

def test_initialize_called_once():
    """Verify initialize() is called exactly once during construction."""
    from v2ecoli.library.ecoli_step import EcoliStep

    class CountingStep(EcoliStep):
        call_count = 0

        def initialize(self, config):
            CountingStep.call_count += 1

    CountingStep.call_count = 0
    step = CountingStep(parameters={})
    assert CountingStep.call_count == 1


# ---------------------------------------------------------------------------
# 8. _extract_defaults helper
# ---------------------------------------------------------------------------

def test_extract_defaults_simple():
    """_extract_defaults picks up _default from dict-style type specs."""
    from v2ecoli.library.ecoli_step import _extract_defaults

    schema = {
        'bulk': {'_type': 'bulk_array', '_default': []},
        'timestep': {'_type': 'integer', '_default': 1},
        'plain': 'float',  # no default
    }
    result = _extract_defaults(schema)
    assert result == {'bulk': [], 'timestep': 1}
    assert 'plain' not in result


def test_extract_defaults_nested():
    """_extract_defaults recurses into nested port dicts."""
    from v2ecoli.library.ecoli_step import _extract_defaults

    schema = {
        'listeners': {
            'mass': {
                'cell_mass': {'_type': 'float[fg]', '_default': 0.0},
                'dry_mass': 'float',  # no default
            },
        },
        'global_time': {'_type': 'float', '_default': 0.0},
    }
    result = _extract_defaults(schema)
    assert result == {
        'listeners': {'mass': {'cell_mass': 0.0}},
        'global_time': 0.0,
    }


def test_extract_defaults_empty():
    """_extract_defaults returns {} for schemas with no defaults."""
    from v2ecoli.library.ecoli_step import _extract_defaults
    assert _extract_defaults({}) == {}
    assert _extract_defaults({'x': 'float', 'y': 'integer'}) == {}


# ---------------------------------------------------------------------------
# 9. port_defaults() auto-extraction from inputs()
# ---------------------------------------------------------------------------

def test_port_defaults_auto_from_inputs():
    """Base port_defaults() returns _default values from inputs()."""
    from v2ecoli.library.ecoli_step import EcoliStep

    class MyStep(EcoliStep):
        def inputs(self):
            return {
                'bulk': {'_type': 'bulk_array', '_default': []},
                'timestep': {'_type': 'integer', '_default': 1},
                'other': 'float',
            }

    step = MyStep(parameters={})
    defaults = step.port_defaults()
    assert defaults == {'bulk': [], 'timestep': 1}


def test_port_defaults_concrete_class():
    """DnaSupercoiling.port_defaults() auto-extracts from its inputs()."""
    from v2ecoli.steps.listeners.dna_supercoiling import DnaSupercoiling

    step = DnaSupercoiling(parameters={
        "relaxed_DNA_base_pairs_per_turn": 10.5,
    })
    defaults = step.port_defaults()
    assert 'chromosomal_segments' in defaults
    assert defaults['chromosomal_segments'] == []
    assert defaults['global_time'] == 0.0
    assert defaults['timestep'] == 1.0


def test_port_defaults_protein_degradation():
    """ProteinDegradation.port_defaults() auto-extracts from inputs()."""
    import numpy as np
    from v2ecoli.processes.protein_degradation import ProteinDegradation

    params = {
        "raw_degradation_rate": 0.01,
        "water_id": "WATER[c]",
        "amino_acid_ids": ["ALA[c]", "GLY[c]"],
        "amino_acid_counts": np.array([[2, 3], [1, 4]]),
        "protein_ids": ["PROT1", "PROT2"],
        "protein_lengths": np.array([10, 15]),
        "seed": 42,
    }
    proc = ProteinDegradation(parameters=params)
    defaults = proc.port_defaults()
    assert defaults == {'bulk': [], 'timestep': 1}


def test_port_defaults_tf_unbinding():
    """TfUnbinding.port_defaults() auto-extracts from inputs()."""
    from v2ecoli.processes.tf_unbinding import TfUnbinding

    proc = TfUnbinding(parameters={
        "tf_ids": ["a", "b"],
        "submass_indices": {"protein": 0},
        "active_tf_masses": [1.0, 2.0],
    })
    defaults = proc.port_defaults()
    assert defaults['bulk'] == []
    assert defaults['promoters'] == []
    assert defaults['global_time'] == 0.0
    assert defaults['timestep'] == 1
    assert defaults['next_update_time'] == 1.0


# ---------------------------------------------------------------------------
# 10. Feature module infrastructure
# ---------------------------------------------------------------------------

def test_build_execution_layers_base():
    """Base layers exclude optional feature steps."""
    from v2ecoli.generate import build_execution_layers
    layers = build_execution_layers([])
    flat = [s for layer in layers for s in layer]
    assert 'dna_supercoiling_listener' not in flat
    assert 'dna-supercoiling-step' not in flat
    assert 'ecoli-chromosome-structure' in flat


def test_build_execution_layers_supercoiling():
    """Supercoiling feature adds step after chromosome-structure and listener."""
    from v2ecoli.generate import build_execution_layers
    layers = build_execution_layers(['supercoiling'])
    flat = [s for layer in layers for s in layer]
    assert 'dna-supercoiling-step' in flat
    assert 'dna_supercoiling_listener' in flat
    cs_idx = flat.index('ecoli-chromosome-structure')
    sc_idx = flat.index('dna-supercoiling-step')
    assert sc_idx > cs_idx


def test_build_execution_layers_ppgpp():
    """ppGpp feature adds step before transcript-initiation."""
    from v2ecoli.generate import build_execution_layers
    layers = build_execution_layers(['ppgpp_regulation'])
    flat = [s for layer in layers for s in layer]
    assert 'ppgpp-initiation' in flat
    ti_idx = flat.index('ecoli-transcript-initiation_requester')
    pp_idx = flat.index('ppgpp-initiation')
    assert pp_idx < ti_idx


def test_build_execution_layers_unknown_feature():
    """Unknown feature names are silently ignored."""
    from v2ecoli.generate import build_execution_layers
    layers = build_execution_layers(['nonexistent_feature'])
    flat = [s for layer in layers for s in layer]
    assert 'ecoli-chromosome-structure' in flat  # base still works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
