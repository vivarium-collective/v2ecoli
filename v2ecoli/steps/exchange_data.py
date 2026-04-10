from v2ecoli.steps.base import V2Step as Step
from v2ecoli.library.unit_defs import units


class ExchangeData(Step):
    """
    Update metabolism exchange constraints according to environment concs.
    """

    name = "exchange_data"
    config_schema = {
        "external_state": {"_default": None},
        "environment_molecules": {"_default": []},
        "saved_media": {"_default": {}},
        "time_step": {"_default": 1},
    }
    topology = {
        "boundary": ("boundary",),
        "environment": ("environment",),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.external_state = self.parameters.get("external_state")
        self.environment_molecules = self.parameters.get("environment_molecules", [])

    def inputs(self):
        from v2ecoli.types.stores import InPlaceDict, ListenerStore
        return {
            "boundary": InPlaceDict(),
            "environment": ListenerStore(),
        }

    def outputs(self):
        from v2ecoli.types.stores import InPlaceDict, ListenerStore
        return {
            "boundary": InPlaceDict(),
            "environment": ListenerStore(),
        }

    def next_update(self, timestep, states):
        # Set exchange constraints for metabolism
        # Convert pint Quantities to plain float magnitudes (in mM) to avoid
        # cross-registry comparison errors when state is deserialized from dill
        env_concs = {}
        for mol in self.environment_molecules:
            val = states["boundary"]["external"][mol]
            if hasattr(val, 'magnitude'):
                env_concs[mol] = float(val.magnitude)
            elif hasattr(val, 'asNumber'):
                env_concs[mol] = float(val.asNumber())
            else:
                env_concs[mol] = float(val)

        # Ensure threshold is a plain float for comparison
        threshold = self.external_state.import_constraint_threshold
        if hasattr(threshold, 'magnitude'):
            self.external_state.import_constraint_threshold = float(threshold.magnitude)
        elif hasattr(threshold, 'asNumber'):
            self.external_state.import_constraint_threshold = float(threshold.asNumber())
        else:
            self.external_state.import_constraint_threshold = float(threshold)
        exchange_data = self.external_state.exchange_data_from_concentrations(env_concs)

        unconstrained = exchange_data["importUnconstrainedExchangeMolecules"]
        constrained = exchange_data["importConstrainedExchangeMolecules"]
        return {
            "environment": {
                "exchange_data": {
                    "constrained": constrained,
                    "unconstrained": list(unconstrained),
                }
            }
        }

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
