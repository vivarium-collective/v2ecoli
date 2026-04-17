from v2ecoli.library.unit_bridge import unum_to_pint
from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.quantity import ureg as units
from v2ecoli.types.stores import InPlaceDict, ListenerStore


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

    def initialize(self, config):
        self.parameters = config or {}
        self.external_state = self.parameters.get("external_state")
        self.environment_molecules = self.parameters.get("environment_molecules", [])

    def inputs(self):
        return {
            "boundary": InPlaceDict(),
            "environment": ListenerStore(),
        }

    def outputs(self):
        return {
            "boundary": InPlaceDict(),
            "environment": ListenerStore(),
        }

    def next_update(self, timestep, states):
        # Set exchange constraints for metabolism. Convert any unit-bearing
        # values (Unum or pint) to plain float magnitudes in their native unit
        # to avoid cross-registry comparison errors after dill round-trips.
        env_concs = {}
        for mol in self.environment_molecules:
            q = unum_to_pint(states["boundary"]["external"][mol])
            env_concs[mol] = float(q.magnitude) if hasattr(q, "magnitude") else float(q)

        threshold = unum_to_pint(self.external_state.import_constraint_threshold)
        self.external_state.import_constraint_threshold = (
            float(threshold.magnitude) if hasattr(threshold, "magnitude") else float(threshold)
        )
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
