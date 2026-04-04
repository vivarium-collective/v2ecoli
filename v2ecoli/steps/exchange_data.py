from typing import Any

from process_bigraph import Step
from v2ecoli.library.units import units


class ExchangeData(Step):
    """
    Update metabolism exchange constraints according to environment concs.
    """

    name = "exchange_data"
    config_schema = {}

    defaults: dict[str, Any] = {
        "external_state": None,
        "environment_molecules": [],
        "saved_media": {},
        "time_step": 1,
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.external_state = self.parameters.get("external_state")
        self.environment_molecules = self.parameters.get("environment_molecules", [])

    def ports_schema(self):
        return {
            "boundary": {"external": {"*": {"_default": 0 * units.mM}}},
            "environment": {
                "exchange_data": {
                    "constrained": {"_default": {}, "_updater": "set"},
                    "unconstrained": {"_default": set(), "_updater": "set"},
                }
            },
        }

    def next_update(self, timestep, states):
        # Set exchange constraints for metabolism
        env_concs = {
            mol: states["boundary"]["external"][mol]
            for mol in self.environment_molecules
        }

        # Converting threshold is faster than converting all of env_concs
        self.external_state.import_constraint_threshold *= units.mM
        exchange_data = self.external_state.exchange_data_from_concentrations(env_concs)
        self.external_state.import_constraint_threshold = (
            self.external_state.import_constraint_threshold.magnitude
        )

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
