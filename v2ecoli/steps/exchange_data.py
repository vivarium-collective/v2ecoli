from typing import Any

from v2ecoli.steps.base import V2Step as Step
from vivarium.library.units import units


class ExchangeData(Step):
    """
    Update metabolism exchange constraints according to environment concs.
    """

    name = "exchange_data"
    config_schema = {}
    topology = {
        "boundary": ("boundary",),
        "environment": ("environment",),
    }

    defaults: dict[str, Any] = {
        "external_state": None,
        "environment_molecules": [],
        "env_to_exchange_map": {},
        "secretion_exchange_molecules": [],
        "import_constraint_threshold": 1e-5,
        "carbon_sources": [],
        "time_step": 1,
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.environment_molecules = self.parameters.get("environment_molecules", [])

        # Support both old format (external_state instance) and new format (extracted attrs)
        es = self.parameters.get("external_state")
        if es is not None and hasattr(es, 'exchange_data_from_concentrations'):
            self._exchange_data_from_conc = es.exchange_data_from_concentrations
            self._import_threshold = es.import_constraint_threshold
        else:
            self._env_to_exchange_map = self.parameters.get("env_to_exchange_map", {})
            self._secretion_set = set(self.parameters.get("secretion_exchange_molecules", []))
            self._import_threshold = self.parameters.get("import_constraint_threshold", 1e-5)
            self._carbon_sources = self.parameters.get("carbon_sources", [])
            self._exchange_data_from_conc = self._exchange_from_conc

    def _exchange_from_conc(self, molecules):
        """Standalone exchange_data_from_concentrations (no external_state needed)."""
        exchange_molecules = {
            self._env_to_exchange_map[mol]: conc
            for mol, conc in molecules.items()
            if mol in self._env_to_exchange_map
        }
        threshold = self._import_threshold
        importUnconstrained = {
            mol_id for mol_id, conc in exchange_molecules.items()
            if conc >= threshold
        }
        externalExchange = set(importUnconstrained)
        importExchange = set(importUnconstrained)

        importConstrained = {}
        oxygen_id = "OXYGEN-MOLECULE[p]"
        for cs_id in self._carbon_sources:
            if cs_id in importUnconstrained:
                if oxygen_id in importUnconstrained:
                    importConstrained[cs_id] = 20.0 * (units.mmol / units.g / units.h)
                else:
                    importConstrained[cs_id] = 100.0 * (units.mmol / units.g / units.h)
                importUnconstrained.remove(cs_id)

        externalExchange.update(self._secretion_set)
        return {
            "externalExchangeMolecules": externalExchange,
            "importExchangeMolecules": importExchange,
            "importConstrainedExchangeMolecules": importConstrained,
            "importUnconstrainedExchangeMolecules": importUnconstrained,
            "secretionExchangeMolecules": self._secretion_set,
        }

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
        env_concs = {
            mol: states["boundary"]["external"][mol]
            for mol in self.environment_molecules
        }

        # Convert threshold to match env_concs units
        saved_threshold = self._import_threshold
        self._import_threshold = saved_threshold * units.mM if not hasattr(saved_threshold, 'asNumber') else saved_threshold * units.mM / units.mM * units.mM
        exchange_data = self._exchange_data_from_conc(env_concs)
        self._import_threshold = saved_threshold

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
