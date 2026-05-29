import numpy as np
from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.quantity import ureg as units
from v2ecoli.types.stores import InPlaceDict


class MediaUpdate(Step):
    """
    Update environment concentrations according to current media ID.
    """

    name = "media_update"
    config_schema = {
        "saved_media": {"_default": {}},
        "time_step": {"_default": 1},
        "media_id": {"_default": "minimal"},
    }
    topology = {
        "boundary": ("boundary",),
        "environment": ("environment",),
    }

    def initialize(self, config):
        self.parameters = config or {}
        self.saved_media = {}
        for media_id, env_concs in self.parameters.get("saved_media", {}).items():
            self.saved_media[media_id] = {}
            for env_mol in env_concs.keys():
                self.saved_media[media_id][env_mol] = env_concs[env_mol] * units.mM
        self.zero_diff = 0 * units.mM
        self.curr_media_id = self.parameters.get("media_id", "minimal")

    def inputs(self):
        return {"boundary": InPlaceDict(), "environment": InPlaceDict()}

    def outputs(self):
        return {"boundary": InPlaceDict(), "environment": InPlaceDict()}

    def next_update(self, timestep, states):
        # Driver-supplied path: when EnvironmentDriver (or a reactor coupler in
        # mbp-03) populates environment.external_concentrations, propagate to
        # boundary.external each step regardless of media_id. The driver path
        # is opt-in by composition — empty (or absent) dict means "no driver,
        # fall through to media_id semantics" and the baseline composite is
        # byte-identical to pre-mbp-01 (regression-guarded by
        # static-env-baseline-unchanged).
        driver_concs = states["environment"].get("external_concentrations") or {}
        if driver_concs:
            boundary_ext = states["boundary"]["external"]
            conc_update = {}
            for mol, conc in driver_concs.items():
                # Driver writes pint mM Quantities (see EnvironmentDriver);
                # tolerate bare floats by promoting to mM.
                if not hasattr(conc, "magnitude"):
                    conc = conc * units.mM
                curr = boundary_ext.get(mol)
                if curr is None:
                    # metabolism doesn't track this molecule — skip silently
                    continue
                diff = conc - curr
                if np.isnan(diff):
                    diff = self.zero_diff
                conc_update[mol] = diff
            if conc_update:
                return {"boundary": {"external": conc_update}}
            return {}

        # Static / media-id-transition path (original behavior).
        if states["environment"]["media_id"] == self.curr_media_id:
            return {}

        self.curr_media_id = states["environment"]["media_id"]
        env_concs = self.saved_media[self.curr_media_id]
        conc_update = {}
        # Calculate concentration delta to get from environment specified
        # by old media ID to the one specified by the current media ID
        for mol, conc in env_concs.items():
            diff = conc - states["boundary"]["external"][mol]
            # Arithmetic with np.inf gets messy
            if np.isnan(diff):
                diff = self.zero_diff
            conc_update[mol] = diff
        return {"boundary": {"external": conc_update}}

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
