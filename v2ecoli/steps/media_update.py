import numpy as np
from process_bigraph import Step
from vivarium.library.units import units


class MediaUpdate(Step):
    """
    Update environment concentrations according to current media ID.
    """

    name = "media_update"
    config_schema = {}
    topology = {
        "boundary": ("boundary",),
        "environment": ("environment",),
    }

    defaults = {"saved_media": {}, "time_step": 1, "media_id": "minimal"}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.saved_media = {}
        for media_id, env_concs in self.parameters.get("saved_media", {}).items():
            self.saved_media[media_id] = {}
            for env_mol in env_concs.keys():
                self.saved_media[media_id][env_mol] = env_concs[env_mol] * units.mM
        self.zero_diff = 0 * units.mM
        self.curr_media_id = self.parameters.get("media_id", "minimal")

    def ports_schema(self):
        return {
            "boundary": {"external": {"*": {"_default": 0 * units.mM}}},
            "environment": {"media_id": {"_default": ""}},
        }

    def next_update(self, timestep, states):
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
