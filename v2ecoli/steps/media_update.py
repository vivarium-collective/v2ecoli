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
        self.curr_media_id = self.parameters.get("media_id", "minimal")

    def inputs(self):
        return {"boundary": InPlaceDict(), "environment": InPlaceDict()}

    def outputs(self):
        return {"boundary": InPlaceDict(), "environment": InPlaceDict()}

    def next_update(self, timestep, states):
        if states["environment"]["media_id"] == self.curr_media_id:
            return {}

        self.curr_media_id = states["environment"]["media_id"]
        env_concs = self.saved_media[self.curr_media_id]
        # ABSOLUTE write, not a delta. boundary.external leaves are declared
        # `overwrite[float[mM]]` (metabolism.inputs), so the apply REPLACES and
        # the new media's concentration is simply the new value.
        #
        # This used to compute `conc - current` and guard the result with
        # `isnan`, which covered only the inf -> inf transition (inf - inf is
        # NaN). The inf -> FINITE transition produced -inf, passed the guard,
        # and then accumulated to NaN in the additive store -- the same defect
        # #548 fixed in EnvironmentMirror, never applied here. Writing the
        # target concentration directly removes the inf arithmetic rather than
        # guarding it, so both transitions are now correct by construction.
        conc_update = dict(env_concs)
        return {"boundary": {"external": conc_update}}

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
