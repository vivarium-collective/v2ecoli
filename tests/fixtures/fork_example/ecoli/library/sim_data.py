"""Minimal stand-in for a fork's ``ecoli.library.sim_data``.

Exists so ``build_fork_config`` can be tested hermetically: the config it returns
carries ``fork_only_key``, which no installed vEcoli has. A config containing that
key therefore PROVES the getter was resolved from this fixture fork and not from
whatever ``ecoli`` happens to be installed in site-packages.
"""


class LoadSimData:
    def __init__(self, sim_data_path=None, **kwargs):
        self.sim_data_path = sim_data_path

    def get_config_by_name(self, name, time_step=1):
        if name != "example-secretion":
            raise KeyError(
                f"Process of name {name} is not known to LoadSimData.get_config_by_name")
        return {"rate": 1.0, "fork_only_key": "present", "time_step": time_step}
