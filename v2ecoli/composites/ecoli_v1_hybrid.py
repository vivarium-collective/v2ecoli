"""ecoli_v1_hybrid — the v2ecoli whole-cell port that HYBRIDIZES with the v1
(upstream vEcoli) fork: its injected metabolism-redux is sourced from the fork
(injected_processes.fork_repo), unlike native ecoli_baseline. Shares the whole
generator body with ecoli_baseline; only the native policy differs.
"""
from typing import Any

from viva_superpowers.composite_generator import composite_generator
from v2ecoli.composites.ecoli_baseline import baseline as _baseline, WCM_PARAMETERS


@composite_generator(
    name="ecoli_v1_hybrid",
    description="55-process whole-cell E. coli port, injections sourced from the v1 (vEcoli) fork",
    parameters=WCM_PARAMETERS,
)
def ecoli_v1_hybrid(core: Any = None, **kwargs) -> dict:
    kwargs.pop("native", None)
    return _baseline(core, native=False, **kwargs)
