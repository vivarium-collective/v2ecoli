def _state(per_obs, name="basal", seeds=1, gens=4, variant=0, config=None):
    return {"name": name, "condition": "basal", "seeds": seeds, "generations": gens,
            "variant": variant, "observables": per_obs, "plot_trajs": {},
            "v2_bounds": [], "config": config or {}, "v2_dir": "", "ve_dir": ""}


def _run_card(name, state):
    from v2ecoli.core import build_core
    from scripts._compare.report_cards import REPORT_CARD_STEPS
    core = build_core()
    core.register_links(REPORT_CARD_STEPS)
    step = core.link_registry[f"{name}_report_card"](config={}, core=core)
    return step.update(state)
