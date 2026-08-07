"""Generate per-condition MetabolismRedux fork configs from cond_<X>.json + the
basal redux swap block. Deterministic; committed to the vEcoli fork."""
from __future__ import annotations
import json, os

CONDITIONS = ["with_aa", "succinate", "no_oxygen", "acetate"]  # basal already exists
SWAP_KEYS = ["swap_processes", "exclude_processes", "flow", "raw_output",
             "strip_pint_ports", "attach_pint_ports", "output_ports"]

def build_redux_config(cond_config: dict, basal_redux: dict) -> dict:
    out = dict(cond_config)  # start from the condition config (media/condition/nutrients)
    for k in SWAP_KEYS:
        if k in basal_redux:
            out[k] = basal_redux[k]
    cond = out.get("condition") or cond_config.get("condition")
    out["condition"] = cond
    out["experiment_id"] = f"metabolism_redux_{cond}"
    return out

def main(fork_dir: str = "/Users/eranagmon/code/vEcoli") -> list[str]:
    cfg = os.path.join(fork_dir, "configs")
    with open(os.path.join(cfg, "metabolism_redux_basal.json"), encoding="utf-8") as f:
        basal_redux = json.load(f)
    written = []
    for c in CONDITIONS:
        with open(os.path.join(cfg, f"cond_{c}.json"), encoding="utf-8") as f:
            cond_config = json.load(f)
        out = build_redux_config(cond_config, basal_redux)
        p = os.path.join(cfg, f"metabolism_redux_{c}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)
        written.append(p)
    return written

if __name__ == "__main__":
    for p in main():
        print("wrote", p)
