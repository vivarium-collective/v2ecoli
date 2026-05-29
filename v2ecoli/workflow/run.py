"""Driver + CLI for v2ecoli workflow sweeps.

Loads a vEcoli-style JSON config, builds the meta-composite, and ticks it
until every branch reports ``complete`` (or a global wall/sim cap is hit).
Saves the sweep document as a .pbg for inspection.

    v2ecoli-workflow --config configs/two_generations.json --out out/two_gen
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from process_bigraph import Composite

from v2ecoli.core import build_core
from v2ecoli.workflow.config import load_config_with_inheritance
from v2ecoli.workflow.meta_composite import (
    build_meta_composite, register_workflow_processes)


def _all_complete(composite) -> bool:
    branches = composite.state.get("branches", {})
    return bool(branches) and all(
        b.get("complete") for b in branches.values())


def run_workflow(config: dict[str, Any], *, max_total_time: float = 1e9,
                 pbg_out: str | None = None) -> dict[str, Any]:
    """Build and run the meta-composite for ``config``. Returns a result dict."""
    core = build_core()
    register_workflow_processes(core)

    doc = build_meta_composite(config)
    composite = Composite(doc, core=core)

    dt = float(config.get("time_step", 1.0))
    elapsed = 0.0
    while not _all_complete(composite) and elapsed < max_total_time:
        composite.run(dt)
        elapsed += dt

    if pbg_out:
        from v2ecoli.pbg import save_pbg_doc
        save_pbg_doc(composite.state, pbg_out)

    branches = composite.state.get("branches", {})
    return {
        "complete": _all_complete(composite),
        "elapsed": elapsed,
        "branches": {k: {"complete": v.get("complete"), "summary": v.get("summary")}
                     for k, v in branches.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a workflow config JSON.")
    parser.add_argument("--out", default=None,
                        help="Output dir for parquet + sweep.pbg (default: out/<experiment_id>).")
    parser.add_argument("--build-only", action="store_true",
                        help="Build + save the .pbg without running.")
    parser.add_argument("--max-total-time", type=float, default=1e9,
                        help="Global sim-time safety cap (seconds).")
    args = parser.parse_args()

    config = load_config_with_inheritance(args.config)
    exp = config.get("experiment_id") or "workflow"
    out_dir = args.out or os.path.join("out", exp)
    config.setdefault("out_dir", os.path.join(out_dir, "parquet"))
    os.makedirs(out_dir, exist_ok=True)
    pbg_out = os.path.join(out_dir, "sweep.pbg")

    if args.build_only:
        core = build_core()
        register_workflow_processes(core)
        composite = Composite(build_meta_composite(config), core=core)
        from v2ecoli.pbg import save_pbg_doc
        save_pbg_doc(composite.state, pbg_out)
        print(f"Built {len(composite.state['branches'])} branches -> {pbg_out}")
        return

    result = run_workflow(config, max_total_time=args.max_total_time, pbg_out=pbg_out)
    n = len(result["branches"])
    done = sum(1 for b in result["branches"].values() if b["complete"])
    print(f"Sweep '{exp}': {done}/{n} branches complete in {result['elapsed']:.0f} s sim.")
    print(f"Saved: {pbg_out}")


if __name__ == "__main__":
    main()
