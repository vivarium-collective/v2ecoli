"""Run a BiRDReactorProcess composite (pbg-bioreactordesign) for a few hours.

Demonstrates that the workspace can actually drive pbg-bioreactordesign end
to end. Saves a JSON snapshot + a matplotlib chart to
`reports/runnable_sims/bird_reactor_<id>.{json,png}`.

This is the REACTOR HALF of the eventual v2ecoli ↔ reactor coupling (mbp-03).
With BiRD's internal Monod biomass ODE active, it's also a complete
self-contained 0D fermentation sim — no cell-side coupling needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter, gather_emitter_results
import pbg_bioreactordesign
from pbg_bioreactordesign import BiRDReactorProcess, make_reactor_document


REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "runnable_sims"


def run_one(label: str, *, sim_hours: float = 12.0, **kwargs) -> dict[str, Any]:
    core = allocate_core()
    # Register the BiRD process + the canonical RAMEmitter
    core.register_link("BiRDReactorProcess", BiRDReactorProcess)
    core.register_link("ram-emitter", RAMEmitter)

    doc = make_reactor_document(**kwargs)
    # The factory uses 'address': 'local:BiRDReactorProcess' for the reactor.
    sim = Composite({"state": doc}, core=core)

    t0 = time.perf_counter()
    sim.run(sim_hours)
    wall = time.perf_counter() - t0

    results = gather_emitter_results(sim) or {}
    # Pick the (single) emitter path's snapshots
    snaps: list[dict[str, Any]] = []
    for _path, rows in results.items():
        if rows:
            snaps = rows
            break

    return {
        "label": label,
        "sim_hours": sim_hours,
        "wall_s": wall,
        "config": kwargs,
        "n_snapshots": len(snaps),
        "snapshots": snaps,
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[bird] pbg-bioreactordesign installed at {pbg_bioreactordesign.__file__}")

    runs = [
        {
            "label": "baseline-bubble-column-20L",
            "kwargs": {
                "reactor_type": "bubble_column",
                "volume_L": 20.0,
                "gas_flow_rate_Lpm": 2.0,
                "temperature_K": 310.15,
                "initial_biomass_gL": 0.5,
                "max_growth_rate_per_h": 0.7,
                "interval": 0.5,
            },
            "sim_hours": 12.0,
        },
        {
            "label": "low-kla-bubble-column-20L",
            "kwargs": {
                "reactor_type": "bubble_column",
                "volume_L": 20.0,
                "gas_flow_rate_Lpm": 0.4,    # lower aeration → lower kLa → DO drops
                "temperature_K": 310.15,
                "initial_biomass_gL": 0.5,
                "max_growth_rate_per_h": 0.7,
                "interval": 0.5,
            },
            "sim_hours": 12.0,
        },
        {
            "label": "stirred-tank-1L",
            "kwargs": {
                "reactor_type": "stirred_tank",
                "volume_L": 1.0,
                "gas_flow_rate_Lpm": 1.0,
                "impeller_power_W": 5.0,
                "temperature_K": 310.15,
                "initial_biomass_gL": 0.5,
                "max_growth_rate_per_h": 0.7,
                "interval": 0.5,
            },
            "sim_hours": 12.0,
        },
    ]

    all_results = []
    for r in runs:
        print(f"[bird] running {r['label']} for {r['sim_hours']} h sim-time…")
        out = run_one(r["label"], sim_hours=r["sim_hours"], **r["kwargs"])
        print(f"[bird]   wall = {out['wall_s']:.2f} s · {out['n_snapshots']} snapshots")
        all_results.append(out)

    out_json = REPORT_DIR / "bird_reactor.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"[bird] wrote {out_json}")

    # Render a small matplotlib chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[bird] matplotlib unavailable; skipping chart")
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for r in all_results:
        snaps = r["snapshots"]
        if not snaps:
            continue
        # Time may come through as a `time` field or as `global_time` — emitter
        # output shape varies. Pick a reasonable time axis.
        t = [s.get("time", s.get("global_time", i * r["config"]["interval"])) for i, s in enumerate(snaps)]
        do = [s.get("dissolved_o2", float("nan")) for s in snaps]
        bm = [s.get("biomass", float("nan")) for s in snaps]
        kl = [s.get("kla_o2", float("nan")) for s in snaps]
        mu = [s.get("specific_growth_rate", float("nan")) for s in snaps]
        axes[0, 0].plot(t, do, label=r["label"], lw=1.5)
        axes[0, 1].plot(t, bm, label=r["label"], lw=1.5)
        axes[1, 0].plot(t, kl, label=r["label"], lw=1.5)
        axes[1, 1].plot(t, mu, label=r["label"], lw=1.5)

    axes[0, 0].set_title("Dissolved O₂ (mg/L)")
    axes[0, 0].set_ylabel("DO [mg/L]")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 1].set_title("Biomass (g/L)")
    axes[0, 1].set_ylabel("X [g/L]")
    axes[1, 0].set_title("kLa_O₂ (1/h)")
    axes[1, 0].set_ylabel("kLa [1/h]")
    axes[1, 0].set_xlabel("time [h]")
    axes[1, 1].set_title("Specific growth rate μ (1/h)")
    axes[1, 1].set_ylabel("μ [1/h]")
    axes[1, 1].set_xlabel("time [h]")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(
        "pbg-bioreactordesign — BiRDReactorProcess (real PBG simulation)",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = REPORT_DIR / "bird_reactor.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[bird] wrote {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
