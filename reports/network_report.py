"""Standalone composition-diagram viewer for a v2ecoli composite.

Loads one architecture, builds the execution layers, extracts the
composition graph, and renders an HTML page with the interactive Cytoscape
network. The rendering is owned by
``v2ecoli.visualizations.network.NetworkVisualization``; this script handles
CLI args + composite construction + writing the HTML.

Usage:
    python reports/network_report.py                          # baseline (default)
    python reports/network_report.py --model departitioned
    python reports/network_report.py --model reconciled
    python reports/network_report.py --model baseline --no-open
    python reports/network_report.py --out out/network_baseline.html
"""

import argparse
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")


MODELS = {
    "baseline":      "Baseline (partitioned)",
    "departitioned": "Departitioned",
    "reconciled":    "Reconciled",
}


def _build_layers(model: str) -> list[list[str]]:
    """Build the execution-layer ordering for the chosen architecture."""
    if model == "baseline":
        from v2ecoli.composites.baseline import build_execution_layers, DEFAULT_FEATURES
        return build_execution_layers(DEFAULT_FEATURES)
    if model == "departitioned":
        from v2ecoli.composites.departitioned import build_execution_layers, DEFAULT_FEATURES
        return build_execution_layers(DEFAULT_FEATURES)
    if model == "reconciled":
        from v2ecoli.composites.reconciled import build_execution_layers, DEFAULT_FEATURES
        return build_execution_layers(DEFAULT_FEATURES)
    raise ValueError(f"unknown model: {model!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS.keys(), default="baseline")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (default: out/reports/network_<model>.html)")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default: 0)")
    parser.add_argument("--cache-dir", default="out/cache",
                        help="ParCa cache directory (default: out/cache)")
    args = parser.parse_args()

    from v2ecoli import build_composite
    from v2ecoli.visualizations._helpers import build_graph
    from v2ecoli.visualizations.network import NetworkVisualization

    composite = build_composite(args.model, seed=args.seed, cache_dir=args.cache_dir)
    layers = _build_layers(args.model)
    spec = build_graph(composite, layers)
    spec["architecture"] = args.model

    out_path = args.out or f"out/reports/network_{args.model}.html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    viz = NetworkVisualization(
        config={
            "title": f"v2ecoli network — {MODELS[args.model]}",
            "subtitle": args.model,
        },
        core=composite.core,
    )
    result = viz.update({"composite_spec": spec})
    with open(out_path, "w") as f:
        f.write(result["html"])
    print(f"wrote {out_path}")
    if not args.no_open and sys.platform == "darwin":
        subprocess.Popen(["open", out_path])


if __name__ == "__main__":
    main()
