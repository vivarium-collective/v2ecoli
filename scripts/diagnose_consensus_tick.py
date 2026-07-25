"""Diagnostic: run one tick of consensus_baseline without swallowing the
exception. Prints the full traceback so we can localize the AssertionError
that's killing Stage A.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    from process_bigraph import Composite
    from v2ecoli.composites.consensus_baseline import consensus_baseline
    from v2ecoli.core import build_core

    core = build_core()
    print("[diagnose] building composite…", flush=True)
    doc = consensus_baseline(
        core=core, seed=0, cache_dir="out/cache",
        config_overrides={
            "ecoli-metabolism.media_id": "minimal",
        },
    )
    composite = Composite(doc, core=core)
    n_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"[diagnose] composite built; running {n_ticks} ticks one at a time…",
          flush=True)
    for i in range(1, n_ticks + 1):
        try:
            composite.run(interval=1.0)
            print(f"[diagnose] tick {i} OK", flush=True)
        except Exception:
            print(f"[diagnose] EXCEPTION at tick {i}:", flush=True)
            traceback.print_exc()
            sys.exit(1)
    print(f"[diagnose] {n_ticks} ticks OK — no error", flush=True)


if __name__ == "__main__":
    main()
