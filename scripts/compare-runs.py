#!/usr/bin/env python3
"""Compare two recorded runs by id. Stub for v1; full impl arrives later.

Guarded behind ``if __name__ == "__main__":`` so importing this module
during composite-registry discovery (``pbg_superpowers.composite_generator
.discover_generators`` walks every subpackage) is a no-op. Without the
guard, the module-level ``sys.exit(0)`` propagates out of
``importlib.import_module`` and crashes every dashboard subprocess run.
"""
import sys


def main() -> int:
    print("compare-runs: stub", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
