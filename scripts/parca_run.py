#!/usr/bin/env python
"""Thin shim — the CLI lives at v2ecoli.cli.parca:main.

Prefer running `v2ecoli-parca` (installed as a console script via
pyproject.toml) instead of invoking this file directly.
"""

from v2ecoli.cli.parca import main


if __name__ == "__main__":
    main()
