#!/usr/bin/env python3
"""Parse comparison_spec.json — the single source-of-truth manifest pinning BOTH
engines (repo+commit+branch) and per-condition seeds/gens/config.

stdlib-only so comparison_harness.sh can call it without a venv. Emits TAB-
separated rows for easy `IFS=$'\t' read` consumption in bash.

Usage:
  _read_spec.py <spec.json> engine {v2ecoli|vecoli}
        -> one line:  repo<TAB>commit<TAB>branch
  _read_spec.py <spec.json> conditions [cli_seeds] [cli_gens]
        -> one line per condition:  name<TAB>config<TAB>seeds<TAB>gens
        Resolution per field: condition value, else defaults value, else CLI value.

This is the unit-testable core of the manifest behaviour (feed it a spec, diff
the rows). Kept separate from comparison_harness.sh so the parsing has tests.
"""
import json
import sys


def load(path):
    with open(path) as fh:
        return json.load(fh)


def engine_row(spec, which):
    """Return (repo, commit, branch) for engine key 'v2ecoli' or 'vecoli'."""
    e = spec[which]
    return e.get("repo", ""), e.get("commit", ""), e.get("branch", "")


def _resolve(cond, defaults, key, cli):
    if key in cond and cond[key] is not None:
        return cond[key]
    if key in defaults and defaults[key] is not None:
        return defaults[key]
    return cli


def condition_rows(spec, cli_seeds, cli_gens):
    """Yield (name, config, seeds, gens) with per-condition > defaults > CLI."""
    defaults = spec.get("defaults", {})
    for cond in spec.get("conditions", []):
        name = cond["name"]
        config = cond.get("config", "cond_%s.json" % name)
        seeds = _resolve(cond, defaults, "seeds", cli_seeds)
        gens = _resolve(cond, defaults, "gens", cli_gens)
        yield name, config, int(seeds), int(gens)


def main(argv):
    if len(argv) < 3:
        sys.exit("usage: _read_spec.py <spec.json> {engine v2ecoli|vecoli | "
                 "conditions [cli_seeds] [cli_gens]}")
    spec = load(argv[1])
    mode = argv[2]
    if mode == "engine":
        which = {"v2ecoli": "v2ecoli", "vecoli": "vecoli",
                 "v2": "v2ecoli", "ve": "vecoli"}.get(argv[3], argv[3])
        repo, commit, branch = engine_row(spec, which)
        print("\t".join([repo, commit, branch]))
    elif mode == "conditions":
        cli_seeds = int(argv[3]) if len(argv) > 3 else 0
        cli_gens = int(argv[4]) if len(argv) > 4 else 0
        for name, config, seeds, gens in condition_rows(spec, cli_seeds, cli_gens):
            print("\t".join([name, config, str(seeds), str(gens)]))
    else:
        sys.exit("unknown mode: %s" % mode)


if __name__ == "__main__":
    main(sys.argv)
