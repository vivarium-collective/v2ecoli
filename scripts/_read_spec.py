#!/usr/bin/env python3
"""Parse comparison_spec.json — the single source-of-truth manifest pinning BOTH
engines (repo+commit+branch) and per-condition seeds/gens/config.

stdlib-only so comparison_harness.sh can call it without a venv. Emits TAB-
separated rows for easy `IFS=$'\t' read` consumption in bash.

Usage:
  _read_spec.py <spec.json> engine {v2ecoli|vecoli}
        -> one line:  repo<TAB>commit<TAB>branch
  _read_spec.py <spec.json> vecoli-fork
        -> one line:  repo<TAB>ref     (ref = vecoli.commit if set, else
           vecoli.branch, else 'master'; repo defaults to CovertLab/vEcoli).
           This is the upstream vEcoli fork the image WRAPS — feed it to
           `docker build --build-arg VECOLI_UPSTREAM_REPO=… VECOLI_UPSTREAM_REF=…`.
  _read_spec.py <spec.json> vecoli-engine
        -> one line:  upstream-wrapper | nextflow  (default upstream-wrapper)
  _read_spec.py <spec.json> from-vecoli-config
        -> one line:  the config path within the vEcoli fork to drive BOTH
           engines from (empty line if unset).
  _read_spec.py <spec.json> conditions [cli_seeds] [cli_gens]
        -> one line per condition:  name<TAB>config<TAB>seeds<TAB>gens
        Resolution per field: condition value, else defaults value, else CLI value.
  _read_spec.py <spec.json> configs
        -> one line per config:  config_path<TAB>cards_csv
        cards_csv: entry 'cards' -> defaults.cards -> 'standard'.

This is the unit-testable core of the manifest behaviour (feed it a spec, diff
the rows). Kept separate from comparison_harness.sh so the parsing has tests.
"""
import json
import sys

DEFAULT_VECOLI_REPO = "https://github.com/CovertLab/vEcoli"
DEFAULT_VECOLI_REF = "master"
DEFAULT_VECOLI_ENGINE = "upstream-wrapper"


def load(path):
    with open(path) as fh:
        return json.load(fh)


def engine_row(spec, which):
    """Return (repo, commit, branch) for engine key 'v2ecoli' or 'vecoli'."""
    e = spec[which]
    return e.get("repo", ""), e.get("commit", ""), e.get("branch", "")


def vecoli_fork(spec):
    """Return (repo, ref) for the upstream vEcoli fork the image should WRAP.

    ``ref`` pins the clone: the explicit ``vecoli.commit`` wins (reproducible),
    else ``vecoli.branch``, else ``master``. Pointing at a different fork is a
    pure spec edit — no Dockerfile/code change.
    """
    v = spec.get("vecoli", {}) or {}
    repo = v.get("repo") or DEFAULT_VECOLI_REPO
    ref = v.get("commit") or v.get("branch") or DEFAULT_VECOLI_REF
    return repo, ref


def vecoli_engine(spec):
    """How the genuine vEcoli side runs:
      - 'upstream-wrapper' (default) — vEcoli decomposed into ~50 pbg steps (external
        wrapper), run as a Ray composite=vecoli job on the v2ecoli image.
      - 'vivarium-process' — genuine vEcoli as a SINGLE pbg node with vivarium-core's
        own Engine inside (faithful by construction: vivarium handles partition /
        reconcile / division natively). Same Ray composite=vecoli job + image.
      - 'nextflow' — legacy separate Nextflow registration (parquet output)."""
    return spec.get("vecoli_engine") or DEFAULT_VECOLI_ENGINE


# Map the spec's vecoli_engine -> run_comparison_ensemble's --vecoli-source flag.
_ENGINE_TO_SOURCE = {
    "upstream-wrapper": "upstream",          # step-decomposed external wrapper (pbg)
    "vivarium-process": "vivarium-process",  # single pbg node, vivarium Engine inside
}


def vecoli_source(spec):
    """The ``run_comparison_ensemble --vecoli-source`` value for the spec's
    ``vecoli_engine``. 'nextflow' has no run_comparison_ensemble source (it registers
    separately), so it falls back to the upstream default."""
    return _ENGINE_TO_SOURCE.get(vecoli_engine(spec), "upstream")


def from_vecoli_config(spec):
    """Config path within the vEcoli fork to drive BOTH engines from ('' if unset)."""
    return spec.get("from_vecoli_config") or ""


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


def config_rows(spec):
    """Yield (config_path, cards_csv) per spec['configs'] entry.

    cards: entry 'cards' -> spec defaults.cards -> ['standard'].
    seeds/gens/variants are NOT here — the runner reads them from each config.
    """
    default_cards = (spec.get("defaults", {}) or {}).get("cards") or ["standard"]
    for entry in spec.get("configs", []):
        config = entry["config"]
        cards = entry.get("cards") or default_cards
        yield config, ",".join(cards)


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
    elif mode == "vecoli-fork":
        repo, ref = vecoli_fork(spec)
        print("\t".join([repo, ref]))
    elif mode == "vecoli-engine":
        print(vecoli_engine(spec))
    elif mode == "from-vecoli-config":
        print(from_vecoli_config(spec))
    elif mode == "conditions":
        cli_seeds = int(argv[3]) if len(argv) > 3 else 0
        cli_gens = int(argv[4]) if len(argv) > 4 else 0
        for name, config, seeds, gens in condition_rows(spec, cli_seeds, cli_gens):
            print("\t".join([name, config, str(seeds), str(gens)]))
    elif mode == "configs":
        for config, cards in config_rows(spec):
            print("\t".join([config, cards]))
    else:
        sys.exit("unknown mode: %s" % mode)


if __name__ == "__main__":
    main(sys.argv)
