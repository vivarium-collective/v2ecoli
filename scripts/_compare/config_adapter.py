"""Resolve a vEcoli config and translate/diff it against v2ecoli's schema."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any


def schema_diff(vecoli: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """Partition keys: only-in-vEcoli, only-in-v2, shared-but-different.

    ``different`` maps each differing shared key to a (vecoli_value,
    v2_value) tuple. Only top-level keys are compared.
    """
    vkeys, v2keys = set(vecoli), set(v2)
    different = {
        k: (vecoli[k], v2[k])
        for k in (vkeys & v2keys)
        if vecoli[k] != v2[k]
    }
    return {
        "only_in_vecoli": sorted(vkeys - v2keys),
        "only_in_v2": sorted(v2keys - vkeys),
        "different": different,
    }


# Keys vEcoli sets that v2ecoli's workflow config does not consume. Their
# values are preserved under ``_dropped_vecoli_keys`` for the report so the
# mapping is explicit rather than silent.
_VECOLI_ONLY = (
    "emitter",
    "emitter_arg",
    "parca_options",
    "fail_at_max_duration",
    "suffix_time",
    "sim_data_path",
    # resolve_vecoli_config already returns the fully-merged config, so the
    # generated v2 config must NOT re-inherit (v2ecoli's loader would look for
    # the vEcoli base files relative to the v2 workdir and fail).
    "inherit_from",
)

# v2ecoli keys with defaults applied when the vEcoli config omits them.
_V2_DEFAULTS = {
    "lineage_seed": 0,
    "single_daughters": True,
}


def translate_vecoli_config(vecoli: dict[str, Any]) -> dict[str, Any]:
    """Map a resolved vEcoli config to a v2ecoli workflow config.

    Shared keys pass through unchanged; vEcoli-only keys are removed from
    the config body and recorded under ``_dropped_vecoli_keys``; missing
    v2ecoli keys get defaults from ``_V2_DEFAULTS``.
    """
    v2: dict[str, Any] = {
        k: v for k, v in vecoli.items() if k not in _VECOLI_ONLY
    }
    v2["_dropped_vecoli_keys"] = {
        k: vecoli[k] for k in _VECOLI_ONLY if k in vecoli
    }
    for k, default in _V2_DEFAULTS.items():
        v2.setdefault(k, default)
    # vEcoli's `max_duration` is its PER-GENERATION sim cap (each generation
    # runs until division OR max_duration). v2ecoli's lineage uses a SEPARATE
    # key, `max_duration_per_gen`, which DEFAULTS to 3600 s and is otherwise
    # never set by this translation. For slow-growth media (acetate, succinate,
    # anaerobic — doubling time > 60 min, i.e. natural cell cycle > 3600 s) the
    # 3600 s default force-divided v2 cells before their cell cycle completed:
    # they divided at low mass, never reached the 2nd round of replication
    # initiation, and so never got the rRNA gene-dosage boost that drives the
    # growth ramp — making v2 look 15-30% smaller than vEcoli purely as a cap
    # artifact. Map vEcoli's per-generation cap onto v2's so both engines
    # truncate generations identically (fast media are unaffected — they divide
    # well before either cap).
    if "max_duration" in vecoli:
        v2.setdefault("max_duration_per_gen", float(vecoli["max_duration"]))
        # max_duration is RENAMED to max_duration_per_gen, not a v2 key — drop
        # the source so the v2 config doesn't carry a redundant/confusing twin
        # (v2ecoli reads only max_duration_per_gen).
        v2.pop("max_duration", None)
    return v2


def resolve_vecoli_config_local(config_path: str, fork_dir: str) -> dict[str, Any]:
    """Resolve a vEcoli fork config WITHOUT needing the fork's own venv.

    ``resolve_vecoli_config`` shells out to ``<fork>/.venv/bin/python`` to honour
    the fork's loader. In the v2ecoli image the upstream clone (``V2E_VECOLI_DIR``)
    is SOURCE-ONLY (no venv), so this resolves the fork's config with v2ecoli's
    OWN inheritance loader instead (identical merge semantics), keyed off the
    config's directory inside the fork. ``config_path`` may be absolute or
    relative to ``fork_dir`` (e.g. ``configs/default.json``).
    """
    import os
    abspath = (config_path if os.path.isabs(config_path)
               else os.path.join(fork_dir, config_path))
    config_dir = os.path.dirname(abspath)
    try:
        from v2ecoli.workflow.config import load_config_with_inheritance
        return load_config_with_inheritance(abspath, config_dir=config_dir)
    except Exception:
        # Last-resort fallback: a flat read (no inheritance) so a missing v2ecoli
        # loader / unexpected schema still yields the config body.
        with open(abspath) as fh:
            return json.load(fh)


VECOLI_REPO = "/Users/eranagmon/code/vEcoli"
VECOLI_PYTHON = f"{VECOLI_REPO}/.venv/bin/python"


def resolve_vecoli_config(config_path: str,
                          vecoli_repo: str = VECOLI_REPO) -> dict[str, Any]:
    """Resolve a vEcoli config (honoring ``inherit_from``) using the fork's
    own loader, returning the fully-merged dict."""
    vecoli_python = f"{vecoli_repo}/.venv/bin/python"
    snippet = (
        "import json,sys;"
        "from runscripts.workflow import load_config_with_inheritance;"
        "json.dump(load_config_with_inheritance(sys.argv[1]), sys.stdout)"
    )
    out = subprocess.check_output(
        [vecoli_python, "-c", snippet, config_path],
        cwd=vecoli_repo, text=True,
    )
    return json.loads(out)



def config_run_shape(config_path: str, fork_dir: str) -> tuple[int, int]:
    """Return (seeds, gens) = (n_init_sims, generations) from a vEcoli config.

    seeds/gens are the config's run shape — the single source of truth. Missing
    or null values default to 1 (one seed, one generation).
    """
    cfg = resolve_vecoli_config_local(config_path, fork_dir)
    seeds = cfg.get("n_init_sims")
    gens = cfg.get("generations")
    return int(seeds) if seeds else 1, int(gens) if gens else 1
