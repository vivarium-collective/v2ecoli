"""Resolve a vEcoli config and translate/diff it against v2ecoli's schema."""
from __future__ import annotations

import json
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
    return v2


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
