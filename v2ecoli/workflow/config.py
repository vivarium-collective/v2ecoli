"""Config loading with vEcoli-style inheritance.

Ported from vEcoli runscripts/workflow.py:load_config_with_inheritance +
_merge_configs. ``inherit_from`` chains resolve with priority
current > first-inherited > ... > last-inherited.
"""

from __future__ import annotations

import json
import os

# Keys whose list values are concatenated + deduplicated across the chain
# rather than overwritten (subset of vEcoli's; extend as configs need it).
LIST_KEYS_TO_MERGE = {
    "save_times",
    "add_processes",
    "exclude_processes",
    "engine_process_reports",
}


def load_config_with_inheritance(config_path: str, config_dir: str | None = None) -> dict:
    """Load a config file, recursively resolving ``inherit_from`` chains.

    ``config_dir`` is the directory inherited paths are resolved against
    (defaults to the directory of ``config_path``).
    """
    with open(config_path) as f:
        config = json.load(f)

    if config_dir is None:
        config_dir = os.path.dirname(os.path.abspath(config_path))

    if "inherit_from" not in config:
        return config

    inherit_chain = []
    for inherit_path in reversed(config["inherit_from"]):
        inherited = load_config_with_inheritance(
            os.path.join(config_dir, inherit_path), config_dir=config_dir)
        inherit_chain.append(inherited)

    result: dict = {}
    for inherited_config in inherit_chain:
        _merge_configs(result, inherited_config)
    _merge_configs(result, config)
    result.pop("inherit_from", None)
    return result


def _merge_configs(base_config: dict, overlay_config: dict) -> None:
    """Merge ``overlay_config`` into ``base_config`` in place (overlay wins)."""
    for key, value in overlay_config.items():
        if key in LIST_KEYS_TO_MERGE:
            base_config.setdefault(key, [])
            base_config[key].extend(value)
            base_config[key] = sorted(set(base_config[key]))
        elif (
            isinstance(value, dict)
            and key in base_config
            and isinstance(base_config[key], dict)
        ):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value
