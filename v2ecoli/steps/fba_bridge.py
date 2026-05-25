"""FBABridge Step — translates between Millard 2017 ODE and v2ecoli WCM namespaces.

v2ecoli-pdmp Phase 1 milestone: this Step is the seam where the kinetic ODE
(central carbon, ~30 metabolites in mM) hands off concentrations to the
v2ecoli WCM bulk store (BioCyc IDs, ~16k metabolites in molecule counts)
and vice versa.

Scope of this implementation (intentionally minimal):
  - Reads a `mapping_file` (YAML, see v2ecoli/data/millard_v2ecoli_species_map.yaml)
  - Translates Millard concentrations (mM) → v2ecoli counts using
    count = conc_mM * 1e-3 * V_cell_L * N_avogadro
  - And the reverse direction (counts → mM)
  - Logs per-step diagnostics into a `bridge_diagnostics` store

Out of scope (deferred to a later Phase 1 milestone):
  - Coupling to v2ecoli's Metabolism process (the full WCM context — cache,
    bulk, unique, boundary, partition allocator — is required by Metabolism
    and a bigger lift than the bridge translation itself).
  - Flux-based mass-balance enforcement (the bridge currently only
    translates concentrations; flux-routing decisions belong in the
    composite topology, not the bridge).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict


# Default cell volume for an average E. coli cell at division.
# Matches the v2ecoli ParCa default (1 fL ~= 1.0e-15 L); the actual mean
# cell volume varies with growth condition but 1 fL is a reasonable
# zero-th order constant. The bridge accepts an override via config.
DEFAULT_CELL_VOLUME_L = 1.0e-15
AVOGADRO = 6.02214076e23


def _load_mapping(mapping_file: str) -> dict[str, Any]:
    """Load and flatten the species-mapping YAML into a usable form.

    Returns:
      {
        "millard_to_v2ecoli": {millard_id: v2ecoli_id, ...},  # role: shared only
        "v2ecoli_to_millard": {v2ecoli_id: millard_id, ...},
        "all_shared_pairs": [(millard_id, v2ecoli_id, notes), ...],
        "unit_conversion": {...},  # passthrough
        "millard_only": [millard_id, ...],
      }
    """
    path = Path(mapping_file)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(
            f"FBABridge mapping file not found: {mapping_file}. "
            f"Resolved to {path}. See "
            f"v2ecoli/data/millard_v2ecoli_species_map.yaml for the canonical map."
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    m2v: dict[str, str] = {}
    v2m: dict[str, str] = {}
    pairs: list[tuple[str, str, str]] = []
    for section in ("adenylates", "redox", "glycolysis", "tca", "ppp", "common"):
        for entry in raw.get(section, []) or []:
            if entry.get("role") != "shared":
                continue
            mid = entry["millard_id"]
            vid = entry["v2ecoli_id"]
            m2v[mid] = vid
            v2m[vid] = mid
            pairs.append((mid, vid, entry.get("notes", "")))

    millard_only = [
        e.get("millard_id") for e in raw.get("millard_only", []) or []
    ]

    return {
        "millard_to_v2ecoli": m2v,
        "v2ecoli_to_millard": v2m,
        "all_shared_pairs": pairs,
        "unit_conversion": raw.get("unit_conversion", {}),
        "millard_only": millard_only,
    }


def mM_to_count(conc_mM: float, cell_volume_L: float) -> float:
    """Concentration (mM) -> molecule count in a cell of given volume."""
    return conc_mM * 1e-3 * cell_volume_L * AVOGADRO


def count_to_mM(count: float, cell_volume_L: float) -> float:
    """Molecule count -> concentration (mM) in a cell of given volume."""
    if cell_volume_L <= 0:
        return 0.0
    return count / (cell_volume_L * AVOGADRO) * 1e3


class FBABridge(Step):
    """Translate metabolite state between Millard ODE and v2ecoli bulk stores.

    Topology (composite-side):
        inputs:
          central_metabolites_millard: ["shared", "central_metabolites"]
          v2ecoli_bulk:                ["v2ecoli", "bulk"]
        outputs:
          central_metabolites_millard: ["shared", "central_metabolites"]
          v2ecoli_bulk:                ["v2ecoli", "bulk"]
          bridge_diagnostics:          ["shared", "bridge_diagnostics"]

    Direction modes:
      - "millard_to_v2ecoli" — Millard concentrations drive v2ecoli counts.
      - "v2ecoli_to_millard" — v2ecoli counts drive Millard concentrations.
      - "bidirectional"      — both directions; v2ecoli is treated as the
                               authoritative state and Millard is rebound
                               to match it (useful when v2ecoli's allocator
                               just updated the bulk pool).
    """

    name = "fba_bridge"
    config_schema = {
        "mapping_file": {"_default": "v2ecoli/data/millard_v2ecoli_species_map.yaml"},
        "direction": {"_default": "millard_to_v2ecoli"},
        "cell_volume_L": {"_default": DEFAULT_CELL_VOLUME_L},
        "time_step": {"_default": 1},
    }

    topology = {
        "central_metabolites_millard": ("shared", "central_metabolites"),
        "v2ecoli_bulk": ("v2ecoli", "bulk"),
        "bridge_diagnostics": ("shared", "bridge_diagnostics"),
    }

    def initialize(self, config):
        self.parameters = config or {}
        self.mapping_file = self.parameters.get(
            "mapping_file", self.config_schema["mapping_file"]["_default"]
        )
        self.direction = self.parameters.get(
            "direction", self.config_schema["direction"]["_default"]
        )
        if self.direction not in (
            "millard_to_v2ecoli", "v2ecoli_to_millard", "bidirectional"
        ):
            raise ValueError(
                f"FBABridge: unknown direction {self.direction!r}. "
                "Must be one of: millard_to_v2ecoli | v2ecoli_to_millard | bidirectional."
            )
        self.cell_volume_L = float(self.parameters.get(
            "cell_volume_L", DEFAULT_CELL_VOLUME_L
        ))
        self._mapping = _load_mapping(self.mapping_file)
        self.m2v = self._mapping["millard_to_v2ecoli"]
        self.v2m = self._mapping["v2ecoli_to_millard"]
        self.millard_only = set(self._mapping["millard_only"])

    def inputs(self):
        return {
            "central_metabolites_millard": InPlaceDict(),
            "v2ecoli_bulk": InPlaceDict(),
        }

    def outputs(self):
        return {
            "central_metabolites_millard": InPlaceDict(),
            "v2ecoli_bulk": InPlaceDict(),
            "bridge_diagnostics": InPlaceDict(),
        }

    def update(self, state):
        return self.next_update(self.parameters.get("time_step", 1), state)

    def next_update(self, timestep, states):
        millard_state = dict(states.get("central_metabolites_millard", {}) or {})
        v2ecoli_state = dict(states.get("v2ecoli_bulk", {}) or {})

        millard_update: dict[str, float] = {}
        v2ecoli_update: dict[str, float] = {}

        shared_pool_count = 0
        mass_residual = 0.0

        if self.direction in ("millard_to_v2ecoli", "bidirectional"):
            for mid, vid in self.m2v.items():
                if mid in millard_state:
                    raw = millard_state[mid]
                    try:
                        conc = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if conc != conc:
                        continue
                    count = mM_to_count(conc, self.cell_volume_L)
                    v2ecoli_update[vid] = count
                    shared_pool_count += 1

        if self.direction in ("v2ecoli_to_millard", "bidirectional"):
            for vid, mid in self.v2m.items():
                if vid in v2ecoli_state:
                    conc_mM = count_to_mM(
                        float(v2ecoli_state[vid]), self.cell_volume_L
                    )
                    prev = float(millard_state.get(mid, conc_mM))
                    mass_residual += abs(conc_mM - prev)
                    millard_update[mid] = conc_mM
                    if self.direction == "v2ecoli_to_millard":
                        shared_pool_count += 1

        millard_unmapped = [
            k for k in millard_state.keys()
            if k not in self.m2v and k not in self.millard_only
        ]
        v2ecoli_keys_in_state = list(v2ecoli_state.keys())[:20]

        diagnostics = {
            "shared_pool_count": shared_pool_count,
            "v2ecoli_unmapped_sample": [
                k for k in v2ecoli_keys_in_state if k not in self.v2m
            ][:10],
            "millard_unmapped": millard_unmapped,
            "mass_balance_residual_mM": round(mass_residual, 6),
            "cell_volume_L_used": self.cell_volume_L,
            "direction": self.direction,
        }

        update: dict[str, Any] = {"bridge_diagnostics": diagnostics}
        if millard_update:
            update["central_metabolites_millard"] = millard_update
        if v2ecoli_update:
            update["v2ecoli_bulk"] = v2ecoli_update
        return update


def register(core):
    """Register FBABridge under the local: address used by composite YAMLs.

    Call this from v2ecoli.core.build_core if the composite needs the bridge,
    or invoke it ad-hoc in a test setup:

        from v2ecoli.core import build_core
        from v2ecoli.steps.fba_bridge import register as register_bridge
        core = build_core()
        register_bridge(core)
    """
    core.register_link("FBABridge", FBABridge)


__all__ = [
    "FBABridge",
    "register",
    "mM_to_count",
    "count_to_mM",
    "_load_mapping",
    "DEFAULT_CELL_VOLUME_L",
    "AVOGADRO",
]
