"""MillardBulkIndexer — translate Millard ODE concentrations to bulk deltas.

The Millard-PDMP composite needs to make the kinetic ODE's central
metabolite changes visible to the rest of the WCM (Equilibrium,
TfBinding, transcription, etc.) — those processes read the structured
`bulk` array. Without this Step, Equilibrium depletes its substrate
pools and the run crashes at t~130s.

Implementation note: an earlier design routed Millard → FBABridge →
`central_metabolite_counts` (dict store) → this step. FBABridge fired
correctly and the dict came back to it on direct-invoke, but the
composite scheduler silently dropped its `v2ecoli_bulk` updates
(InPlaceDict-as-output + plain-dict-store-target seems to mismatch
somewhere in process-bigraph apply dispatch). To avoid that path
entirely, this Step now reads `central_metabolites` (mM, populated
reliably by Millard) and does the mM→count translation in-place
using the same species map FBABridge would have used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


DEFAULT_MAPPING = "v2ecoli/data/millard_v2ecoli_species_map.yaml"
DEFAULT_CELL_VOLUME_L = 1.0e-15
AVOGADRO = 6.02214076e23


def _load_millard_to_v2ecoli(mapping_file: str) -> dict[str, str]:
    path = Path(mapping_file)
    if not path.is_absolute():
        path = Path.cwd() / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    m2v: dict[str, str] = {}
    for section in ("adenylates", "redox", "glycolysis", "tca", "ppp", "common"):
        for entry in raw.get(section, []) or []:
            if entry.get("role") != "shared":
                continue
            mid = entry.get("millard_id") or entry.get("millard")
            vid = entry.get("v2ecoli_id") or entry.get("v2ecoli")
            if mid and vid:
                m2v[mid] = vid
    return m2v


class MillardBulkIndexer(Step):
    """Translate Millard central_metabolites (mM) → bulk count deltas."""

    name = "millard-bulk-indexer"
    topology = {
        "bulk": ("bulk",),
        # Port name intentionally not "central_metabolites" — when the
        # port and store path share a key, process-bigraph wires the
        # store such that any other writer's update is dropped. Verified
        # empirically: with port "central_metabolites" the indexer's
        # presence in the composite (even in a downstream layer) caused
        # millard-with-lqr's writes to be silently lost.
        "cm_view": ("shared", "central_metabolites"),
    }

    config_schema = {
        "mapping_file": {"_type": "string", "_default": DEFAULT_MAPPING},
        "cell_volume_L": {"_type": "float", "_default": DEFAULT_CELL_VOLUME_L},
        "min_count": {"_type": "float", "_default": 0.0},
        "stochastic_round": {"_type": "boolean", "_default": False},
        "seed": {"_type": "integer", "_default": 0},
    }

    def initialize(self, config):
        self.mapping_file = self.parameters.get("mapping_file", DEFAULT_MAPPING)
        self.cell_volume_L = float(self.parameters.get(
            "cell_volume_L", DEFAULT_CELL_VOLUME_L))
        self.min_count = float(self.parameters.get("min_count", 0.0))
        self.stochastic_round = bool(self.parameters.get("stochastic_round", False))
        self._rng = np.random.RandomState(self.parameters.get("seed", 0))
        self._m2v = _load_millard_to_v2ecoli(self.mapping_file)
        # Lazy index-resolution: bulk IDs available on first update.
        self._mids: list[str] | None = None
        self._bulk_idx: np.ndarray | None = None
        # mM × 1e-3 × V_L × Avogadro = count
        self._conc_to_count = 1e-3 * self.cell_volume_L * AVOGADRO

    def inputs(self):
        return {
            "bulk": "bulk_array",
            # No type — declaring "map[float]" promoted the store at the
            # wire's destination to a map, which prevented millard-with-lqr
            # (which writes an untyped dict) from landing its updates.
            "cm_view": {"_default": {}},
        }

    def outputs(self):
        return {"bulk": "bulk_array"}

    def update(self, states, interval=None):
        cm = states.get("cm_view") or {}
        if not cm:
            return {}

        # Lazy resolution: filter to (millard_id → v2ecoli_id) pairs that
        # actually exist in this run's bulk store. Skip pairs whose
        # v2ecoli_id is unknown (different ParCa fixtures expose different
        # bulk IDs).
        if self._mids is None:
            bulk_ids = states["bulk"]["id"]
            resolved_mids: list[str] = []
            resolved_idx: list[int] = []
            for mid, vid in self._m2v.items():
                if mid not in cm:
                    continue
                idx = bulk_name_to_idx(vid, bulk_ids, strict=False)
                if idx is None or (isinstance(idx, np.ndarray) and idx.size == 0):
                    continue
                resolved_mids.append(mid)
                resolved_idx.append(int(idx))
            self._mids = resolved_mids
            self._bulk_idx = np.asarray(resolved_idx, dtype=np.int64)

        if self._bulk_idx is None or self._bulk_idx.size == 0:
            return {}

        targets = np.fromiter(
            (float(cm.get(mid, 0.0)) * self._conc_to_count for mid in self._mids),
            dtype=np.float64,
            count=len(self._mids),
        )

        current = counts(states["bulk"], self._bulk_idx).astype(np.float64, copy=False)
        delta = targets - current

        if self.stochastic_round:
            # stochasticRound is the upstream-WCM canonical converter; using
            # it here mirrors metabolism.py:513. For a first pass, plain
            # round is fine — leaving this as a config knob.
            from wholecell.utils.random import stochasticRound
            delta_int = stochasticRound(self._rng, delta).astype(np.int64)
        else:
            delta_int = np.rint(delta).astype(np.int64)

        # Clip the resulting bulk count to min_count (default 0). Translate
        # that back to a delta that respects the floor.
        if self.min_count is not None:
            floor = self.min_count
            new_counts = current + delta_int
            below = new_counts < floor
            if below.any():
                delta_int = np.where(below, np.int64(floor - current.astype(np.int64)),
                                     delta_int)

        if not delta_int.any():
            return {}

        return {"bulk": [(self._bulk_idx, delta_int)]}
