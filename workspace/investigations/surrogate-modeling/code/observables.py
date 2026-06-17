"""Broad-panel observable extractor: v2ecoli baseline state -> flat float vector.

The pbg-torch surrogate operates on a flat, fixed-order numeric vector. The
v2ecoli baseline state, by contrast, is a deeply nested dict containing pint
Quantities, dict-valued exchange fluxes, and high-dimensional count/flux
vectors, all under a per-agent key that changes at division. This module
bridges the two: it discovers a stable PanelLayout from a sample state and
extracts a flat float64 vector in that fixed order on every subsequent step.

The "broad / high-dimensional" panel spans every observable category the
surrogate is asked to approximate:

    group         where                                               ~dim
    ----------    ------------------------------------------------    ----
    mass          listeners.mass.<scalar>  (growth + mass fractions)     7
    exchange      environment.exchange  (dict, sorted by molecule)      87
    base_flux     listeners.fba_results.base_reaction_fluxes          2820
    monomer       listeners.monomer_counts  (protein abundance)       4309
    mrna          listeners.rna_counts.mRNA_cistron_counts            4345
    chromosome    listeners.replication_data.number_of_oric              1
                                                                   -------
                                                                   ~11569

The layout is serializable (to_dict / from_dict) so sampling, training, and
evaluation all share the exact same column ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Default broad panel. Order here defines the order of groups in the vector.
DEFAULT_GROUPS: Tuple[str, ...] = (
    "mass", "exchange", "base_flux", "monomer", "mrna", "chromosome",
)

_MASS_SCALARS = (
    "cell_mass", "dry_mass", "protein_mass", "rna_mass", "dna_mass",
    "volume", "instantaneous_growth_rate",
)


def _mag(x) -> float:
    """Strip pint units / numpy scalars to a plain float."""
    return float(getattr(x, "magnitude", x))


def get_cell(state: dict) -> dict:
    """Return the single live agent's state subtree.

    Sampling halts before division, so exactly one agent is expected.
    """
    agents = state["agents"]
    keys = list(agents.keys())
    if len(keys) != 1:
        raise ValueError(f"expected exactly one live agent, found {keys}")
    return agents[keys[0]]


@dataclass
class PanelLayout:
    """Fixed column ordering for the broad observable panel.

    ``labels`` are (group, name) pairs, one per vector column. ``group_slices``
    maps each group to its [start, end) column span.
    """

    groups: List[str]
    labels: List[Tuple[str, str]]
    group_slices: Dict[str, Tuple[int, int]]
    exchange_keys: List[str] = field(default_factory=list)

    @property
    def n_dims(self) -> int:
        return len(self.labels)

    @classmethod
    def discover(cls, state: dict, groups: Tuple[str, ...] = DEFAULT_GROUPS) -> "PanelLayout":
        """Build the layout from a sample (stepped) composite state."""
        cell = get_cell(state)
        labels: List[Tuple[str, str]] = []
        slices: Dict[str, Tuple[int, int]] = {}
        exchange_keys: List[str] = []

        for group in groups:
            start = len(labels)
            if group == "mass":
                labels += [("mass", k) for k in _MASS_SCALARS]
            elif group == "exchange":
                exchange_keys = sorted(cell["environment"]["exchange"].keys())
                labels += [("exchange", k) for k in exchange_keys]
            elif group == "base_flux":
                n = np.asarray(cell["listeners"]["fba_results"]["base_reaction_fluxes"]).size
                labels += [("base_flux", str(i)) for i in range(n)]
            elif group == "monomer":
                n = np.asarray(cell["listeners"]["monomer_counts"]).size
                labels += [("monomer", str(i)) for i in range(n)]
            elif group == "mrna":
                n = np.asarray(cell["listeners"]["rna_counts"]["mRNA_cistron_counts"]).size
                labels += [("mrna", str(i)) for i in range(n)]
            elif group == "chromosome":
                labels += [("chromosome", "number_of_oric")]
            else:
                raise ValueError(f"unknown observable group: {group}")
            slices[group] = (start, len(labels))

        return cls(groups=list(groups), labels=labels, group_slices=slices,
                   exchange_keys=exchange_keys)

    def extract(self, state: dict) -> np.ndarray:
        """Extract the flat float64 observable vector from a composite state."""
        cell = get_cell(state)
        out = np.empty(self.n_dims, dtype=np.float64)
        for group, (start, end) in self.group_slices.items():
            if group == "mass":
                mass = cell["listeners"]["mass"]
                out[start:end] = [_mag(mass[k]) for k in _MASS_SCALARS]
            elif group == "exchange":
                ex = cell["environment"]["exchange"]
                out[start:end] = [_mag(ex[k]) for k in self.exchange_keys]
            elif group == "base_flux":
                out[start:end] = np.asarray(
                    cell["listeners"]["fba_results"]["base_reaction_fluxes"], dtype=np.float64)
            elif group == "monomer":
                out[start:end] = np.asarray(cell["listeners"]["monomer_counts"], dtype=np.float64)
            elif group == "mrna":
                out[start:end] = np.asarray(
                    cell["listeners"]["rna_counts"]["mRNA_cistron_counts"], dtype=np.float64)
            elif group == "chromosome":
                out[start:end] = [_mag(cell["listeners"]["replication_data"]["number_of_oric"])]
        return out

    def synthetic_paths(self) -> List[List[str]]:
        """A unique [group, name] path per column, for the pbg-torch SurrogateSpec.

        These are not real bigraph state paths (the surrogate is trained on the
        extracted vector, not wired live into the composite); they only need to
        be unique and order-stable.
        """
        return [[g, n] for (g, n) in self.labels]

    def to_dict(self) -> dict:
        return {
            "groups": self.groups,
            "labels": [list(lbl) for lbl in self.labels],
            "group_slices": {k: list(v) for k, v in self.group_slices.items()},
            "exchange_keys": self.exchange_keys,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PanelLayout":
        return cls(
            groups=list(d["groups"]),
            labels=[tuple(lbl) for lbl in d["labels"]],
            group_slices={k: tuple(v) for k, v in d["group_slices"].items()},
            exchange_keys=list(d.get("exchange_keys", [])),
        )


def build_spec(layout: PanelLayout, timestep: float = 1.0):
    """Construct a pbg-torch SurrogateSpec for the panel (fully autoregressive:
    every observable is both a feature and a target; no exogenous drivers)."""
    from pbg_torch import SurrogateSpec

    return SurrogateSpec(
        target_paths=layout.synthetic_paths(),
        driver_paths=[],
        timestep=timestep,
        parameterization="delta",
    )
