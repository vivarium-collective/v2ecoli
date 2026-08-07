"""Translation-level gene perturbations (the straightforward KO tranche).

RFC-007 variant set 2, ported to v2ecoli's cache-config surface. A perturbation
is a per-monomer multiplier on translation efficiency:

    multiplier == 0    -> knockout      (no ribosome initiates on that mRNA)
    0 < multiplier < 1 -> knockdown
    multiplier > 1     -> overexpression

The reference (``CovertLab/vEcoli`` branch ``strain-modification-variants``,
``ecoli/variants/native_translation_perturbation.py``) edits
``sim_data.process.translation.translation_efficiencies_by_monomer`` at ParCa
time. v2ecoli's ``baseline`` composite builds from a cached process config, not a
live sim_data — so the same edit lands here on the cached
``ecoli-polypeptide-initiation`` config's ``translation_efficiencies`` array,
fed through ``baseline``'s existing ``config_overrides`` seam. No re-ParCa.

Caveats (same as the reference — translation-only edits):
  * mRNAs are still transcribed and consume RNAP capacity; only translation is
    zeroed.
  * Polycistronic transcripts share one mRNA — scaling one monomer's efficiency
    does not touch other monomers on the same operon.
  * Non-coding genes have no monomer and cannot be perturbed here (rejected).

Targets are addressed two ways, both resolved against the cache:
  * an EcoCyc gene id (``EG10527``) — the reference's identity, joined
    gene -> cistron -> monomer through the cached config tables;
  * a monomer id (``LACZ-MONOMER[c]``, or bare ``LACZ-MONOMER``) — matched
    directly against the config's ``monomer_ids`` for precise targeting.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Cached-config keys this module reads. The polypeptide-initiation config
# carries the per-monomer translation efficiencies + the monomer<->cistron join;
# the rna-synth-prob listener config carries the cistron-index-space gene ids.
_PI = "ecoli-polypeptide-initiation"
_RS = "rna_synth_prob_listener"


class UnknownPerturbationTarget(ValueError):
    """A requested target is not a known gene/monomer, or is non-coding."""


def _config_tables(bundle: Mapping[str, Any]) -> tuple[list[str], list[str], dict[int, int], np.ndarray]:
    """Pull the id lists + join + efficiencies out of the cache bundle.

    Returns ``(gene_ids, monomer_ids, cistron_idx_to_monomer_idx, efficiencies)``
    where ``gene_ids`` is indexed by cistron index (the join space) and
    ``monomer_ids`` / ``efficiencies`` are indexed by monomer index.
    """
    configs = bundle.get("configs") or {}
    pi = configs.get(_PI)
    rs = configs.get(_RS)
    if not isinstance(pi, dict) or _PI not in configs:
        raise KeyError(
            f"cache bundle has no {_PI!r} config — cannot apply translation "
            "perturbations (is this a full ParCa cache?)")
    monomer_ids = [str(x) for x in pi["monomer_ids"]]
    efficiencies = np.asarray(pi["translation_efficiencies"], dtype=float)
    # monomer_idx -> cistron_idx, inverted to cistron_idx -> monomer_idx.
    m2c = pi["monomer_index_to_cistron_index"]
    cistron_to_monomer = {int(ci): int(mi) for mi, ci in m2c.items()}
    # gene ids live in the rna-synth-prob listener config, in cistron-index order.
    gene_ids = [str(x) for x in rs["gene_ids"]] if isinstance(rs, dict) and "gene_ids" in rs else []
    return gene_ids, monomer_ids, cistron_to_monomer, efficiencies


def resolve_targets(
    bundle: Mapping[str, Any], targets: list[str]
) -> dict[str, int]:
    """Resolve each target (gene id or monomer id) to a monomer index.

    Order of resolution per target: exact monomer id, bare monomer id (no
    ``[compartment]`` suffix), then EcoCyc gene id via the gene -> cistron ->
    monomer join. Raises :class:`UnknownPerturbationTarget` listing every target
    that did not resolve or that resolved to a non-coding gene (no monomer).
    """
    gene_ids, monomer_ids, cistron_to_monomer, _ = _config_tables(bundle)

    monomer_exact = {mid: i for i, mid in enumerate(monomer_ids)}
    monomer_bare: dict[str, list[int]] = {}
    for i, mid in enumerate(monomer_ids):
        bare = mid.split("[", 1)[0]
        monomer_bare.setdefault(bare, []).append(i)
    gene_to_cistron_idx = {g: i for i, g in enumerate(gene_ids)}

    resolved: dict[str, int] = {}
    unknown: list[str] = []
    non_coding: list[str] = []
    ambiguous: list[tuple[str, int]] = []
    for t in targets:
        if t in monomer_exact:
            resolved[t] = monomer_exact[t]
        elif t in monomer_bare:
            hits = monomer_bare[t]
            if len(hits) > 1:
                ambiguous.append((t, len(hits)))
            else:
                resolved[t] = hits[0]
        elif t in gene_to_cistron_idx:
            mi = cistron_to_monomer.get(gene_to_cistron_idx[t])
            if mi is None:
                non_coding.append(t)
            else:
                resolved[t] = mi
        else:
            unknown.append(t)

    errors: list[str] = []
    if unknown:
        errors.append(
            f"unknown targets (not an EcoCyc gene id in {_RS}.gene_ids nor a "
            f"monomer id in {_PI}.monomer_ids): {unknown}")
    if non_coding:
        errors.append(
            f"targets resolved to a cistron with no monomer (non-coding RNA — "
            f"cannot be knocked out at the translation level): {non_coding}")
    if ambiguous:
        errors.append(
            f"bare monomer names matching multiple compartments (give the full "
            f"'NAME[compartment]' id): {[t for t, _ in ambiguous]}")
    if errors:
        raise UnknownPerturbationTarget("; ".join(errors))
    return resolved


def translation_efficiency_override(
    bundle: Mapping[str, Any],
    perturbations: Mapping[str, float] | list[str],
) -> dict[str, Any]:
    """Build the ``config_overrides`` entry that applies these perturbations.

    ``perturbations`` is either a mapping ``{target: multiplier}`` or a plain
    list of targets (each treated as a knockout, multiplier 0). Returns
    ``{"ecoli-polypeptide-initiation.translation_efficiencies": <patched array>}``
    — a full replacement array so ``baseline``'s override seam
    (``configs[proc][key] = value``) applies it in one assignment. Returns an
    empty dict when there is nothing to perturb, so callers can merge
    unconditionally.
    """
    if not perturbations:
        return {}
    if isinstance(perturbations, Mapping):
        mult = {str(k): float(v) for k, v in perturbations.items()}
    else:
        mult = {str(t): 0.0 for t in perturbations}

    bad = {k: v for k, v in mult.items() if v < 0 or not np.isfinite(v)}
    if bad:
        raise ValueError(f"multipliers must be finite and non-negative: {bad}")

    resolved = resolve_targets(bundle, list(mult))
    _, _, _, efficiencies = _config_tables(bundle)
    patched = efficiencies.copy()
    for target, monomer_idx in resolved.items():
        patched[monomer_idx] = efficiencies[monomer_idx] * mult[target]
    return {f"{_PI}.translation_efficiencies": patched}
