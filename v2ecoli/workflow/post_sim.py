"""Unified registry for post-simulation Steps (analyses, visualizations, report
cards). Each registered step is kind-tagged so the analysis flush can discover
and route every post-sim output from one place. Existing per-kind registries
(ANALYSIS_REGISTRY, REPORT_CARD_REGISTRY) remain the canonical homes; this is an
additive parallel index they funnel into."""
from __future__ import annotations

KINDS = ("analysis", "visualization", "report_card")

# name -> {"cls": <Step subclass>, "kind": <one of KINDS>}
POST_SIM_REGISTRY: dict[str, dict] = {}


def register_post_sim(cls, kind: str, name: "str | None" = None) -> None:
    """Register a post-sim Step subclass under ``name`` (default ``cls.name``)
    with its ``kind``. No-op when the resolved name is falsy (abstract bases).
    Raises ValueError for an unknown kind."""
    if kind not in KINDS:
        raise ValueError(f"unknown post-sim kind {kind!r}; expected one of {KINDS}")
    nm = name if name is not None else getattr(cls, "name", "")
    if not nm:
        return
    POST_SIM_REGISTRY[nm] = {"cls": cls, "kind": kind}


def iter_post_sim(kind: "str | None" = None) -> list:
    """[(name, cls), ...] sorted by name, optionally filtered to one kind."""
    out = [(nm, e["cls"]) for nm, e in POST_SIM_REGISTRY.items()
           if kind is None or e["kind"] == kind]
    return sorted(out, key=lambda t: t[0])
