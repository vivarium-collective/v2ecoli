"""Unified registry for post-simulation Steps (analyses, visualizations, report
cards). Each registered step is kind-tagged so the analysis flush can discover
and route every post-sim output from one place. Existing per-kind registries
(ANALYSIS_REGISTRY, REPORT_CARD_REGISTRY) remain the canonical homes; this is an
additive parallel index they funnel into."""
from __future__ import annotations

from v2ecoli.steps.base import V2Step

KINDS = ("analysis", "visualization", "report_card")

# name -> {"cls": <Step subclass>, "kind": <one of KINDS>}
POST_SIM_REGISTRY: dict[str, dict] = {}

# name -> ReportCardStep subclass. Populated by __init_subclass__ for any
# subclass that defines its own ``name`` (mirrors ANALYSIS_REGISTRY).
REPORT_CARD_REGISTRY: dict[str, type] = {}


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


class Visualization(V2Step):
    """A post-sim visualization Step: emits a rendered ``view`` (HTML) + ``data``
    (map), like ``Analysis`` but tagged ``kind="visualization"`` so the flush can
    route it distinctly. Subclasses set ``name`` and implement
    ``render(study) -> (html, data) | None``. Inputs default to a StudyContext;
    override ``inputs()`` to consume the run extraction instead."""

    name: str = ""
    config_schema: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("name"):
            register_post_sim(cls, "visualization")

    def inputs(self):
        return {"study": "any"}

    def outputs(self):
        return {"view": "string", "data": "map"}

    def render(self, study) -> "tuple[str, dict] | None":
        raise NotImplementedError

    def update(self, state, interval=None):
        study = state.get("study")
        res = self.render(study)
        if not res:
            return {"view": "", "data": {}}
        view, data = res
        return {"view": view, "data": data}


class ReportCardStep(V2Step):
    """A report card as a visualization-like Step (sibling of ``Analysis`` and
    ``Visualization``): emits ``view`` (HTML) + ``data`` (verdict map). Unlike
    ``Analysis`` — which consumes a live DuckDB sim-output connection — a report
    card's input is a ``StudyContext`` (the study's spec + dir), so cards grade
    run-free. Subclasses set ``name`` and implement ``applies(study)`` +
    ``build(study) -> (verdict_dict, html) | None``. A named subclass auto-registers
    in ``REPORT_CARD_REGISTRY`` and, kind-tagged, in ``POST_SIM_REGISTRY``."""

    name: str = ""
    config_schema: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("name"):
            REPORT_CARD_REGISTRY[cls.name] = cls
            register_post_sim(cls, "report_card")

    def inputs(self):
        return {"study": "any"}

    def outputs(self):
        return {"view": "string", "data": "map"}

    def applies(self, study) -> bool:
        return True

    def build(self, study) -> "tuple[dict, str] | None":
        """Return ``(verdict_json_dict, html_str)`` or None. Subclasses override."""
        raise NotImplementedError

    def update(self, state, interval=None):
        study = state.get("study")
        res = self.build(study) if study is not None else None
        if not res:
            return {"view": "", "data": {}}
        verdict, html = res
        return {"view": html, "data": verdict}
