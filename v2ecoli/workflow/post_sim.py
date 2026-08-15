"""Post-simulation Step bases (Visualization/ReportCardStep) + the unified
post-sim registry.

Moved to ``viva_superpowers.post_sim`` (the family's one home, shared with
every other pbg workspace) and re-exported here for back-compat: existing call
sites (``from v2ecoli.workflow.post_sim import ReportCardStep`` etc.) keep
working unchanged. ``AnalysisStep``/``Analysis``/``ANALYSIS_REGISTRY`` live
alongside the concrete analyses in ``v2ecoli/workflow/analysis.py`` (also
re-exported from the shared home); ``StudyContext``/``write_card``/``prune``
live alongside the concrete report cards in
``v2ecoli/workflow/report_cards/__init__.py``.

``Visualization`` is v2ecoli's historical name for the shared
``VisualizationStep`` base; kept as an alias so existing subclasses/imports
don't need to change.
"""
from __future__ import annotations

from viva_superpowers.post_sim import (  # noqa: F401
    ANALYSIS_REGISTRY,
    KINDS,
    POST_SIM_REGISTRY,
    REPORT_CARD_REGISTRY,
    VISUALIZATION_REGISTRY,
    ReportCardStep,
    VisualizationStep,
    iter_post_sim,
    register_post_sim,
)

# Back-compat alias: v2ecoli's concrete visualizations subclass ``Visualization``.
Visualization = VisualizationStep
