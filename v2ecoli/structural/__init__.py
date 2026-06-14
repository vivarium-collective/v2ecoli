"""3D structural E. coli — translate a v2ecoli molecular state into a packed 3D
cell via pbg-parsimony, and render it in the bundled webapp.

- :func:`build_model` — the bridge (state → ingredients → pbg_parsimony.build_pack).
- ``parsimony-ecoli`` composite — the process-bigraph wiring (see ``composite``).
"""
from v2ecoli.structural.build import build_model, select_ingredients, load_state, categorize

__all__ = ["build_model", "select_ingredients", "load_state", "categorize"]
