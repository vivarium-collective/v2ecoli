"""
Composite loading for v2ecoli.

Uses process-bigraph's Composite directly — no custom simulation engine.
"""

from bigraph_schema import allocate_core
from process_bigraph import Composite

from v2ecoli.types import ECOLI_TYPES


def _build_core():
    """Create and configure a bigraph-schema core with ecoli types."""
    core = allocate_core()
    core.register_types(ECOLI_TYPES)
    return core


def make_composite(document=None, sim_data_path=None, seed=0, core=None):
    """Create a Composite from a document or by building one.

    Args:
        document: Pre-built document dict. If None, builds one from sim_data_path.
        sim_data_path: Path to simData pickle. Used if document is None.
        seed: Random seed for initial state generation.
        core: Pre-configured core. If None, creates one.

    Returns:
        A Composite ready for .run(interval).
    """
    if core is None:
        core = _build_core()

    if document is None:
        from v2ecoli.generate import build_document
        document = build_document(sim_data_path=sim_data_path, seed=seed)

    composite = Composite(document, core=core)
    return composite
