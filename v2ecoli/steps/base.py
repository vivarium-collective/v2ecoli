"""
Base step class for v2ecoli.

Subclasses define inputs()/outputs() directly using bigraph-schema types.
"""

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate


class V2Step(Step):
    """Step base class for v2ecoli.

    Subclasses override inputs()/outputs() with explicit bigraph-schema
    types.  Wraps update() in error handling so missing data doesn't
    crash the Composite's step cascade.
    """

    config_schema = {}

    def inputs(self):
        return {}

    def outputs(self):
        return {}

    def invoke(self, state, interval=None):
        """Override invoke to catch errors from missing data."""
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)
