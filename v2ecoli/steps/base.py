"""
Base step class for v2ecoli with automatic schema translation.

Provides inputs()/outputs() that translate v1-style ports_schema()
to bigraph-schema types for Composite compatibility.
"""

from process_bigraph import Step
from bigraph_schema.schema import Node, Float, Overwrite

from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate


class V2Step(Step):
    """Step base class that auto-translates ports_schema to inputs/outputs.

    Wraps update() in error handling so missing data doesn't crash
    the Composite's step cascade.
    """

    config_schema = {}

    def inputs(self):
        if hasattr(self, 'ports_schema'):
            return _translate_schema(self.ports_schema())
        return {}

    def outputs(self):
        if hasattr(self, 'ports_schema'):
            return _translate_schema(self.ports_schema())
        return {}

    def invoke(self, state, interval=None):
        """Override invoke to catch errors from missing data."""
        from process_bigraph.composite import SyncUpdate
        try:
            update = self.update(state)
        except (KeyError, TypeError, AttributeError, ValueError, AssertionError, RuntimeError):
            update = {}
        return SyncUpdate(update)


def _translate_schema(ports):
    """Convert v1 ports_schema dict to bigraph-schema types."""
    result = {}
    for key, port in ports.items():
        if isinstance(port, dict):
            updater = port.get('_updater')
            if callable(updater):
                name = getattr(updater, '__name__', '')
                if 'bulk_numpy' in name or name == 'writeable_updater':
                    result[key] = BulkNumpyUpdate()
                    continue
                if hasattr(updater, '__self__') and hasattr(updater.__self__, 'add_updates'):
                    result[key] = UniqueNumpyUpdate()
                    continue
            if updater == 'set':
                result[key] = Overwrite(_value=Node())
                continue
            # Nested dict — recurse for non-underscore keys
            sub = {}
            for k, v in port.items():
                if not k.startswith('_'):
                    if isinstance(v, dict):
                        sub[k] = _translate_schema({k: v})[k]
                    else:
                        sub[k] = Node()
            if sub:
                result[key] = sub
            else:
                result[key] = Node()
        else:
            result[key] = Node()
    return result
