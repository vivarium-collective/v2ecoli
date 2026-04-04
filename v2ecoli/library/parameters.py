"""Stub parameters module — only needed for antibiotic simulations."""

class _ParamStore:
    def get(self, key):
        from collections import namedtuple
        P = namedtuple('P', ['magnitude'])
        return P(magnitude=0.0)

param_store = _ParamStore()
