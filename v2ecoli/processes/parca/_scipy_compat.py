"""scipy cross-version compatibility for pickled interpolators.

scipy's array-API refactor (1.15+) reimplemented ``PPoly`` / ``CubicSpline`` so
that ``.c`` and ``.x`` became *properties* backed by an internal ``_delegate_to``
object (plus ``_xp`` / ``_xp_internal``) constructed in ``__init__``.  An
interpolator **pickled by an older scipy** stored ``c`` / ``x`` as plain instance
attributes and carries none of the new-format guts.  Unpickled and then *called*
under the new scipy it raises, e.g.::

    AttributeError: 'CubicSpline' object has no attribute '_delegate_to'
    AttributeError: 'CubicSpline' object has no attribute '_xp'

ParCa ``sim_data`` embeds such interpolators (the growth-rate-dependent mass
fits in ``growth_rate_dependent_parameters.py`` build ``CubicSpline``s that get
pickled into the state), so a state pickled *before* a scipy upgrade breaks the
analysis viewers — even though the simulation itself, which rebuilds/uses its own
sim_data, is fine.

``install()`` adds a lazy ``__getattr__`` to ``PPoly`` / ``BPoly`` that, the first
time a stale instance is touched, rebuilds an equivalent new-format object from
its stored ``(c, x)`` and transplants that object's ``__dict__`` in place — after
which every method (``__call__``, ``derivative``, ``integrate``, re-pickling, …)
works.  Fresh instances already carry ``_delegate_to`` so the hook never fires
for them.  Idempotent, and a no-op on scipy versions that predate the refactor
(where nothing needs bridging).
"""

from __future__ import annotations


def install() -> None:
    """Idempotently install the delegate-rehydration hook (see module docstring).

    Safe to call repeatedly and safe on any scipy version — it self-disables
    when scipy/numpy are absent or when the running scipy predates the
    delegate-backed interpolator design.
    """
    try:
        import numpy as np
        from scipy.interpolate import BPoly, PPoly
    except Exception:
        return

    # Only the new, delegate-backed scipy needs bridging: a freshly built PPoly
    # there carries ``_delegate_to``.  On older scipy a pickled interpolator IS
    # already in the format the running scipy expects, so there is nothing to do.
    try:
        probe = PPoly(np.zeros((1, 1)), np.array([0.0, 1.0]))
        if "_delegate_to" not in probe.__dict__:
            return
    except Exception:
        return

    def _flatten_state(state):
        """Merge a pickle *state* into one attr mapping.

        Handles both shapes seen across scipy versions: a plain ``__dict__``
        mapping (new, dict-based) and the ``(dict_state, slots_state)`` 2-tuple
        the old ``__slots__``-based ``PPoly`` produced.
        """
        merged = {}
        if isinstance(state, tuple) and len(state) == 2:
            for part in state:
                if isinstance(part, dict):
                    merged.update(part)
        elif isinstance(state, dict):
            merged.update(state)
        return merged

    def _old_coeffs(merged):
        """``(c, x, extrapolate, axis)`` from an old-format attr map, or None.

        Old scipy stored coefficients in ``__slots__`` as ``_c`` / ``_x`` (public
        ``c`` / ``x`` were properties); some builds used plain ``c`` / ``x``.
        """
        c = merged.get("_c", merged.get("c"))
        x = merged.get("_x", merged.get("x"))
        if c is None or x is None:
            return None
        return c, x, merged.get("extrapolate", True), merged.get("axis", 0)

    def _rebuild_dict(base, c, x, extrapolate, axis):
        """New-format ``__dict__`` for a delegate-backed object from coeffs."""
        rebuilt = base(c, x, extrapolate=bool(extrapolate), axis=int(axis))
        return dict(rebuilt.__dict__)

    def _apply_plain(self, state):
        """Default state restore (new-format), tolerant of the 2-tuple shape."""
        if isinstance(state, tuple) and len(state) == 2:
            base_dict, slots = state
            if isinstance(base_dict, dict):
                self.__dict__.update(base_dict)
            if isinstance(slots, dict):
                self.__dict__.update(slots)
        elif isinstance(state, dict):
            self.__dict__.update(state)

    def _make_setstate(base):
        def __setstate__(self, state):
            # An OLD-scipy pickle restores through here with a state that lacks
            # the new-format guts; rebuild them from the stored coefficients so
            # the property setters (which read ``_delegate_to``) never run
            # against a half-built object. New-format states pass through.
            merged = _flatten_state(state)
            if "_delegate_to" not in merged:
                coeffs = _old_coeffs(merged)
                if coeffs is not None:
                    self.__dict__.update(_rebuild_dict(base, *coeffs))
                    return
            _apply_plain(self, state)
        return __setstate__

    def _make_getattr(base):
        def __getattr__(self, name):
            # Belt-and-suspenders for any stale object built via ``__new__`` +
            # direct ``__dict__`` assignment (not the pickle path): rehydrate on
            # first missing-attr access.
            d = object.__getattribute__(self, "__dict__")
            if "_delegate_to" not in d:
                coeffs = _old_coeffs(d)
                if coeffs is not None:
                    new_dict = _rebuild_dict(base, *coeffs)
                    d.clear()
                    d.update(new_dict)
                    if name in d:
                        return d[name]
            raise AttributeError(name)
        return __getattr__

    # PPoly covers CubicSpline (a PPoly subclass) via inheritance; BPoly is the
    # Bernstein-basis sibling. Each gets its own basis-correct rebuild closures.
    for cls in (PPoly, BPoly):
        if cls.__dict__.get("_v2e_delegate_shim"):
            continue
        cls.__setstate__ = _make_setstate(cls)
        cls.__getattr__ = _make_getattr(cls)
        cls._v2e_delegate_shim = True
