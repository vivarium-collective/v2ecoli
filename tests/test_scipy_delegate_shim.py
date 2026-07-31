"""Regression tests for the scipy interpolator cross-version pickle shim.

scipy's array-API refactor (1.15+) made ``CubicSpline``/``PPoly`` delegate to an
internal ``_delegate_to`` built in ``__init__``.  ParCa states pickled by an
older scipy carry the old slotted layout (``_c`` / ``_x``) and no delegate, and
break when unpickled + called under the new scipy.  ``_scipy_compat.install()``
bridges them.  See ``v2ecoli/processes/parca/_scipy_compat.py``.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.interpolate import CubicSpline, PPoly  # noqa: E402

from v2ecoli.processes.parca import _scipy_compat  # noqa: E402


def _is_new_scipy():
    _scipy_compat.install()
    return "_delegate_to" in PPoly(np.zeros((1, 1)), np.array([0.0, 1.0])).__dict__


def test_old_slotted_pickle_state_rehydrates():
    """An old ``(None, {_c, _x, axis, extrapolate})`` state must rebuild + call."""
    _scipy_compat.install()
    if not _is_new_scipy():
        pytest.skip("scipy predates the delegate-backed interpolator design")

    truth = CubicSpline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0], bc_type="natural")
    old_state = (None, {
        "_c": np.array(truth.c), "_x": np.array(truth.x),
        "axis": 0, "extrapolate": True,
    })

    obj = CubicSpline.__new__(CubicSpline)
    obj.__setstate__(old_state)

    xs = np.array([0.3, 1.5, 2.7])
    assert np.allclose(obj(xs), truth(xs))
    assert np.isclose(float(obj.integrate(0, 3)), float(truth.integrate(0, 3)))
    assert np.allclose(obj.c, truth.c) and np.allclose(obj.x, truth.x)


def test_stale_dict_object_rehydrates_lazily():
    """A stale object built via __new__ + __dict__ (no delegate) heals on call."""
    _scipy_compat.install()
    if not _is_new_scipy():
        pytest.skip("scipy predates the delegate-backed interpolator design")

    truth = CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    stale = CubicSpline.__new__(CubicSpline)
    stale.__dict__.update({"_c": np.array(truth.c), "_x": np.array(truth.x),
                           "axis": 0, "extrapolate": True})
    assert np.isclose(float(stale(1.5)), float(truth(1.5)))


def test_fresh_objects_unaffected():
    """The shim must never fire for a freshly built interpolator."""
    _scipy_compat.install()
    cs = CubicSpline([0.0, 1.0, 2.0], [0.0, 2.0, 8.0])
    if _is_new_scipy():
        assert "_delegate_to" in cs.__dict__  # already new-format; no rehydration
    # Sane values regardless of scipy version.
    assert np.isfinite(float(cs(0.5)))
    assert np.isclose(float(cs(1.0)), 2.0)  # interpolates the knot exactly


def test_install_is_idempotent():
    _scipy_compat.install()
    _scipy_compat.install()  # must not raise or double-wrap
    # On the new (delegate-backed) scipy the shim installs; on older scipy there
    # is nothing to bridge and install() is a deliberate no-op.
    assert getattr(PPoly, "_v2e_delegate_shim", False) is _is_new_scipy()
