"""Classify exceptions raised from ``composite.run()`` as division-vs-not.

Regression guard: the lineage/bridge division detectors used to treat ANY
exception whose message merely contained the substring "divide"/"division" as a
genuine division event — so a ``ZeroDivisionError: float division by zero`` got
silently mislabeled as a division (reported ``divided=True``, duration ~1s) and
the real failure was masked. ``is_division_exception`` excludes builtin
computation errors so real bugs surface instead.
"""
import pytest

from v2ecoli.library.division import is_division_exception, NON_DIVISION_ERRORS

pytestmark = pytest.mark.fast


def test_zero_division_error_is_not_a_division():
    """The canonical trap: 'float division by zero' contains 'division' but is a
    real arithmetic failure, not a cell division."""
    assert is_division_exception(ZeroDivisionError("float division by zero")) is False


@pytest.mark.parametrize("exc", [
    KeyError("division_time"),
    TypeError("unsupported operand for division"),
    ValueError("cannot divide by this"),
    IndexError("division index out of range"),
    AttributeError("object has no attribute 'divide'"),
])
def test_builtin_code_errors_are_never_divisions(exc):
    """Builtin computation/lookup errors are real failures even when their
    message contains the divide/division token."""
    assert is_division_exception(exc) is False
    assert isinstance(exc, NON_DIVISION_ERRORS)


@pytest.mark.parametrize("msg", [
    "cell will divide now",
    "DIVISION event on agents map",
    "structural update: divide the mother",
])
def test_genuine_division_signal_is_detected(msg):
    """A non-builtin exception carrying the divide/division token (the structural
    update process-bigraph raises through) IS a division signal."""
    assert is_division_exception(Exception(msg)) is True


def test_unrelated_exception_is_not_a_division():
    """No token, not a builtin code error -> not a division (caller re-raises)."""
    assert is_division_exception(Exception("emitter flush failed")) is False


def test_runtimeerror_without_token_is_not_a_division():
    assert is_division_exception(RuntimeError("solver did not converge")) is False
