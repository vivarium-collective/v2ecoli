"""Regression test for the Evolver-before-Requester ordering edge case in
``v2ecoli.steps.partition.Evolver.update``.

On fast-growth media the Evolver can fire before its paired Requester ever
has, so the wrapped process's ``request_set`` attribute may not be set at
all. The stock code either raised ``AttributeError`` on the bare
``process.request_set`` read, or returned ``{}`` (no ``next_update_time``),
which the global clock reads as a non-advancing 0.0 interval and deadlocks
the composite. This used to be carried only as a downstream monkeypatch in
sms-ecoli (``pbg_v2ecoli/_upstream_patches.py``); it is now fixed directly in
``Evolver.update``.
"""
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.fast


def _invoke_update(process, global_time=5.0, timestep=2.0):
    from v2ecoli.steps.partition import Evolver

    fake_self = SimpleNamespace()
    states = {
        "allocate": {},
        "process": (process,),
        "global_time": global_time,
        "timestep": timestep,
    }
    return Evolver.update(fake_self, states)


def test_evolver_update_reschedules_when_request_not_set_attribute_missing():
    """process has no request_set attribute at all (Evolver fired before its
    Requester ever ran) -- must not AttributeError, must reschedule."""
    process = SimpleNamespace()  # deliberately no `request_set`
    result = _invoke_update(process, global_time=5.0, timestep=2.0)
    assert result == {"next_update_time": 7.0}


def test_evolver_update_reschedules_when_request_set_false():
    process = SimpleNamespace(request_set=False)
    result = _invoke_update(process, global_time=10.0, timestep=1.5)
    assert result == {"next_update_time": 11.5}


def test_evolver_update_falls_back_to_one_timestep_when_timestep_falsy():
    process = SimpleNamespace()
    result = _invoke_update(process, global_time=3.0, timestep=0.0)
    assert result == {"next_update_time": 4.0}


def test_evolver_update_no_process_returns_empty_dict():
    """Preserve existing behavior: no process wired at all -> {}."""
    from v2ecoli.steps.partition import Evolver

    fake_self = SimpleNamespace()
    states = {"allocate": {}, "process": (), "global_time": 5.0, "timestep": 2.0}
    assert Evolver.update(fake_self, states) == {}
