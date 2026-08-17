"""Tests for the parameterized `quantity[...]` schema syntax.

Matches the pattern vEcoli's `UnumUnits` uses for `unum[<mag>,<unit>]`,
but backed by pint (v2ecoli is pint-first — see AGENTS.md:50 and the
save-state format memory). Enables config_schema declarations like::

    'n_avogadro': 'quantity[float,1/mol]'
    'cell_density': 'quantity[g/L]'
    'trna_concs': 'quantity[array[float],mmol/L]'

Verifies:
  1. Two-parameter form `quantity[<mag>,<unit>]` parses and populates
     both `magnitude` and `_units`.
  2. Single-parameter shorthand `quantity[<unit>]` sets `_units` and
     defaults the magnitude to Float.
  3. Rendering a parameterized schema round-trips back to the string
     form.
  4. The bare `quantity` (no parameters) still works for inferred
     schemas.
"""
from __future__ import annotations

import pytest
from bigraph_schema.core import Core, BASE_TYPES
from bigraph_schema.schema import Float

from v2ecoli.types import ECOLI_TYPES
from v2ecoli.types.quantity import Quantity


pytestmark = pytest.mark.fast


@pytest.fixture(scope='module')
def core():
    # Skip bigraph-schema's package discovery — it traverses installed
    # packages and trips on the Cython-gated test modules under
    # v2ecoli/processes/parca/wholecell/tests/. Constructing the core
    # directly from BASE_TYPES + ECOLI_TYPES is enough to exercise
    # schema parsing.
    c = Core(BASE_TYPES)
    c.register_types(ECOLI_TYPES)
    return c


def test_two_param_form_parses(core):
    """`quantity[float,1/mol]` yields a Quantity with Float magnitude
    and _units='1/mol'."""
    schema = core.access('quantity[float,1/mol]')
    assert isinstance(schema, Quantity)
    assert schema._units == '1/mol'
    assert isinstance(schema.magnitude, Float)


def test_single_param_is_unit_string(core):
    """`quantity[g/L]` treats the single parameter as the unit string;
    magnitude defaults to Float."""
    schema = core.access('quantity[g/L]')
    assert isinstance(schema, Quantity)
    assert schema._units == 'g/L'
    assert isinstance(schema.magnitude, Float)


def test_bare_quantity_still_works(core):
    """`quantity` with no parameters yields an un-parameterized schema.
    The _units field is empty; dispatchers that run on inferred schemas
    still populate the dict-form `units` field from a pint value."""
    schema = core.access('quantity')
    assert isinstance(schema, Quantity)
    assert schema._units == ''


def test_render_roundtrips_parameterized_form(core):
    """A parameterized schema renders back to `quantity[<mag>,<unit>]`
    form so the declared type survives inspection/serialization."""
    from bigraph_schema.methods import render
    schema = core.access('quantity[float,mmol/L]')
    # Float magnitude is short-form: `quantity[<unit>]`
    assert render(schema) == 'quantity[mmol/L]'

    schema2 = core.access('quantity[integer,count]')
    assert render(schema2) == 'quantity[integer,count]'


def test_serialize_state_handles_raw_float(core):
    """A real state value can arrive as a bare Python float rather than a
    pint Quantity (e.g. a scalar that was never wrapped) -- _serialize_state
    must treat it the same as its existing raw-int branch, not fall through
    to the Quantity-assumed branch and crash on `.magnitude`.

    Regression test for backlog item 56: a real chain-dispatch generation job
    crashed at the very last step (writing final_state.json, after
    composite.run() had already completed) with `AttributeError: 'float'
    object has no attribute 'magnitude'` -- confirmed via real CloudWatch
    logs against the deployed commit, not a synthetic scenario.
    """
    schema = core.access('quantity[fg]')
    result = schema._serialize_state(1234.5)
    assert result == {'units': schema.units, 'magnitude': 1234.5}


def test_serialize_state_still_handles_raw_int(core):
    """The pre-existing raw-int branch must keep working unchanged."""
    schema = core.access('quantity[count]')
    result = schema._serialize_state(7)
    assert result == {'units': schema.units, 'magnitude': 7}


def test_serialize_state_handles_real_quantity(core):
    """The Quantity branch (a real pint value with .magnitude) must keep
    working unchanged -- this is the common real case."""
    from v2ecoli.types.quantity import ureg as units

    schema = core.access('quantity[fg]')
    result = schema._serialize_state(3.0 * units.fg)
    assert result == {'units': schema.units, 'magnitude': 3.0}


def test_reify_populates_units_dict(core):
    """`reify_schema` must populate both `_units` (string) and `units`
    (dict form derived from pint) so downstream `realize` can wrap bare
    numeric values with the correct unit — otherwise a declared
    `quantity[1/mol]` field realizes bare floats as dimensionless."""
    schema = core.access('quantity[float,1/mol]')
    assert schema._units == '1/mol'
    assert schema.units  # non-empty
    # pint yields {'mole': -1} for `1/mol` — exact keys depend on pint's
    # canonical form, but the mole dimension must be present as -1.
    assert schema.units.get('mole') == -1

    schema2 = core.access('quantity[g/L]')
    assert schema2.units  # populated despite short-form syntax
