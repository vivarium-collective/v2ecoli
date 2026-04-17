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
