"""Tests for .pbg model files.

Validates that .pbg files exist, are loadable, and match the current
codebase. Regenerates them if out of date.
"""

import os
import json
import pytest
from tests.conftest import skip_no_cache

PBG_DIR = 'models'
DEP_PBG = os.path.join(PBG_DIR, 'departitioned.pbg')
PART_PBG = os.path.join(PBG_DIR, 'partitioned.pbg')


def generate_pbg(cache_dir='out/cache'):
    """Generate both .pbg files from the current codebase."""
    from v2ecoli.composite import make_composite
    from v2ecoli.partitioned import make_partitioned_composite

    os.makedirs(PBG_DIR, exist_ok=True)

    dep = make_composite(cache_dir=cache_dir)
    dep_serialized = dep.core.serialize(dep.schema, dep.state)
    with open(DEP_PBG, 'w') as f:
        json.dump(dep_serialized, f, default=str, indent=2)

    part = make_partitioned_composite(cache_dir=cache_dir)
    part_serialized = part.core.serialize(part.schema, part.state)
    with open(PART_PBG, 'w') as f:
        json.dump(part_serialized, f, default=str, indent=2)

    return DEP_PBG, PART_PBG


@skip_no_cache
class TestPbgFiles:

    def test_departitioned_pbg_exists(self):
        """Departitioned .pbg file should exist."""
        assert os.path.exists(DEP_PBG), (
            f"{DEP_PBG} not found — run: python -c "
            f"'from tests.test_pbg_files import generate_pbg; generate_pbg()'")

    def test_partitioned_pbg_exists(self):
        """Partitioned .pbg file should exist."""
        assert os.path.exists(PART_PBG), (
            f"{PART_PBG} not found — run: python -c "
            f"'from tests.test_pbg_files import generate_pbg; generate_pbg()'")

    def test_departitioned_pbg_loadable(self):
        """Departitioned .pbg should be valid JSON with expected structure."""
        with open(DEP_PBG) as f:
            doc = json.load(f)
        assert 'agents' in doc, "Missing 'agents' key"
        cell = doc['agents']['0']
        assert 'bulk' in cell, "Missing 'bulk' in cell state"

    def test_partitioned_pbg_loadable(self):
        """Partitioned .pbg should be valid JSON with expected structure."""
        with open(PART_PBG) as f:
            doc = json.load(f)
        assert 'agents' in doc, "Missing 'agents' key"
        cell = doc['agents']['0']
        assert 'bulk' in cell, "Missing 'bulk' in cell state"

    def test_departitioned_has_steps(self):
        """Departitioned .pbg should have step edges."""
        with open(DEP_PBG) as f:
            doc = json.load(f)
        cell = doc['agents']['0']
        steps = [k for k, v in cell.items()
                 if isinstance(v, dict) and 'address' in v]
        assert len(steps) > 30, f"Only {len(steps)} steps found"

    def test_partitioned_has_more_steps(self):
        """Partitioned .pbg should have more steps than departitioned."""
        with open(DEP_PBG) as f:
            dep_doc = json.load(f)
        with open(PART_PBG) as f:
            part_doc = json.load(f)

        dep_steps = sum(1 for v in dep_doc['agents']['0'].values()
                       if isinstance(v, dict) and 'address' in v)
        part_steps = sum(1 for v in part_doc['agents']['0'].values()
                        if isinstance(v, dict) and 'address' in v)
        assert part_steps > dep_steps, (
            f"Partitioned ({part_steps}) should have more steps than "
            f"departitioned ({dep_steps})")

    @pytest.mark.slow
    def test_regenerate_matches(self, cache_dir):
        """Freshly generated .pbg should match committed version."""
        from v2ecoli.composite import make_composite

        # Build fresh
        dep = make_composite(cache_dir=cache_dir)
        fresh = dep.core.serialize(dep.schema, dep.state)

        # Load committed
        with open(DEP_PBG) as f:
            committed = json.load(f)

        # Compare step names (structure test, not exact state match)
        fresh_cell = fresh['agents']['0']
        committed_cell = committed['agents']['0']
        fresh_steps = sorted(k for k, v in fresh_cell.items()
                           if isinstance(v, dict) and 'address' in v)
        committed_steps = sorted(k for k, v in committed_cell.items()
                                if isinstance(v, dict) and 'address' in v)
        assert fresh_steps == committed_steps, (
            f"Step names changed — regenerate .pbg files:\n"
            f"  python -c 'from tests.test_pbg_files import generate_pbg; generate_pbg()'")
