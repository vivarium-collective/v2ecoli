import os
import pytest

from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import (
    KnowledgeBaseEcoli,
)
from v2ecoli.processes.parca.reconstruction.ecoli.sources import SourceBundle

FLAT = "v2ecoli/processes/parca/reconstruction/ecoli/flat"


def _snapshot(kb):
    """Flatten KB into a comparable dict of {attr_path: value}."""
    from v2ecoli.processes.parca.reconstruction.ecoli.knowledge_base_raw import DataStore
    out = {}

    def walk(obj, prefix):
        for name, val in vars(obj).items():
            if name.startswith("_"):
                continue  # skip private implementation attrs (e.g. _bundle)
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(val, DataStore):
                walk(val, key)
            else:
                out[key] = val
    walk(kb, "")
    return out


FLAGS = dict(operons_on=True, remove_rrna_operons=False, remove_rrff=False, stable_rrna=False)


@pytest.mark.skipif(not os.path.isdir(FLAT), reason="local flat/ deleted post-migration")
def test_bundle_kb_matches_legacy_kb():
    legacy = KnowledgeBaseEcoli(**FLAGS)              # reads local flat/
    bundled = KnowledgeBaseEcoli(bundle=SourceBundle(), **FLAGS)
    a, b = _snapshot(legacy), _snapshot(bundled)
    assert a.keys() == b.keys(), set(a) ^ set(b)
    diffs = [k for k in a if repr(a[k]) != repr(b[k])]
    assert diffs == [], f"raw_data differs for: {diffs[:10]}"
