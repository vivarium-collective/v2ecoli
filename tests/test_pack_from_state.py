import json

import pytest

from v2ecoli.structural.build import bulk_to_counts, pack_from_state


def test_bulk_to_counts_strips_compartments_and_sums():
    import numpy as np
    bulk = np.array([("EG10893-MONOMER[c]", 10), ("EG10893-MONOMER[m]", 5),
                     ("CPLX0-3964[c]", 3)],
                    dtype=[("id", "U40"), ("count", "i8")])
    counts = bulk_to_counts(bulk)
    assert counts["EG10893-MONOMER"] == 15   # summed across compartments
    assert counts["CPLX0-3964"] == 3


@pytest.mark.slow
def test_pack_from_state_writes_valid_pack(tmp_path):
    counts = {"EG10893-MONOMER": 5000, "CPLX0-3964": 500}
    pack_from_state(str(tmp_path), "initial", counts, volume_fl=1.0, top_n=2)
    pack = json.loads((tmp_path / "initial.pack.json").read_text())
    assert pack["format"] == "parsimony.pack.v1"
    assert "ingredients" in pack and "placements" in pack
