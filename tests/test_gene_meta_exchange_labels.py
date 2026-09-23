"""The exchange vector's one true ordering, pinned against a committed fixture.

The 87 external exchange fluxes arrive as a POSITIONAL array with no ids in the
parquet, exactly like the omics vectors. Getting the order wrong does not fail —
it attributes every flux to the wrong metabolite and produces a plausible-looking
comparison. Until ``gene_meta.exchange_labels`` existed the ordering lived only
inside one pin script, so a second consumer had to re-derive it.

No sim_data and no sweep: the ids come from the ``flux_ids`` list pinned in the
committed basal reference, which is the same list the pin script wrote out of a
real ParCa state.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from v2ecoli.library.gene_meta import exchange_labels

_REPO = Path(__file__).resolve().parents[1]
_REFERENCE = _REPO / "tests" / "fixtures" / "population_phenotype_basal_reference.json"


def _pinned_flux_ids() -> list[str]:
    ref = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    return ref["axes"]["fluxes.exchange"]["criterion"]["flux_ids"]


class _StubSimData(dict):
    """The two accesses ``exchange_labels`` makes: ``sim_data["external_state"]``
    then ``.all_external_exchange_molecules``. A stub rather than a real ParCa
    state so this runs in CI with no pickle."""

    def __init__(self, molecules):
        super().__init__(external_state=type("ExternalState", (), {
            "all_external_exchange_molecules": list(molecules)})())


class ExchangeLabels(unittest.TestCase):

    def test_the_order_is_sorted_regardless_of_the_order_sim_data_holds_them_in(self):
        """WOULD CATCH: returning sim_data's own iteration order.

        The set is stored unordered, so an implementation that just lists it
        would agree with ``sorted`` by luck on some builds and not others — the
        worst kind of pass. Feeding it REVERSED input makes luck impossible.
        """
        pinned = _pinned_flux_ids()
        self.assertEqual(len(pinned), 87)
        got = exchange_labels(_StubSimData(reversed(pinned)))
        self.assertEqual(got, pinned)

    def test_the_pinned_reference_itself_is_in_sorted_order(self):
        """The premise the test above rests on, asserted rather than assumed:
        if the committed fixture were NOT sorted, agreeing with it would prove
        the opposite of what we want."""
        pinned = _pinned_flux_ids()
        self.assertEqual(pinned, sorted(pinned))

    def test_the_positions_the_render_script_hardcodes_still_point_where_it_thinks(self):
        """★ The check that makes this ordering falsifiable rather than merely
        stated. ``render_basal_vs_literature`` slices glucose, CO2 and acetate by
        1-indexed POSITION (37 / 11 / 3) — three magic numbers that are only
        correct under this exact ordering and that fail silently if it changes.

        WOULD CATCH: a reordering of the exchange vector from either direction —
        this helper drifting from the render script, or the underlying molecule
        set changing so that both are wrong together against the pinned fixture.
        """
        import re

        src = (_REPO / "scripts" / "render_basal_vs_literature.py").read_text(
            encoding="utf-8")
        m = re.search(r"_GLC_IDX,\s*_CO2_IDX,\s*_ACET_IDX\s*=\s*(\d+),\s*(\d+),\s*(\d+)",
                      src)
        self.assertIsNotNone(m, "the render script's flux index constants moved")
        glc, co2, acet = (int(g) for g in m.groups())

        labels = exchange_labels(_StubSimData(reversed(_pinned_flux_ids())))
        # 1-indexed in the script (DuckDB's list_extract), 0-indexed here.
        self.assertEqual(labels[glc - 1], "GLC[p]")
        self.assertEqual(labels[co2 - 1], "CARBON-DIOXIDE[p]")
        self.assertEqual(labels[acet - 1], "ACET[p]")


if __name__ == "__main__":
    unittest.main()
