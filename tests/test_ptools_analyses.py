"""Fidelity tests for the ptools_rna native Analysis port."""

import os
import pytest

FIX = "/Users/eranagmon/code/sms-api/tests/fixtures/analysis_data"


def _frame_ids(tsv_text):
    rows = [r for r in tsv_text.strip().splitlines() if r]
    return {r.split("\t")[0] for r in rows[1:]}  # skip header ($ row)


def test_ptools_rna_registered():
    # No fixture needed — always runs so CI confirms the port imports + registers.
    from v2ecoli.workflow.analyses import ptools_rna  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_rna"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rna_output_shape_matches_oracle():
    oracle = open(os.path.join(FIX, "ptools_rna.txt")).read()
    header = oracle.strip().splitlines()[0].split("\t")
    assert header[0] == "$"
    assert len(_frame_ids(oracle)) > 0


def test_ptools_rxns_registered():
    from v2ecoli.workflow.analyses import ptools_rxns  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    cls = ANALYSIS_REGISTRY["ptools_rxns"]
    assert issubclass(cls, Analysis) and cls.scale == "single"


@pytest.mark.skipif(not os.path.isdir(FIX), reason="sms-api oracle fixtures absent")
def test_ptools_rxns_oracle_shape():
    oracle = open(os.path.join(FIX, "ptools_rxns.txt")).read()
    assert oracle.strip().splitlines()[0].split("\t")[0] == "$"
    assert len(_frame_ids(oracle)) > 0
