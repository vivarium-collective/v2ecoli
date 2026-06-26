"""Unit tests for classify_domains — pure chromosome-index + is_daughter classifier.

Tests mirror the exact example in task-BF2-1-brief.md.
"""
import numpy as np
import pytest
from v2ecoli.structural.build import classify_domains


def test_classify_domains_two_chromosomes():
    # chromosome A root=1 -> children 3,4 ; chromosome B root=2 -> children 5,6
    tree = {1: [3, 4], 2: [5, 6]}
    full_chrom_domains = [1, 2]
    q = np.array([1, 3, 4, 2, 5, 6], dtype="i4")
    ci, isd = classify_domains(tree, full_chrom_domains, q)
    assert list(ci) == [0, 0, 0, 1, 1, 1]       # 1/3/4 -> chrom0 ; 2/5/6 -> chrom1
    assert list(isd) == [False, True, True, False, True, True]  # roots not daughters
    assert isd.dtype == bool and ci.dtype == np.int32


def test_classify_domains_unmatched_domain():
    """A domain not in any chromosome's lineage → chromosome_index=0, is_daughter=False."""
    tree = {1: [3, 4]}
    full_chrom_domains = [1]
    q = np.array([99], dtype="i4")
    ci, isd = classify_domains(tree, full_chrom_domains, q)
    assert ci[0] == 0
    assert isd[0] == False


def test_classify_domains_single_chromosome_no_replication():
    """Birth state: one chromosome, no children (domain 0 is the root)."""
    tree = {}
    full_chrom_domains = [0]
    q = np.array([0, 0, 0], dtype="i4")
    ci, isd = classify_domains(tree, full_chrom_domains, q)
    assert list(ci) == [0, 0, 0]
    assert list(isd) == [False, False, False]


def test_classify_domains_empty_query():
    """Empty RNAP array returns empty arrays with correct dtypes."""
    tree = {1: [2, 3]}
    full_chrom_domains = [1]
    q = np.array([], dtype="i4")
    ci, isd = classify_domains(tree, full_chrom_domains, q)
    assert len(ci) == 0
    assert len(isd) == 0
    assert ci.dtype == np.int32
    assert isd.dtype == bool


def test_classify_domains_transitive_descendants():
    """Descendants must be transitive: root→2→4, root→3, so 4 is also in chrom 0."""
    tree = {1: [2, 3], 2: [4]}
    full_chrom_domains = [1]
    q = np.array([1, 2, 3, 4], dtype="i4")
    ci, isd = classify_domains(tree, full_chrom_domains, q)
    assert list(ci) == [0, 0, 0, 0]
    assert list(isd) == [False, True, True, True]
