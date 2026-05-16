"""v2ecoli process modules.

Lightweight discovery-safe exports — these imports do NOT pull in the vEcoli
ParCa stack or any Cython extensions, so ``bigraph_schema.discover_packages``
can register them even in environments that only have ``process-bigraph`` and
``bigraph-schema`` installed.
"""
from v2ecoli.processes.chromosome_initiation import DnaABinder, ChromosomePartition

__all__ = [
    "DnaABinder",
    "ChromosomePartition",
]
