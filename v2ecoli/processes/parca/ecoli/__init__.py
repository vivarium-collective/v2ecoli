"""Trimmed `ecoli` package for v2ecoli.processes.parca.

The original `vivarium-ecoli` `ecoli/__init__.py` registers vivarium emitters,
dividers, serializers, and the parquet emitter — all of which belong to the
simulation side and are not needed to run the ParCa. Only `ecoli.library`
(schema + initial_conditions) and the five `ecoli.processes` modules reached
from `create_bulk_container()` are vendored here, so this `__init__` is empty.
"""
