"""Wrapper around vEcoli's ecoli_master_sim.main() that runtime-patches
``numpy.in1d`` to ``numpy.isin`` (removed in numpy 2.x) so vEcoli's
chromosome_replication runs cleanly under the v2ecoli environment.

The installed vEcoli package is NOT modified. The patch only lives in
this process. Use the same CLI flags as ecoli_master_sim.py.
"""
import numpy as np

# numpy.in1d was removed in numpy 2.x. vEcoli's chromosome_replication.py
# still uses it. np.isin is the drop-in replacement.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from ecoli.experiments.ecoli_master_sim import main

if __name__ == "__main__":
    main()
