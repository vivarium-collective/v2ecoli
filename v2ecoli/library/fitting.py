"""Fitting utilities for v2ecoli."""

import numpy as np


def normalize(array):
    """Normalize array by its L1 norm."""
    return np.array(array).astype("float") / np.linalg.norm(array, 1)
