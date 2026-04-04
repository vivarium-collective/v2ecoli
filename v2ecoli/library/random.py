"""Random number utilities for v2ecoli."""

import numpy as np


def stochasticRound(randomState, value):
    """Stochastically round values to integers."""
    value = np.array(value)
    valueShape = value.shape
    valueRavel = np.ravel(value)
    roundUp = randomState.rand(valueRavel.size) < (valueRavel % 1)
    valueRavel[roundUp] = np.ceil(valueRavel[roundUp])
    valueRavel[~roundUp] = np.floor(valueRavel[~roundUp])
    if valueShape != () and len(valueShape) > 1:
        return np.unravel_index(valueRavel, valueShape)
    return valueRavel
