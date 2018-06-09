import random

import numpy as np

__all__ = ["clip_random", "asf"]


def clip_random(number: float, low_b: float, upp_b: float) -> float:
    """A random clipping.

    --------------------
    Args:
         'number': The number for the clipping.
         'low_b': The lower bound.
         'upp_b': The upper bound.

    --------------------
    Returns:
         'random.uniform(low_b, upp_b)' if  'number' < 'low_b' or 'number' > 'upp_b', otherwise 'number'.
    """
    assert low_b <= upp_b, "The lower bound must be less or equal then the upper bound."

    return random.uniform(low_b, upp_b) if number < low_b or number > upp_b else number


def asf(fitness: np.ndarray, weights: np.ndarray) -> float:
    return (fitness / weights).max()
