"""The module contains additional functions for 'nsga3'.

"""

from typing import Sequence, Iterable, Optional, Any, Tuple, List
import random

__all__ = ["generate_coeff_convex_hull",]


def _generate_coeff_convex_hull_recursive(amount_in_lin_comb: int, count_unique_values: int, level: int = 1,
                                          prev_m: Tuple[int] = None, prev_coeff: Tuple[float] = None)\
        -> List[Tuple[float]]:
    """The recursive procedure generates coefficients for the convex hull.

    The algorithm described in the articel:
    Das, Indraneel & Dennis, J. (2000). 
    Normal-Boundary Intersection: A New Method for Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.
    SIAM Journal on Optimization. 8. . 10.1137/S1052623496307510. 

    --------------------
    Args:
         'amount_in_lin_comb': The number of coefficients in the convex hull.
         'count_unique_values': The amount of the unique values of each coefficient in the convex hull. 
                                For example, if 'amount_unique_values' = 2, then the first coefficient is {0, 1},
                                if 'amount_unique_values' = 3, then it is {0, 0.5, 1}.
                                Similarly for the rest.
         'level': Recursive level.
         'prev_m': The acceptable multipliers of the step in the previous level.
         'prev_coeff': The acceptable coefficients in the previous level.

    --------------------
    Returns:
         The list of tuples. Each tuple is coefficients for the convex hull.

    """
    if prev_coeff is None:
        prev_coeff = tuple()

    if level == amount_in_lin_comb:
        return [prev_coeff + (1 - sum(prev_coeff),)]

    vector_of_coeff = []

    if prev_m is None:
        prev_m = tuple()

    step = 1 / (count_unique_values - 1)

    for i in range(count_unique_values - sum(prev_m)):
        coeff = i * step
        vector_of_coeff.extend(_generate_coeff_convex_hull_recursive(amount_in_lin_comb, count_unique_values, level + 1
                                                                     , (i,) + prev_m, (coeff,) + prev_coeff))

    return vector_of_coeff


def generate_coeff_convex_hull(amount_in_lin_comb: int, amount_unique_values: int) -> List[Tuple[float]]:
    """The procedure generates coefficients for the convex hull.

    The algorithm described in the article:
        Das, Indraneel & Dennis, J. (2000). 
        Normal-Boundary Intersection: A New Method for Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.
        SIAM Journal on Optimization. 8. . 10.1137/S1052623496307510.
    

    --------------------
    Args:
         'amount_in_lin_comb': The number of coefficients in the convex hull.
         'count_unique_values': The amount of the unique values of each coefficient in the convex hull. 
                                For example, if 'amount_unique_values' = 2, then the first coefficient is {0, 1},
                                if 'amount_unique_values' = 3, then it is {0, 0.5, 1}.
                                Similarly for the rest.

    --------------------
    Returns:
         The list of tuples. Each tuple is coefficients for the convex hull.

    """

    assert amount_in_lin_comb > 0, "'amount_in_lin_comb' must be > 0."
    assert amount_unique_values > 1, "'amount_unique_values' must be > 1."

    return _generate_coeff_convex_hull_recursive(amount_in_lin_comb, amount_unique_values)


