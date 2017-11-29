"""The module contains additional functions for 'ndomsort' and 'nsga3'.

"""

from typing import Sequence, Iterable, Optional, Any, Tuple, List
import random

__all__ = ["is_dominate", "find_low_median", "generate_coeff_convex_hull", "clip_random"]

def is_dominate(leftv : Sequence[Any], rightv : Sequence[Any]) -> bool:
    """Check. Is 'leftv'  to dominate a 'rightv'?

    'leftv' to dominate a 'rightv', if and only if leftv[i] <= rightv[i], for all i in {0,1,..., len(leftv) - 1}, 
    and there exists j in {0,1,...,len(leftv) - 1} : leftv[j] < rightv[j].

    --------------------
    Args:
        'leftv': A first vector of objectives.
        'rightv': A second vector of objectives.

    --------------------
    Returns:
        True if 'leftv' to dominate a 'rightv', otherwise False.
    """

    assert len(leftv) == len(rightv), "'leftv' must have a same length as 'rightv'."

    is_all_values_less_or_eq = True
    is_one_value_less = False

    for i in range(len(leftv)):
        if leftv[i] < rightv[i]:
            is_one_value_less = True
        elif leftv[i] > rightv[i]:
            is_all_values_less_or_eq = False
            break
    return is_all_values_less_or_eq and is_one_value_less

def find_low_median(iterable :  Iterable[Any]) -> Optional[Any]:
    """Find median of sequence, if length of sequence is odd, otherwise the sequence has two median. The median is the smallest value from them.

    --------------------
    Args:
         'iterable': an input sequence.
        
    --------------------
    Returns:
       'None', if length of sequence is equals 0, otherwise median of sequence.
    """
      
    elements = list(iterable)

    if not elements:
        return None

    median_index = (len(elements) - 1) // 2 

    left = 0
    right = len(elements) - 1

    while True:
        if left != right:
            swap_index = random.randint(left, right - 1)
            elements[swap_index], elements[right] = elements[right], elements[swap_index]
        split_elem = elements[right]
        i = left - 1
        for j in range(left, right + 1):
            if elements[j] <= split_elem:
                i += 1
                elements[i], elements[j] = elements[j], elements[i]
        if i < median_index:
            left = i + 1
        elif i > median_index:
            right = i - 1
        else:
            median = elements[i]
            break
    return median

def _generate_coeff_convex_hull_recursive(amount_in_lin_comb : int, step_by_axis : float, level : int = 1, prev_coeff : Tuple[float] = None) -> List[Tuple[float]]:
    """The recursive procedure generates coefficients for the convex hull.

    The algorithm described in the articel:
    Das, Indraneel & Dennis, J. (2000). 
    Normal-Boundary Intersection: A New Method for Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.
    SIAM Journal on Optimization. 8. . 10.1137/S1052623496307510. 

    --------------------
    Args:
         'amount_in_lin_comb': The number of coefficients in the convex hull.
         'step_by_axis': The step between two consecutive values for the all coefficients.
         'level': Recursive level.
         'preff_coeff': The acceptable coefficients in the previous level.

    --------------------
    Returns:
         The list of tuples. Each tuple is coefficients for the convex hull.

    """

    if level == amount_in_lin_comb:
        return [prev_coeff + (1 - sum(prev_coeff),)]

    vector_of_coeff = []

    if prev_coeff is None:
        prev_coeff = tuple()
    
    coeff = 0
    step = step_by_axis
    sum_prev_coeff = sum(prev_coeff)

    while coeff < 1 + step_by_axis - sum_prev_coeff:
        vector_of_coeff.extend(_generate_coeff_convex_hull_recursive(amount_in_lin_comb, step_by_axis, level + 1, prev_coeff + (coeff,)))
        coeff += step

    return vector_of_coeff

def generate_coeff_convex_hull(amount_in_lin_comb : int, step_by_axis : float) -> List[Tuple[float]]:
    """The procedure generates coefficients for the convex hull.

    The algorithm described in the article:
        Das, Indraneel & Dennis, J. (2000). 
        Normal-Boundary Intersection: A New Method for Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems.
        SIAM Journal on Optimization. 8. . 10.1137/S1052623496307510. 

    --------------------
    Args:
         'amount_in_lin_comb': The number of coefficients in the convex hull.
         'step_by_axis': The step between two consecutive values for the all coefficients.

    --------------------
    Returns:
         The list of tuples. Each tuple is coefficients for the convex hull.

    """

    assert amount_in_lin_comb > 0, "'amount_in_lin_comb' must be > 0."
    assert step_by_axis > 0, "'step_by_axis' must be > 0."

    return _generate_coeff_convex_hull_recursive(amount_in_lin_comb, step_by_axis)

def clip_random(number : float, low_b : float, upp_b : float) -> float:
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

    return random.uniform(low_b, upp_b) if number < low_b or number > upp_b else number