"""The module contains additional functions for 'ndomsort' and 'nsga3'.

"""

from typing import Sequence, Iterable, TypeVar, Optional
import random

T = TypeVar('T')

__all__ = ["is_dominate", "find_low_median"]


def is_dominate(leftv : Sequence[T], rightv : Sequence[T]) -> bool:
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


def find_low_median(iterable :  Iterable[T]) -> Optional[T]:
    """Find median of sequence, if length of sequence is odd, otherwise the sequence has two median. The median is the smallest value from them.

    --------------------
    Args:
         'iterable': an input sequence.
        
    --------------------
    Returns:
       'None', if length of sequence equals 0, otherwise median of sequence.
    """
      
    elements = list(iterable)

    if len(elements) == 0:
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