"""Module is implementation of none-dominated sorting.

Original algorithm described in the paper:

Buzdalov M., Shalyto A.
A Provably Asymptotically Fast Version of the Generalized Jensen Algorithm for Non-dominated Sorting 
// Parallel Problem Solving from Nature XIII.
- 2015. - P. 528-537. - (Lecture Notes on Computer Science ; 8672)

"""

from typing import List, Iterable, Tuple, Sequence, Callable, Dict, Any
from collections import defaultdict
import itertools

import stools as st

__all__ = ["non_domin_sort"]


def _is_seq_has_one_uniq_value(iterable : Iterable[Any]) -> bool:
    """Check. Has 'iterable' only a one unique value?

    It is equivalent following: 'len({item for item in iterable}) == 1'.

    --------------------
    Args:
         'iterable': an input sequence.
         
    --------------------
    Returns:
         True, if 'iterable' contains only one unique value, otherwise False.

    """

    iterator = iter(iterable)

    try:
        first_value = next(iterator)

        is_has_uniq_value = True

        while True:
            value = next(iterator)
            if value != first_value:
                is_has_uniq_value = False
                raise StopIteration
    except StopIteration:
        pass

    return is_has_uniq_value
 
def _split_by(seq_fitness_front : dict, indices : List[int], split_value : Any, count_of_obj : int) -> Tuple[List[int], List[int], List[int]]:
    """'indices' splits into three lists.
   
   The three lits: the list of indices, where 'count_of_obj - 1'th value of fitness is less than a 'split_value', the list of indices,
   where 'count_of_obj - 1'th value of fitness is equal a 'split_value', the list of indices, where 'count_of_obj - 1'th value of fitness is greater than a 'split_value'.

    --------------------
    Args:
         'seq_fitness_front': a dictionary contains fitness and front index.
         'indices': indices in 'seq_fitness_front'.
         'split_value': a value for splitting 'indices'.
         'count_of_obj':   
         
    --------------------
    Returns:
         The tuple of lists of the indices.

    """

    indices_less_split_value = [index for index in indices if seq_fitness_front[index]["fitness"][count_of_obj - 1] < split_value]

    indices_greater_split_value = [index for index in indices if seq_fitness_front[index]["fitness"][count_of_obj - 1] > split_value ]

    indices_equal_split_value = [index for index in indices  if index not in indices_greater_split_value and index not in indices_less_split_value]

    return indices_less_split_value, indices_equal_split_value, indices_greater_split_value

def _sweep_a(seq_fitness_front : dict, indices : List[int]) -> None:
    """Two-objective sorting.

    It attributes front index to lexicographically ordered fitnesses in the  'seq_fitness_front', with the indices in the 'indices', 
    based on the first two values of the fitness using a line-sweep algorithm.
    
    --------------------
    Args:
         'seq_fitness_front': a dictionary contains fitness and front index.
         'indices': indices in 'seq_fitness_front'.
         
    --------------------
    Returns:
         None

    """
    init_ind = set((indices[0],))

    for k  in range(1, len(indices)):
        i = indices[k]
        u_ind = [index for index in init_ind if seq_fitness_front[index]["fitness"][1] <= seq_fitness_front[i]["fitness"][1]]
        if u_ind:
            max_front = max(seq_fitness_front[index]["front"] for index in u_ind)
            seq_fitness_front[i]["front"] = max(seq_fitness_front[i]["front"], max_front + 1)

        init_ind -= { index for index in init_ind if seq_fitness_front[index]["front"] == seq_fitness_front[i]["front"] }
        init_ind.add(i)

def _sweep_b(seq_fitness_front : dict, comp_indices : List[int], assign_indices : List[int]) -> None:
    """Two-objective sorting procedure.

    It attributes front indices to fitness in the 'seq_fitness_front', with the indices in the 'assign_indices', based on the first two values of the fitness
    by comparing them to fitnesses, with the indices in the  'comp_indices', using a line-sweep algorithm.
        
    --------------------
    Args:
         'seq_fitness_front': a dictionary contains fitness and front index.
         'comp_indices': indices for comparing.
         'assign_indices': indices for assign front.
         
    --------------------
    Returns:
         None

    """

    init_ind = set()
    p = 0

    for j in assign_indices:
        if p < len(comp_indices):
            fitness_right = seq_fitness_front[j]["fitness"][:2]
            
        while p < len(comp_indices):
            i = comp_indices[p]
            fitness_left = seq_fitness_front[i]["fitness"][:2]
            if fitness_left <= fitness_right:
                r = { index for index in init_ind if seq_fitness_front[index]["front"] == seq_fitness_front[i]["front"]
                      and seq_fitness_front[index]["fitness"][1] < seq_fitness_front[i]["fitness"][1] }

                if not r:
                    init_ind -= { index for index in init_ind if seq_fitness_front[index]["front"] == seq_fitness_front[i]["front"] }
                    init_ind.add(i)
                p += 1
            else:
                break
        u = {index for index in init_ind if seq_fitness_front[index]["fitness"][1] <= seq_fitness_front[j]["fitness"][1] }

        if u:
            max_front = max((seq_fitness_front[index]["front"] for index in u))
            seq_fitness_front[j]["front"] = max(seq_fitness_front[j]["front"], max_front + 1)

def _nd_helper_a(seq_fitness_front : dict, indices : List[int] , count_of_obj : int) -> None:
    """Recursive procedure.

    It attributes front indices to all fitnesses in the 'seq_fitness_front', with the indices in the 'indices', for the first 'count_of_obj' values of the fitness.
        
    --------------------
    Args:
         'seq_fitness_front': a dictionary contains fitness and front index.
         'indices': indices for assign front.
         'count_of_obj': the count of values of the fitness for assign front.
         
    --------------------
    Returns:
         None

    """

    if len(indices) < 2:
        return
    elif len(indices) == 2:
        index_l, index_r = indices[0], indices[1]
        fitness1, fitness2 = seq_fitness_front[index_l]["fitness"][:count_of_obj], seq_fitness_front[index_r]["fitness"][:count_of_obj]

        if st.is_dominate(fitness1, fitness2):
            seq_fitness_front[index_r]["front"] = max(seq_fitness_front[index_r]["front"], seq_fitness_front[index_l]["front"] + 1)
    elif count_of_obj == 2:
        _sweep_a(seq_fitness_front, indices)
    elif _is_seq_has_one_uniq_value(seq_fitness_front[index]["fitness"][count_of_obj - 1]  for index in indices):
        _nd_helper_a(seq_fitness_front, indices, count_of_obj - 1)
    else:
        median = st.find_low_median((seq_fitness_front[index]["fitness"][count_of_obj - 1] for index in indices))
       
        less_median, equal_median, greater_median = _split_by(seq_fitness_front, indices, median, count_of_obj)

        _nd_helper_a(seq_fitness_front, less_median, count_of_obj)
        _nd_helper_b(seq_fitness_front, less_median, equal_median, count_of_obj - 1)
        _nd_helper_a(seq_fitness_front, equal_median, count_of_obj - 1)
        _nd_helper_b(seq_fitness_front, less_median + equal_median, greater_median, count_of_obj - 1)
        _nd_helper_a(seq_fitness_front, greater_median, count_of_obj)

def _nd_helper_b(seq_fitness_front : dict, comp_indices : List[int], assign_indices : List[int], count_of_obj : int) -> None:
    """Recursive procedure.

    It attributes a front index to fitnesses in the 'seq_fitness_front', with the indices in the  'assign_indices', for the first 
    'count_of_obj' values of fitness by comparing them to fitness, with the indices in the 'comp_indices'.
            
    --------------------
    Args:
         'seq_fitness_front': a dictionary contains fitness and front index.
         'comp_indices': indices for comparing.
         'assign_indices': indices for assign front.
         'count_of_obj': the count of values of the fitness for assign front.
         
    --------------------
    Returns:
         None

    """

    if not comp_indices or not assign_indices:
        return
    elif len(comp_indices) == 1 or len(assign_indices) == 1:
        for i in assign_indices:
            for j in comp_indices:
                lv = seq_fitness_front[j]["fitness"][:count_of_obj]
                hv = seq_fitness_front[i]["fitness"][:count_of_obj] 
                if st.is_dominate(lv, hv) or lv == hv:
                    seq_fitness_front[i]["front"] = max(seq_fitness_front[i]["front"], seq_fitness_front[j]["front"] + 1)
    elif count_of_obj == 2:
        _sweep_b(seq_fitness_front, comp_indices, assign_indices)
    else:
        values_objective_from_comp_indices = { seq_fitness_front[i]["fitness"][count_of_obj - 1] for i in comp_indices}
        values_objective_from_assign_indices = { seq_fitness_front[j]["fitness"][count_of_obj - 1] for j in assign_indices}

        min_from_comp_indices, max_from_comp_indices = min(values_objective_from_comp_indices), max(values_objective_from_comp_indices)

        min_from_assign_indices, max_from_assign_indices = min(values_objective_from_assign_indices), max(values_objective_from_assign_indices)

        if max_from_comp_indices <= min_from_assign_indices:
            _nd_helper_b(seq_fitness_front, comp_indices, assign_indices, count_of_obj - 1)
        elif min_from_comp_indices <= max_from_assign_indices:
            median = st.find_low_median((values_objective_from_comp_indices | values_objective_from_assign_indices))

            less_median_indices_1, equal_median_indices_1, greater_median_indices_1 = _split_by(seq_fitness_front, comp_indices, median, count_of_obj)
            less_median_indices_2, equal_median_indices_2, greater_median_indices_2 = _split_by(seq_fitness_front, assign_indices, median, count_of_obj)

            _nd_helper_b(seq_fitness_front, less_median_indices_1, less_median_indices_2, count_of_obj)
            _nd_helper_b(seq_fitness_front, less_median_indices_1, equal_median_indices_2, count_of_obj - 1)
            _nd_helper_b(seq_fitness_front, equal_median_indices_1, equal_median_indices_2, count_of_obj - 1)
            _nd_helper_b(seq_fitness_front, less_median_indices_1 + equal_median_indices_1, equal_median_indices_2, count_of_obj - 1)
            _nd_helper_b(seq_fitness_front, equal_median_indices_1, equal_median_indices_2, count_of_obj)

def non_domin_sort(points : Sequence[Any], get_fitness : Callable[[Any], Iterable[Any]] = None ) -> Dict[int, Tuple[Any]]:
    """A non-dominated sorting of 'points'.
    
    If 'get_fitness' is 'None', then it is identity map. 'get_fitness = lambda x : x'.

    --------------------
    Args:
        'points': the points for non-dominated sorting.
        'get_fitness': a callable object, which mapping points to their fitness.

    --------------------
    Returns:
        A dictionary. It contains indices of fronts as keys and values are tuple consist of 'points' which have same index of front.

    """

    assert points, "The length of points must be > 0."

    if get_fitness is None:
        count_of_obj = len(points[0])  
    else:
        count_of_obj = len(get_fitness(points[0]))
        
    assert count_of_obj > 1, "The length of fitness must be > 1."

    # The dictionary contains the fitnesses as keys and their indices in the list 'fitnesses'.
    fitness_dict = defaultdict(list)

    if get_fitness is None:
        fitnesses_gen = map(tuple, points)
    else:
        fitnesses_gen = map(lambda x: tuple(get_fitness(x)), points)

    for (index, fitness) in enumerate(fitnesses_gen):
        fitness_dict[fitness].append(index)

    # The list 'unique_fitnesses' never changes, but its elements yes.
    # It sorted in the lexicographical order.
    unique_fitnesses = [{"fitness" : fitness, "front" : 0}  for fitness in sorted(fitness_dict.keys())]

    # Further, algorithm works only with the indices of list 'unique_fitnesses'.
    indices_uniq_fitnesses = list(range(len(unique_fitnesses)))
    _nd_helper_a(unique_fitnesses, indices_uniq_fitnesses, count_of_obj)

    # The dictionary contains indices of the fronts as keys and the tuple of 'points' as values.
    fronts = defaultdict(tuple)

    # Generate fronts.
    for ff in unique_fitnesses:
        fronts[ff["front"]] += tuple(points[index] for index in fitness_dict[ff["fitness"]])

    return fronts