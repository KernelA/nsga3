"""Module is implementation of none-dominated sorting.

Original algorithm described in paper:

Buzdalov M., Shalyto A.
A Provably Asymptotically Fast Version of the Generalized Jensen Algorithm for Non-dominated Sorting 
// Parallel Problem Solving from Nature XIII.
- 2015. - P. 528-537. - (Lecture Notes on Computer Science ; 8672)

"""

import functools
import itertools
import stools as st
from typing import List, Iterable, TypeVar, Tuple, Sequence, Callable, Dict

__all__ = ["non_domin_sort"]

T = TypeVar('T')
Fitness = TypeVar('Fitness')

class FitnessFront:
    """Class to store fitness and front index.
    
     --------------------
     Attributes:
        'fitness': A fitness. # type: Tuple[T]
        'front': A front index. # type: int

    """
    __slots__ = ["fitness", "front"]

    def __init__(self, fitness : Tuple[T]):
        """Set fitness. 

        A front index by default is 0.

        --------------------
        Args:
            'fitness': A fitness.

        """
        self.fitness = fitness
        self.front = 0

        def __eq__(self, other):
            return self.fitness == other.fitness and self.front == other.front

        def __ne__(self, other):
            return self.fitness != other.fitness or self.front != other.front


def is_seq_has_one_uniq_value(iterable : Iterable[T]) -> bool:
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
    first_value = next(iterator, False)
    is_has_uniq_value = False

    if not first_value:
        return is_has_uniq_value
    else:
        is_has_uniq_value = True

    while True:
         value = next(iterator, False)

         if not value:
             break
         else:
             if value != first_value:
                 is_has_uniq_value = False
                 break
    return is_has_uniq_value
 
def split_by(seq_fitness_front : dict, indices : List[int], split_value : T, count_of_obj : int) -> Tuple[List[int], List[int], List[int]]:
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

    indices_less_split_value = [index for index in indices if seq_fitness_front[index].fitness[count_of_obj - 1] < split_value]

    indices_greater_split_value = [index for index in indices if seq_fitness_front[index].fitness[count_of_obj - 1] > split_value ]

    indices_equal_split_value = [index for index in indices  if index not in indices_greater_split_value and index not in indices_less_split_value]

    return indices_less_split_value, indices_equal_split_value, indices_greater_split_value

def sweep_a(seq_fitness_front : dict, indices : List[int]) -> None:
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
    init_ind = set(indices[0],)

    for i  in indices[1:]:
        u_ind = [index for  index in init_ind if  seq_fitness_front[index][1] <= seq_fitness_front[i][1]]
        if len(u_ind) != 0:
            max_front = max(u_ind, key = lambda index : seq_fitness_front[index].front)
            seq_fitness_front[i].front = max(fitness_front[i].front, max_front + 1)

        init_ind -= { index for index in init_ind if seq_fitness_front[index].front == seq_fitness_front[i].front }
        init_ind.add(i)

def sweep_b(seq_fitness_front : dict, comp_indices : List[int], assign_indices : List[int]) -> None:
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
            fitness_right = seq_fitness_front[j].fitness[:2]
            
        while p < len(comp_indices):
            i = comp_indices[p]
            fitness_left = seq_fitness_front[i].fitness[:2]
            if fitness_left <= fitness_right:
                r = { index for index in init_ind if seq_fitness_front[index].front == seq_fitness_front[i].front  and seq_fitness_front[index].fitness[1] <  seq_fitness_front[i].fitness[1] }

                if len(r) == 0:
                    init_ind -= { index for index in init_ind if seq_fitness_front[index].front == seq_fitness_front[i].front }
                    init_ind.add(i)
                p += 1
            else:
                break
        u = {index for index in init_ind if seq_fitness_front[index].fitness[1] <= seq_fitness_front[j].fitness[1] }

        if len(u) != 0:
            max_front = max((seq_fitness_front[index].front for index in u))
            seq_fitness_front[j].front = max(seq_fitness_front[j].front, max_front + 1)

def nd_helper_a(seq_fitness_front : dict, indices : List[int] , count_of_obj : int) -> None:
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
        fitness1, fitness2 = seq_fitness_front[index_l].fitness[:count_of_obj], seq_fitness_front[index_r].fitness[:count_of_obj]

        if st.is_dominate(fitness1, fitness2):
            seq_fitness_front[index_r].front = max(seq_fitness_front[index_r].front, seq_fitness_front[index_l].front + 1)
    elif count_of_obj == 2:
        sweep_a(seq_fitness_front, indices)
    elif is_seq_has_one_uniq_value(fit_fr.fitness[count_of_obj - 1]  for fit_fr in seq_fitness_front):
        nd_helper_a(seq_fitness_front, indices, count_of_obj - 1)
    else:
        median = st.find_low_median((seq_fitness_front[index].fitness[count_of_obj - 1] for index in indices))
       
        less_median, equal_median, greater_median = split_by(seq_fitness_front, indices, median, count_of_obj)

        nd_helper_a(seq_fitness_front, less_median, count_of_obj)
        nd_helper_b(seq_fitness_front, less_median, equal_median, count_of_obj - 1)
        nd_helper_a(seq_fitness_front, equal_median, count_of_obj - 1)
        nd_helper_b(seq_fitness_front, less_median + equal_median, greater_median, count_of_obj - 1)
        nd_helper_a(seq_fitness_front, greater_median, count_of_obj)

def nd_helper_b(seq_fitness_front : dict,  comp_indices : List[int], assign_indices : List[int], count_of_obj : int) -> None:
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
    if len(comp_indices) == 0 or len(assign_indices) == 0:
        return
    elif len(comp_indices) == 1 or len(assign_indices) == 1:
        for i in assign_indices:
            for j in comp_indices:
                lv = seq_fitness_front[j].fitness[:count_of_obj]
                hv = seq_fitness_front[i].fitness[:count_of_obj] 
                if st.is_dominate(lv, hv) or lv == hv:
                    seq_fitness_front[i].front = max(seq_fitness_front[i].front, seq_fitness_front[j].front + 1)
    elif count_of_obj == 2:
        sweep_b(seq_fitness_front, comp_indices, assign_indices)
    else:
        values_objective_from_comp_indices = { seq_fitness_front[i].fitness[count_of_obj - 1] for i in comp_indices}
        values_objective_from_assign_indices = { seq_fitness_front[j].fitness[count_of_obj - 1] for j in assign_indices}

        min_from_comp_indices, max_from_comp_indices = min(values_objective_from_comp_indices), max(values_objective_from_comp_indices)

        min_from_assign_indices, max_from_assign_indices = min(values_objective_from_assign_indices), max(values_objective_from_assign_indices)

        if max_from_comp_indices <= min_from_assign_indices:
            nd_helper_b(seq_fitness_front, comp_indices, assign_indices, count_of_obj - 1)
        elif min_from_comp_indices <= max_from_assign_indices:
            median = st.find_low_median((values_objective_from_comp_indices | values_objective_from_assign_indices))

            less_median_indices_1, equal_median_indices_1, greater_median_indices_1 = split_by(seq_fitness_front, comp_indices, median, count_of_obj)
            less_median_indices_2, equal_median_indices_2, greater_median_indices_2 = split_by(seq_fitness_front, assign_indices, median, count_of_obj)

            nd_helper_b(seq_fitness_front, less_median_indices_1, less_median_indices_2, count_of_obj)
            nd_helper_b(seq_fitness_front, less_median_indices_1, equal_median_indices_2, count_of_obj - 1)
            nd_helper_b(seq_fitness_front, equal_median_indices_1, equal_median_indices_2, count_of_obj - 1)
            nd_helper_b(seq_fitness_front, less_median_indices_1 + equal_median_indices_1, equal_median_indices_2, count_of_obj - 1)
            nd_helper_b(seq_fitness_front, equal_median_indices_1, equal_median_indices_2, count_of_obj)

def non_domin_sort(points : Iterable[T], get_fitness : Callable[[T], Iterable[Fitness]] = None ) -> Dict[int, Tuple[T]]:
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

    if get_fitness is None:
        get_fitness = lambda x : x


    fitnesses = [tuple(get_fitness(point)) for point in points]

    count_of_obj = len(fitnesses[0])

    # The dictionary contains the fitnesses as keys and their indices in the list 'fitnesses'.
    fitness_dict = {}

    for i in range(len(fitnesses)):
        if fitnesses[i] in fitness_dict:
            fitness_dict[fitnesses[i]].append(i)
        else:
            fitness_dict[fitnesses[i]] = [i]

    del fitnesses

    # The tuple 'unique_fitnesses' never changes, but his elements yes.
    # The tuple sorted in the lexicographical order.
    unique_fitnesses = tuple((FitnessFront(fitness) for fitness in sorted(fitness_dict.keys())))

 
    if count_of_obj == 1:
        first_new_ff = unique_fitnesses[0]
        new_index_front = first_new_ff.front

        # Assign index of front. 
        for i in range(1, len(unique_fitnesses)):
            if unique_fitnesses[i] != first_new_ff:
                new_index_front += 1
                first_new_ff = unique_fitnesses[i]
                first_new_ff.front = new_index_front
            else:
                unique_fitnesses[i].front = new_index_front

    else:
        # Further, algorithm works only with the indices of tuple 'unique_fitnesses'.
        indices_uniq_fitnesses = list(range(len(unique_fitnesses)))
        nd_helper_a(unique_fitnesses, indices_uniq_fitnesses, count_of_obj)

    fronts = {}

    # Generate fronts.
    for ff in unique_fitnesses:
        elemnts = tuple((points[index] for index in fitness_dict[ff.fitness]))

        if ff.front not in fronts:
            fronts[ff.front] = elemnts
        else:
            fronts[ff.front] += elemnts
        
    return fronts