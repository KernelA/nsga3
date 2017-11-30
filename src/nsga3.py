"""The module contains implementation of algorithm NSGA-3.



"""
import random
import math
import itertools
from typing import Tuple, List, Any, TypeVar, Callable, Union, Iterable, Sequence

import numpy
import scipy
from scipy import linalg
import ndomsort
import stools as st

__all__ = ["NSGA3", "SBXBound", "PolynomialMutationBound"]

NumpyArray = TypeVar("NumpyArray")
Number = TypeVar("Number")


def _achive_scal_fun(fitness : Sequence[Number], weights  : Sequence[Number]) -> float:
    max_value = fitness[0] / weights[0]
    for (value, weight) in zip(fitness, weights):
        max_value = max(max_value, value / weight)
    return max_value


class SBXBound:
    def __init__(self, distr_index : float):
        self._distr_index = distr_index

    def cross(self, parent1 : Tuple[Number], parent2 : Tuple[Number], **kwargs) -> Tuple[Number]:

        child1 = numpy.array(parent1, dtype = float)
        child2 = numpy.array(parent2, dtype = float)

        lower_bounds = kwargs["lower_bounds"]
        upper_bounds = kwargs["upper_bounds"]

        for i in range(len(parent1)):

            if math.isclose(parent1[i], parent2[i]):
                continue

            sum_par = parent1[i] + parent2[i]
            abs_diff = abs(parent2[i] - parent1[i])

            limit_int_low = (sum_par - 2 * lower_bounds[i]) / abs_diff
            limit_int_upp = (2 * upper_bounds[i] - sum_par) / abs_diff

            uniform_var = random.uniform(0,1)
            
            if limit_int_low <= 1:
                probability_lower = 0.5 * math.pow(limit_int_low, self._distr_index + 1)
            else:
                probability_lower = 0.5 * (2 - 1 / math.pow(limit_int_low, self._distr_index + 1))

            if limit_int_upp <= 1:
                probability_upper = 0.5 * math.pow(limit_int_upp, self._distr_index + 1)
            else:
                probability_upper = 0.5 * (2 - 1 / math.pow(limit_int_upp, self._distr_index + 1))
             

            probability_factor1 = uniform_var * probability_lower
            probability_factor2 = uniform_var * probability_upper


            if  probability_factor1 <= 0.5:
                factor_upper = math.pow(2 * probability_factor1, 1.0 / self._distr_index)
            else:
                factor_upper = math.pow(0.5 * 1 /  (1 - probability_factor1), self._distr_index + 1)

            if probability_factor2 <= 0.5:
                factor_lower = math.pow(2 * probability_factor2, 1.0 / self._distr_index)
            else:
                factor_lower = math.pow(0.5 * 1 /  (1 - probability_factor2), self._distr_index + 1)

            child1[i] = 0.5 * (sum_par - factor_lower * abs_diff)
            child2[i] = 0.5 * (sum_par + factor_upper * abs_diff)

            child1[i] = min(max(child1[i], lower_bounds[i]), upper_bounds[i])
            child2[i] = min(max(child2[i], lower_bounds[i]), upper_bounds[i])

        return child1, child2

class PolynomialMutationBound:

    # 'exp_mut' type ?
    def __init__(self, prob_mut : float, exp_mut : float):
        self._prob_mut = prob_mut
        self._exp_mut = exp_mut

    def mutate(self, individual : Tuple[Number], **kwargs) -> NumpyArray:

        lower_bounds = kwargs["lower_bounds"]
        upper_bounds = kwargs["upper_bounds"]

        mut_ind = numpy.array(individual, dtype = float)

        for i in range(len(individual)):
            if random.uniform(0,1) < self._prob_mut:
                x = mut_ind[i]
                delta1 = (x - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i])
                delta2 = (upper_bounds[i] - x) / (upper_bounds[i] - lower_bounds[i])

                mut_pow = 1 / (self._exp_mut + 1)

                uniform_var = random.uniform(0,1)

                if uniform_var < 0.5:
                    var_x = 1 - delta1
                    val = 2 * uniform_var + (1 - 2 * uniform_var) * math.pow(var_x, (self._exp_mut + 1))
                    delta_q = math.pow(val, mut_pow) - 1
                else:
                    var_x = 1 - delta2
                    val = 2 * (1 - uniform_var) + 2 * (uniform_var - 0.5) * math.pow(var_x, (self._exp_mut + 1))
                    delta_q = 1 - math.pow(val, mut_pow)
 

                x += delta_q * (upper_bounds[i] - lower_bounds[i])
                x = min(max(x, lower_bounds[i]), upper_bounds[i])
                mut_ind[i] = x

        return mut_ind


class _Individual:

    __slots__ = ["point", "fitness", "normalized_fitness"]

    def __init__(self, point : NumpyArray, fitness : NumpyArray) -> None:
        self.point = point
        self.fitness = fitness

    def __setattr__(self, name, value):
        if name == "fitness":
           _Individual.__dict__[name].__set__(self, value)
           _Individual.__dict__["normalized_fitness"].__set__(self, value.copy())
        else:
           _Individual.__dict__[name].__set__(self, value)

    def eval(self, objectives : Tuple[Callable[[Tuple[Number]], Number]]) -> None:
        safe_point  = tuple(self.point)
        for (i, obj) in enumerate(objectives):
            self.fitness[i] = obj(safe_point)
            self.normalized_fitness[i] = self.fitness[i]

    def translate_obj(self, ideal_point : NumpyArray) -> None:
        self.normalized_fitness -= ideal_point

    def normalize_obj(self, ideal_point : NumpyArray, intercepts : NumpyArray) -> None:
        for i in range(len(self.normalized_fitness)):
            if math.isclose(ideal_point[i], intercepts[i], rel_tol = 1E-14):
               self.normalized_fitness[i] /= 1E-15
            else:
               self.normalized_fitness[i] /= (intercepts[i] - ideal_point[i])
            if self.normalized_fitness[i] < 0:
                raise AssertionError("< 0")

    def __str__(self) -> str:
        return "Point: {0}\nFitness: {1}\nNormalized fitness: {2}".format(self.point, self.fitness, self.normalized_fitness)

class _RefPoint:

    __slots__ = ["fitness", "fitness_on_hyperplane"]

    def __init__(self, fitness : NumpyArray):
        self.fitness = fitness

    def __setattr__(self, name, value):
        _RefPoint.__dict__[name].__set__(self, value)
        _RefPoint.__dict__["fitness_on_hyperplane"].__set__(self, value.copy())
    
    def map_on_hyperplane(self, ideal_point : NumpyArray, intercepts : NumpyArray) -> None:
        numpy.copyto(self.fitness_on_hyperplane, self.fitness)
        self.fitness_on_hyperplane -= ideal_point
        for i in range(len(self.fitness_on_hyperplane)):
            self.fitness_on_hyperplane[i] /= (intercepts[i] - ideal_point[i])

    def __str__(self) -> str:
        return "Fitness: {0}\nNormalized fitness: {1}".format(self.fitness, self.fitness_on_hyperplane)

class NSGA3:

    _divisions_axis = {
        2 : (4,),
        3 : (10,),
        4 : (8,),
        5 : (6,),
        6 : (5,),
        7 : (4,),
        8 : (2, 3),
        9 : (2, 3),
        10 : (2, 3),
        11 : (1, 2)
    }

    def __init__(self, oper_cross, oper_mut):
        self._ref_points = []
        self._prev_params = {}
        self._population = []

        self._ideal_point = None
        self._niche_counts = None

        self._crossover = oper_cross
        self._mut = oper_mut

        self._is_ref_points_int = False
        self._is_ref_points_str = False

        self._is_size_pop_str = False
        
    def _generate_init_pop(self, size_pop, bounds, amount_obj):

        dim_size = len(bounds)

        if 'size_pop' not in self._prev_params:
            self._population = [_Individual(numpy.zeros((dim_size,)), numpy.zeros((amount_obj,))) for i in range(size_pop)]
        else:
            old_size_pop = self._prev_params['size_pop']
            if  old_size_pop < size_pop:
                size_add = size_pop - old_size_pop
                self._population.extend([_Individual(numpy.zeros((dim_size,)), numpy.zeros((amount_obj,))) for i in range(size_add)])
            elif old_size_pop > size_pop:
                del self._population[size_pop:]

        for ind in self._population:
            if ind.point.size != dim_size:
                ind.point = numpy.zeros((dim_size,))
            if ind.fitness.size != amount_obj:
                ind.fitness = numpy.zeros((amount_obj,))

            for (index, (low_b, upp_b)) in zip(range(dim_size), bounds):
                ind.point[index] = random.uniform(low_b, upp_b)

        self._prev_params['size_pop'] = size_pop

    def _generate_reference_points(self, divisions, amount_obj, supplied_asp_points):

        is_gen_ref_point = True

        if divisions is None:
            self._ref_points = [_RefPoint(numpy.array(sal, dtype = float)) for sal in supplied_asp_points]
            is_gen_ref_point = False
        elif 'divisions' in self._prev_params:
            if self._prev_params['divisions'] is not None:
                if self._prev_params['divisions'] == divisions:
                    is_gen_ref_point = False

        if is_gen_ref_point:  
            step = 1 / len(divisions)
     
            def compute_point_on_hyperplane(coord_base_vector, vec_coeff, amount_obj):
                point_on_hyperplane = numpy.zeros((amount_obj,))
     
                for index_array in range(point_on_hyperplane.size):
                    point_on_hyperplane[index_array] = coord_base_vector * vec_coeff[index_array]
                return point_on_hyperplane
      
            for i in range(0, len(divisions)):
                coord_base_vector = (i + 1) * step
                vecs_coeffs = st.generate_coeff_convex_hull(amount_obj, 1 / divisions[i])
      
                self._ref_points += [_RefPoint(compute_point_on_hyperplane(coord_base_vector, vec_coeff, amount_obj)) 
                                      for vec_coeff in vecs_coeffs]

        self._prev_params['divisions'] = divisions

    def _check_params(self, num_pop, objectives, bounds, size_pop, ref_points, supplied_asp_points):
             
        assert len(objectives) > 1, "The length of 'objectives' must be > 1."
        assert num_pop > 0, "'num_pop' must be > 0."
        assert bounds, "'len(bounds)' must be > 0."

        for (i, (low_b, upp_b)) in zip(range(len(bounds)), bounds):
            assert low_b < upp_b, "The lower bounds must be less then the upper bounds." \
                "The lower bound at position {0} is greater or equal than upper bound.".format(i)

        if isinstance(size_pop, str):
            if size_pop != 'auto':
                raise ValueError("The parameter 'size_pop' is not equal 'auto'.")
            self._is_size_pop_str = True
        elif isinstance(size_pop, int):
            assert size_pop > 0, "'size_pop' must be > 0."
        else:
            raise TypeError("'size_pop' must be 'int' or 'str'.")

        if isinstance(ref_points, str):
            self._is_ref_points_str = True
        elif isinstance(ref_points, int):
            self._is_ref_points_int = True
        elif isinstance(ref_points, tuple):
            self._is_ref_points_int = self._is_ref_points_str = False
        else:
            raise TypeError("'ref_points' must be 'int', 'tuple' or 'str'.")


        if self._is_ref_points_str:
            if ref_points != 'auto':
                raise ValueError("The parameter 'ref_points' is not equal 'auto'.")
        elif self._is_ref_points_int or not (self._is_ref_points_str or self._is_ref_points_int): # or 'ref_points' is tuple.
            if supplied_asp_points is not None:
                raise ValueError("The parameter 'supplied_asp_points' must be 'None', when 'ref_points' is not 'auto'.")

        if self._is_ref_points_int:
            assert ref_points > 0, "'ref_points' must be > 0."
        elif not (self._is_ref_points_str or self._is_ref_points_int): # 'ref_points' is tuple.
            assert ref_points, "Length 'ref_points' must be > 0."

            for (i, ref) in enumerate(ref_points):
                assert ref > 0, "The division at position {0} in the 'ref_points' must be > 0.".format(i)

        if supplied_asp_points is not None:
            is_empty = True
            for (i, sal) in enumerate(supplied_asp_points):
                is_empty = False
                assert len(sal) == len(objectives), "The length of the supplied aspiration point at position {0} is not equals length 'objectives'.".format(i)
            assert not is_empty, "The supplied aspiration points are empty."
 
    def _find_pop_size(self, size_pop):

        amount_ref_points = len(self._ref_points)

        if not self._is_size_pop_str: # 'size_pop' is 'int'.
            return size_pop
        else:
            new_size_pop = amount_ref_points
            while (new_size_pop + 1) % 4 != 0:
                new_size_pop += 1
            return new_size_pop + 1

    def _find_ideal_point(self, population):

        iterator = iter(population)

        try:
           fisrt_ind = next(iterator)
           numpy.copyto(self._ideal_point, fisrt_ind.fitness)

           while True:
               ind = next(iterator)
               for i in range(len(self._ideal_point)):
                   if ind.fitness[i] < self._ideal_point[i]:
                       self._ideal_point[i] = ind.fitness[i]
        except StopIteration:
            pass

    def _find_extreme_points(self, population, amount_obj):

        weights = numpy.zeros((amount_obj,))

        extreme_points = [None for i in range(amount_obj)]

        for num_obj in range(amount_obj):
            weights.fill(1E-6)
            weights[num_obj] = 1
            
            iterator = iter(population)

            try:
                first_ind = next(iterator)

                # Here 'normalized_fitness' is the translated fitness.
                # The ideal point has not been subtracting from the fitness at the moment.
                min_asf = _achive_scal_fun(first_ind.normalized_fitness, weights)

                extreme_points[num_obj] = fisrt_ind.normalized_fitness

                while True:
                    ind = next(iterator)
                    cur_asf = _achive_scal_fun(ind.normalized_fitness, weights)
                    if cur_asf < min_asf:
                        min_asf = cur_asf
                        extreme_points[num_obj] = ind.normalized_fitness
            except StopIteration:
                pass
      
        return extreme_points

    def _find_divisions(self, amount_obj, ref_points, supplied_asp_points):
        if supplied_asp_points is not None:
            divisions = None
        elif self._is_ref_points_str:
            if amount_obj in NSGA3._divisions_axis:
                divisions = NSGA3._divisions_axis[amount_obj]
            else:
                divisions = NSGA3._divisions_axis[max(NSGA3._divisions_axis.keys())]
        elif self._is_ref_points_int:
            divisions = (ref_points,)
        else:
            divisions = ref_points

        return divisions
       
    def _find_intercepts(self, population, extreme_points):

        def _is_seq_numpy_array_has_duplicate(sequence : Sequence[NumpyArray]) -> bool:
    
            for i in range(len(sequence)):
                for j in range(i + 1, len(sequence)):
                    if numpy.array_equal(sequence[i], sequence[j]):
                            return True
            return False

        is_seq_has_duplicate = _is_seq_numpy_array_has_duplicate(extreme_points)

        if is_seq_has_duplicate:
            solution = numpy.zeros((len(extreme_points,)))
        else:
            a = numpy.array(extreme_points, dtype = float)
            b = numpy.ones(len(extreme_points,))

            solution = linalg.solve(a, b, overwrite_b = True)
         

            for i in range(len(solution)):
                assert abs(solution[i]) < 1E-13, "Solution is zero."
                solution[i] = 1 / solution[i]

        # Find the maximum values for the all coordinates of fitness in the population.
        if is_seq_has_duplicate or numpy.any(solution < 0):
            iterator = iter(population)
            try:
                first_ind = next(population)
                numpy.copyto(solution, first_ind.fitness)
                while True:
                    ind = next(iterator)
                    for i in range(len(ind.fitness)):
                        if solution[i] < ind.fitness[i]:
                            solution[i] = ind.fitness[i]
            except StopIteration:
                pass
   
        return solution

    def _compute_distance(self, direction, point):
        dot_prod = scipy.dot(direction, point)
        squared_norm = scipy.power(direction, 2).sum()

        coeff = dot_prod / squared_norm

        res = 0

        for i in range(len(direction)):
            res += (point[i] - direction[i] * coeff) * (point[i] - direction[i] * coeff)

        return math.sqrt(res)

    def _associate(self, population, len_pop):

        closest_ref_points_and_distances = { i : { "distance" : 0, "ref_points" : [] }   for i in range(len_pop) }

        index_pop = 0

        distances = numpy.zeros((len(self._ref_points),))

        for ind in population:
   
            distances[0] = self._compute_distance(self._ref_points[0].fitness_on_hyperplane, ind.normalized_fitness)
 
            for index_ref_point in range(1, len(self._ref_points)):
                distances[index_ref_point] = self._compute_distance(self._ref_points[index_ref_point].fitness_on_hyperplane, ind.normalized_fitness)

            min_dist = distances.min()
            closest_ref_points_and_distances[index_pop]["distance"] = min_dist
            closest_ref_points_and_distances[index_pop]["ref_points"].extend(self._ref_points[i] for i in range(len(distances)) if distances[i] == min_dist)

            index_pop += 1

        return closest_ref_points_and_distances

    def _niche_counting(self, closest_ref_points_and_distances, range_indices):

        self._niche_counts.fill(0)

        for index_ref_p in range(len(self._ref_points)):
            for index in range_indices:
                if self._ref_points[index_ref_p] in closest_ref_points_and_distances[index]["ref_points"]:
                    self._niche_counts[index_ref_p] += 1

    def _niching(self, amount_to_choose, closest_ref_points_and_distances, last_front_pareto):

        k = 1
        indices_last_front_paretor = set(range(len(last_front_pareto)))

        add_pop = []

        type_info = numpy.iinfo(self._niche_counts.dtype)

        while k <= amount_to_choose:
            min_niche_count = self._niche_counts.min()

            indices_min = tuple(index for index in range(len(self._niche_counts)) if min_niche_count == self._niche_counts[index])

            random_index = random.choice(indices_min)

            indices_pop_closest_to_ref_point = [len(closest_ref_points_and_distances) - len(last_front_pareto) + index for index in indices_last_front_paretor
                                                 if self._ref_points[random_index] in closest_ref_points_and_distances[len(closest_ref_points_and_distances) - len(last_front_pareto) + index]["ref_points"]]

            if indices_pop_closest_to_ref_point:
                index_for_del = 0
                if self._niche_counts[random_index] == 0:
                    index_min = indices_pop_closest_to_ref_point[0]
                    min_dist = closest_ref_points_and_distances[index_min]["distance"]
                    for index in indices_pop_closest_to_ref_point:
                        if closest_ref_points_and_distances[index]["distance"] < min_dist:
                            min_dist = closest_ref_points_and_distances[index]["distance"]
                            index_min = index
                    index_for_del = index_min + len(last_front_pareto) - len(closest_ref_points_and_distances)
                    add_pop.append(last_front_pareto[index_for_del])

                else:
                    index_for_del = random.choice(indices_pop_closest_to_ref_point) + len(last_front_pareto) - len(closest_ref_points_and_distances)
                    add_pop.append(last_front_pareto[index_for_del])

                self._niche_counts[random_index] += 1
                indices_last_front_paretor.remove(index_for_del)
                k += 1
            else:
                # Deleted a reference point.
                self._niche_counts[random_index] = type_info.max
        return add_pop

    def _cross_mutate_and_eval(self, parents, children, objectives, params):

        amount_obj = len(objectives)

        lower_bounds = params["lower_bounds"]
        upper_bounds = params["upper_bounds"]
   
        j = 0
        for i in range(len(parents) // 2):
            index_parent1 = random.randint(0, len(parents) - 1)
            index_parent2 = random.randint(0, len(parents) - 1)

            parent1 = tuple(parents[index_parent1].point)
            parent2 = tuple(parents[index_parent2].point)

            temp_children = self._crossover.cross(parent1, parent2, **params)

            assert temp_children, "The crossover operator must return at least one child."
            
            for child in temp_children:
                assert len(child) == len(parent1), "The length of the child must be same as the length of the parent."

            if j + len(temp_children) <= len(children):
                for (index_child, child) in enumerate(temp_children):
                    for (index_val, value) in zip(range(len(child)), child):
                        children[j + index_child].point[index_val] = value
            else:
                children.extend(_Individual(numpy.array(child, dtype = float), numpy.zeros((amount_obj,))) for child in temp_children)

            j += len(temp_children)

        if len(children) > j:
            del children[j:]

        for child in children:
            for (i, low_b, upp_b) in  zip(range(len(child.point)), lower_bounds, upper_bounds):
                child.point[i] = st.clip_random(child.point[i], low_b, upp_b)

            mut_child = self._mut.mutate(tuple(child.point), **params)

            assert len(mut_child) == len(child.point), "The length of the mutated child must be same as it was."

            for (i, low_b, upp_b) in zip(range(len(child.point)), lower_bounds, upper_bounds):
                child.point[i] = st.clip_random(mut_child[i], low_b, upp_b)

            child.eval(objectives)

    def _normalize(self, population, amount_to_choose, amount_obj):

        for ind in population:
            ind.translate_obj(self._ideal_point)

        extreme_points = self._find_extreme_points(population, amount_obj)

        intercepts = self._find_intercepts(population, extreme_points)

        for ind in population:
            ind.normalize_obj(self._ideal_point, intercepts)

        if self._prev_params["divisions"] is None:
            for ref_point in self._ref_points:
                ref_point.map_on_hyperplane(self._ideal_point, intercepts)

    def _init_vectors(self):

        amount_obj = len(self._ref_points[0].fitness)

        if self._ideal_point is None:
            self._ideal_point = numpy.zeros((amount_obj,))
        else:
            if len(self._ideal_point) != amount_obj:
                self._ideal_point = numpy.zeros((amount_obj,))

        if self._niche_counts is None:
            self._niche_counts = numpy.zeros((len(self._ref_points),), dtype = int)
        else:
            if len(self._niche_counts) != len(self._ref_points):
                self._niche_counts = numpy.zeros((len(self._ref_points),), dtype = int)

    def minimize(self, num_pop : int, bounds : Sequence[Sequence[Number]], objectives : Sequence[Callable[[Sequence[Number]], Number]]
                 ,size_pop : Union[int, str] = 'auto' , ref_points : Union[int, str, Tuple[Number]] = 'auto', supplied_asp_points : Iterable[Sequence[Number]] = None):

        self._check_params(num_pop, objectives, bounds, size_pop, ref_points, supplied_asp_points)

        amount_obj = len(objectives)

        divisions = self._find_divisions(amount_obj, ref_points, supplied_asp_points)

        self._generate_reference_points(divisions, amount_obj, supplied_asp_points)

        self._init_vectors()

        new_size_pop = self._find_pop_size(size_pop)

        self._generate_init_pop(new_size_pop, bounds, amount_obj)

        lower_bounds = tuple(lower_b for (lower_b, upper_b) in bounds)
        upper_bounds = tuple(upper_b for (lower_b, upper_b) in bounds)

        children = []

        params = {"iter" : 1, "lower_bounds" : lower_bounds, "upper_bounds" : upper_bounds }

        for ind in self._population:
            ind.eval(objectives)

        for num_iter in range(num_pop):

            params["iter"] = num_iter
            if num_iter % 10 == 0:
                print("Iteration: ", num_iter)

            self._cross_mutate_and_eval(self._population, children, objectives, params)

            fronts = ndomsort.non_domin_sort(self._population + children, lambda x : x.fitness)

            ndomsort_size_pop = 0

            last_index_pareto_front = 0

            #deb
            a = set(id(i) for i in fronts.values())
            assert a != len(self._population), "N dom sort error"

            while ndomsort_size_pop <= new_size_pop:
                ndomsort_size_pop += len(fronts[last_index_pareto_front])
                last_index_pareto_front += 1

            last_index_pareto_front -= 1

            if ndomsort_size_pop == new_size_pop:
                index_pop = 0
                for i in range(last_index_pareto_front + 1):
                    for ind in fronts[i]:
                        numpy.copyto(self._population[index_pop].point, ind.point)
                        index_pop += 1
            else:             
                pop_exclude_last_front = tuple()

                for i in range(last_index_pareto_front):
                    pop_exclude_last_front += fronts[i]

                last_pareto_front = fronts[last_index_pareto_front]

                amount_to_choose = len(self._population) - len(pop_exclude_last_front)

                # Ideal point must be in the first pareto front.
                self._find_ideal_point(fronts[0])

                self._normalize(itertools.chain(pop_exclude_last_front, last_pareto_front), amount_to_choose, amount_obj)

                closest_ref_points_and_distances = self._associate(itertools.chain(pop_exclude_last_front, last_pareto_front), len(last_pareto_front) + len(pop_exclude_last_front))

                self._niche_counting(closest_ref_points_and_distances, range(len(pop_exclude_last_front)))

                pop_to_include = self._niching(amount_to_choose, closest_ref_points_and_distances, last_pareto_front)

                for (i, ind) in enumerate(itertools.chain(pop_exclude_last_front, pop_to_include)):
                   numpy.copyto(self._population[i].point, ind.point)

        return tuple((tuple(ind.point), tuple(ind.fitness)) for ind in self._population)        