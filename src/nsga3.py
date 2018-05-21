"""The module contains implementation of algorithm NSGA-3.

"""
import random
import math
import itertools
import sys
from typing import Tuple, Any, Callable, Union, Iterable, Sequence, Type


import numpy as np
import scipy
from scipy import linalg
from nds import ndomsort

import convhull
import bproblem

__all__ = ["NSGA3"]


_EPS = sys.float_info.epsilon * 100
_REL_TOL = sys.float_info.dig - 2 if sys.float_info.dig - 2 > 0 else sys.float_info.dig


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


def _ASF(fitness: np.ndarray, weights: np.ndarray) -> float:
    return (fitness / weights).max()


class _Individual:

    __slots__ = ["point", "fitness", "normalized_fitness"]

    def __init__(self, point: np.ndarray, fitness: np.ndarray) -> None:
        self.point = point
        self.fitness = fitness
        self.normalized_fitness = fitness.copy()

    def eval(self, objectives: Tuple[Callable[[np.ndarray], float]]) -> None:
        for (i, obj) in enumerate(objectives):
            self.fitness[i] = obj(self.point)

    def translate_obj(self, ideal_point: np.ndarray) -> None:
        np.copyto(self.normalized_fitness, self.fitness)
        self.normalized_fitness -= ideal_point

    def normalize_obj(self, ideal_point: np.ndarray, intercepts: np.ndarray) -> None:
        for i in range(len(self.normalized_fitness)):
            if math.isclose(ideal_point[i], intercepts[i], rel_tol=_REL_TOL):
                self.normalized_fitness[i] /= _EPS
            else:
                self.normalized_fitness[i] /= (intercepts[i] - ideal_point[i])

    def copy_from_ind(self, ind):
        np.copyto(self.point, ind.point)
        np.copyto(self.fitness, ind.fitness)

    def __str__(self) -> str:
        return "Point: {0}\nFitness: {1}\nNormalized fitness: {2}".format(self.point, self.fitness, self.normalized_fitness)


class _RefPoint:

    __slots__ = ["fitness", "fitness_on_hyperplane"]

    def __init__(self, fitness: np.ndarray):
        self.fitness = fitness
        self.fitness_on_hyperplane = fitness.copy()

    def map_on_hyperplane(self, ideal_point: np.ndarray, intercepts: np.ndarray) -> None:
        np.copyto(self.fitness_on_hyperplane, self.fitness)

        self.fitness_on_hyperplane -= ideal_point

        indices_with_close_values = np.isclose(ideal_point, intercepts, rtol=_REL_TOL, atol=_EPS)
        diff = (intercepts - ideal_point)[~indices_with_close_values]

        self.fitness_on_hyperplane[:, indices_with_close_values] /= _EPS
        self.fitness_on_hyperplane[:, ~indices_with_close_values] /= diff

    def __str__(self) -> str:
        return "Fitness: {0}\nNormalized fitness: {1}".format(self.fitness, self.fitness_on_hyperplane)


class NSGA3:

    __divisions_axis = {
        2: (4,),
        3: (10,),
        4: (8,),
        5: (6,),
        6: (5,),
        7: (4,),
        8: (2, 3),
        9: (2, 3),
        10: (2, 3),
        11: (1, 2)
    }

    def __init__(self, crossover_op, mutation_operator):
        self._ref_points = []
        self.__points = None
        self.__fitnesses = None
        self.__normalized_fitnesses = None
        self.__children_points = None
        self.__children_fitness = None
        self.__children_normalized_fitness = None
        self.__points_copy = None
        self.__fitnesses_copy = None
        self.__prev_params = {}

        self._ideal_point = None
        self._niche_counts = None

        self._crossover_op = crossover_op
        self._mut_op = mutation_operator

        self._is_ref_points_int = False
        self._is_ref_points_str = False

        self._is_size_pop_str = False
        
    def _generate_init_pop(self, size_pop: int, dim_decision: int, amount_obj: int):
        if self.__points is None:
            self.__points = np.zeros((size_pop, dim_decision))
        else:
            self.__points.resize((size_pop, dim_decision))

        if self.__fitnesses is None:
            self.__fitnesses = np.zeros((size_pop, amount_obj))
        else:
            self.__fitnesses.resize((size_pop, amount_obj))

        self.__points_copy = self.__points.copy()
        self.__fitnesses_copy = self.__fitnesses.copy()

        if self.__normalized_fitnesses is None:
            self.__normalized_fitnesses = np.zeros((size_pop, amount_obj))
        else:
            self.__normalized_fitnesses.resize((size_pop, amount_obj))

        self.__prev_params["size_pop"] = size_pop

    def _generate_reference_points(self, divisions, amount_obj, supplied_asp_points):
        is_gen_ref_point = True

        if divisions is None:
            self._ref_points = [_RefPoint(np.array(sap, dtype=float)) for sap in supplied_asp_points]
            is_gen_ref_point = False
        elif 'divisions' in self.__prev_params:
            if self.__prev_params['divisions'] is not None:
                if self.__prev_params['divisions'] == divisions:
                    is_gen_ref_point = False

        if is_gen_ref_point:  
            step = 1 / len(divisions)
     
            def compute_point_on_hyperplane(coord_base_vector, vec_coeff, amount_obj):
                point_on_hyperplane = np.zeros(amount_obj)
     
                for index_array in range(len(point_on_hyperplane)):
                    point_on_hyperplane[index_array] = coord_base_vector * vec_coeff[index_array]
                return point_on_hyperplane
      
            for i in range(len(divisions)):
                coord_base_vector = (i + 1) * step
                vecs_coeffs = convhull.generate_coeff_convex_hull(amount_obj, divisions[i] + 1)
      
                self._ref_points += [_RefPoint(compute_point_on_hyperplane(coord_base_vector, vec_coeff, amount_obj)) 
                                      for vec_coeff in vecs_coeffs]

        self.__prev_params['divisions'] = divisions

    def _check_params(self, num_pop: int, amount_objs: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                      size_pop: int, ref_points, supplied_asp_points):
             
        assert amount_objs > 1, "The length of 'objectives' must be > 1."
        assert num_pop > 0, "'num_pop' must be > 0."
        assert len(lower_bounds) > 0 and len(upper_bounds), "The length of 'lower_bounds' or 'upper_bounds' must be > 0."
        assert len(lower_bounds) == len(upper_bounds), "The 'lower_bounds' and 'upper_bound' have unequal length."

        indices = upper_bounds > lower_bounds

        assert indices.any(), "The lower bounds must be less then the upper bounds." \
                              "The lower bound at position: {0} is greater or equal than upper bound.".format(indices.nonzero())

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
        # or 'ref_points' is tuple.
        elif self._is_ref_points_int or not (self._is_ref_points_str or self._is_ref_points_int):
            if supplied_asp_points is not None:
                raise ValueError("The parameter 'supplied_asp_points' must be 'None', when 'ref_points' is not 'auto'.")

        if self._is_ref_points_int:
            assert ref_points > 0, "'ref_points' must be > 0."
        # 'ref_points' is tuple.
        elif not (self._is_ref_points_str or self._is_ref_points_int):
            assert ref_points, "Length 'ref_points' must be > 0."

            for (i, ref) in enumerate(ref_points):
                assert ref > 0, "The division at position {0} in the 'ref_points' must be > 0.".format(i)

        if supplied_asp_points is not None:
            is_empty = True
            for (i, sal) in enumerate(supplied_asp_points):
                is_empty = False
                assert len(sal) == len(amount_objs), "The length of the supplied aspiration point at position {0}" \
                                                     " is not equals length 'objectives'.".format(i)
            assert not is_empty, "The supplied aspiration points are empty."
 
    def _find_pop_size(self, size_pop: int):
        amount_ref_points = len(self._ref_points)

        new_size_pop = size_pop

        if self._is_size_pop_str:
            new_size_pop = amount_ref_points
            while (new_size_pop + 1) % 4 != 0:
                new_size_pop += 1
            new_size_pop += 1

        return new_size_pop

    def _find_ideal_point(self, fitnesses: np.ndarray):
        fitnesses.min(axis=0, out=self._ideal_point)

    def _find_extreme_points(self, normalized_fitnesses: np.ndarray, amount_obj: int):
        weights = np.full(amount_obj, 1E-6)
        extreme_points = np.zeros((amount_obj, amount_obj))

        for num_obj in range(amount_obj):
            weights[num_obj - 1] = 1E-6
            weights[num_obj] = 1

            # Here 'normalized_fitness' is the translated fitness.
            # The ideal point has been subtracting from the fitness at the moment.
            min_normalized_fitness = min(normalized_fitnesses, key=lambda x: _ASF(x, weights))
            extreme_points[num_obj] = min_normalized_fitness
      
        return extreme_points

    def _find_divisions(self, amount_obj, ref_points, supplied_asp_points):
        if supplied_asp_points is not None:
            divisions = None
        elif self._is_ref_points_str:
            if amount_obj in NSGA3.__divisions_axis:
                divisions = NSGA3.__divisions_axis[amount_obj]
            else:
                divisions = NSGA3.__divisions_axis[max(NSGA3.__divisions_axis.keys())]
        elif self._is_ref_points_int:
            divisions = (ref_points,)
        else:
            divisions = ref_points

        return divisions
       
    def _find_intercepts(self, fitnesses: np.ndarray, extreme_points: np.ndarray):
        unique_rows = np.unique(extreme_points, axis=0)
        is_seq_has_duplicates = len(unique_rows) != len(extreme_points)

        if is_seq_has_duplicates:
            solution = np.zeros(len(extreme_points))
        else:
            b = np.ones(len(extreme_points))

            solution = linalg.solve(extreme_points, b, overwrite_a=True, overwrite_b=True)
         
            for i in range(len(solution)):
                solution[i] = 1 / solution[i]

        # Find the maximum values for the all coordinates of fitness in the population.
        if is_seq_has_duplicates or np.any(solution < 0):
            fitnesses.max(axis=0, out=solution)

        return solution

    def _compute_distance(self, direction: np.ndarray, point: np.ndarray):
        dot_prod = scipy.dot(direction, point)
        squared_norm = scipy.power(direction, 2).sum()

        coeff = dot_prod / squared_norm
        res = 0

        for i in range(len(direction)):
            res += (point[i] - direction[i] * coeff) ** 2

        return math.sqrt(res)

    def _associate_and_niche_counting(self, normalized_fitnesses_exclude_last_front: np.ndarray
                                      , normalized_fitnesses_last_front: np.ndarray):

        len_pop = len(normalized_fitnesses_exclude_last_front) + len(normalized_fitnesses_last_front)

        self._niche_counts.fill(0)

        closest_ref_points_and_distances = {i: {"distance": 0, "ref_points": []} for i in range(len_pop)}

        index_pop = 0

        distances = np.zeros(len(self._ref_points))

        for ind in normalized_fitnesses_exclude_last_front:
            for (index_rp, ref_point) in enumerate(self._ref_points):
                distances[index_rp] = self._compute_distance(ref_point.fitness_on_hyperplane, ind)

            min_dist = distances.min()
            closest_ref_points_and_distances[index_pop]["distance"] = min_dist
            indices = (distances == min_dist).nonzero()[0]
            for i in indices:
                self._niche_counts[i] += 1
                closest_ref_points_and_distances[index_pop]["ref_points"].append(self._ref_points[i])

            index_pop += 1

        for ind in normalized_fitnesses_last_front:
            for (index_rp, ref_point) in enumerate(self._ref_points):
                distances[index_rp] = self._compute_distance(ref_point.fitness_on_hyperplane, ind)

            min_dist = distances.min()
            closest_ref_points_and_distances[index_pop]["distance"] = min_dist
            closest_ref_points_and_distances[index_pop]["ref_points"].extend(self._ref_points[i]
                                                                             for i in range(len(distances))
                                                                             if distances[i] == min_dist)

            index_pop += 1

        return closest_ref_points_and_distances

    def _niching(self, amount_to_choose: int, closest_ref_points_and_distances, pop_last_front: np.ndarray):

        k = 1
        indices_last_front = set(range(len(pop_last_front)))

        indices_to_add = []

        type_info = np.iinfo(self._niche_counts.dtype)

        diff_len = len(closest_ref_points_and_distances) - len(pop_last_front) 

        while k <= amount_to_choose:
            min_niche_count = self._niche_counts.min()

            indices_min = (self._niche_counts == min_niche_count).nonzero()[0]

            random_index = random.choice(indices_min)

            indices_pop_closest_to_ref_point = [diff_len + index for index in indices_last_front
                                                if self._ref_points[random_index] in
                                                closest_ref_points_and_distances[diff_len + index]["ref_points"]]

            if indices_pop_closest_to_ref_point:
                if self._niche_counts[random_index] == 0:
                    index_min = min(indices_pop_closest_to_ref_point, key=lambda index: closest_ref_points_and_distances[index]["distance"])
                    index_for_del = index_min - diff_len 
                    indices_to_add.append(index_for_del)
                else:
                    index_for_del = random.choice(indices_pop_closest_to_ref_point) - diff_len 
                    indices_to_add.append(index_for_del)

                self._niche_counts[random_index] += 1
                indices_last_front.remove(index_for_del)
                k += 1
            else:
                # Delete a reference point.
                self._niche_counts[random_index] = type_info.max
        return indices_to_add

    def _cross_mutate_and_eval(self, problem, params):

        amount_obj = problem.amount_objs

        lower_bounds = params["lower_bounds"]
        upper_bounds = params["upper_bounds"]

        new_children = self._crossover_op.cross(self.__points, **params)

        if self.__children_points is None:
            self.__children_points = np.zeros((len(new_children), len(lower_bounds)))
        else:
            self.__children_points.resize((len(new_children), len(lower_bounds)))

        if self.__children_fitness is None:
            self.__children_fitness = np.zeros((len(new_children), amount_obj))
        else:
            self.__children_fitness.resize((len(new_children), amount_obj))

        if self.__children_normalized_fitness is None:
            self.__children_normalized_fitness = np.zeros((len(new_children), amount_obj))
        else:
            self.__children_normalized_fitness.resize((len(new_children), amount_obj))

        for (i, child) in enumerate(new_children):
            for (j, child_val) in enumerate(child):
                self.__children_points[i, j] = child_val

            indices_less_bounds = self.__children_points[i] < lower_bounds
            indices_greater_bounds = self.__children_points[i] > upper_bounds
            for index in (indices_less_bounds | indices_greater_bounds).nonzero()[0]:
                self.__children_points[i, index] = clip_random(self.__children_points[i, index], lower_bounds[index], upper_bounds[index])

            is_mutaded = self._mut_op.mutate(self.__children_points[i], **params)

            if is_mutaded is True:
                indices_less_bounds = self.__children_points[i] < lower_bounds
                indices_greater_bounds = self.__children_points[i] > upper_bounds
                for index in (indices_less_bounds | indices_greater_bounds).nonzero()[0]:
                    self.__children_points[i, index] = clip_random(self.__children_points[i, index], lower_bounds[index], upper_bounds[index])

            for (j, val) in enumerate(problem.eval(self.__children_points[i])):
                self.__children_fitness[i, j] = val

        np.copyto(self.__normalized_fitnesses, self.__children_fitness)

    def _normalize(self, fitnesses: np.ndarray, normalized_fitnesses: np.ndarray, amount_obj: int):
        normalized_fitnesses -= self._ideal_point

        extreme_points = self._find_extreme_points(normalized_fitnesses, amount_obj)

        intercepts = self._find_intercepts(fitnesses, extreme_points)

        indices_with_close_values = np.isclose(self._ideal_point, intercepts, rtol=_REL_TOL, atol=_EPS)
        diff = (intercepts - self._ideal_point)[~indices_with_close_values]

        normalized_fitnesses[:, indices_with_close_values] /= _EPS
        normalized_fitnesses[:, ~indices_with_close_values] /= diff

        if self.__prev_params["divisions"] is None:
            for ref_point in self._ref_points:
                ref_point.map_on_hyperplane(self._ideal_point, intercepts)

    def _init_vectors(self):

        amount_obj = len(self._ref_points[0].fitness)

        if self._ideal_point is None:
            self._ideal_point = np.zeros(amount_obj)
        else:
            self._ideal_point.resize(amount_obj)

        if self._niche_counts is None:
            self._niche_counts = np.zeros(len(self._ref_points), dtype=int)
        else:
            self._niche_counts.resize(len(self._ref_points))

    def minimize(self, num_pop: int, problem: Type[bproblem.MOProblem], size_pop: Union[int, str] = 'auto'
                 , ref_points: Union[int, str, Tuple[int]] = 'auto', supplied_asp_points: Iterable[Sequence[Any]] = None):

        lower_bounds = np.array(problem.lower_bounds, dtype=float)
        upper_bounds = np.array(problem.upper_bounds, dtype=float)

        self._check_params(num_pop, problem.amount_objs, lower_bounds, upper_bounds
                           , size_pop, ref_points, supplied_asp_points)

        amount_obj = problem.amount_objs

        divisions = self._find_divisions(amount_obj, ref_points, supplied_asp_points)

        self._generate_reference_points(divisions, amount_obj, supplied_asp_points)

        self._init_vectors()

        new_size_pop = self._find_pop_size(size_pop)

        self._generate_init_pop(new_size_pop, len(lower_bounds), amount_obj)

        params = {"iter": 1, "lower_bounds": lower_bounds, "upper_bounds": upper_bounds}

        for i, point in enumerate(self.__points):
            for j, val in enumerate(problem.eval(point)):
                self.__fitnesses[i, j] = val

        np.copyto(self.__normalized_fitnesses, self.__fitnesses)

        for num_iter in range(num_pop):
            params["iter"] = num_iter

            if num_iter % 10 == 0:
                print("Iteration: ", num_iter)

            self._cross_mutate_and_eval(problem, params)

            new_pop_points = scipy.vstack((self.__points, self.__children_points))
            new_pop_fitnesses = scipy.vstack((self.__fitnesses, self.__children_fitness))

            front_indices = np.array(ndomsort.non_domin_sort(new_pop_fitnesses, only_front_indices=True))
            fronts, counts = np.unique(front_indices, return_counts=True)

            fronts = dict(zip(fronts, counts))

            size_selected_pop = 0
            # First front index.
            last_front_index = front_indices.min()

            while size_selected_pop <= new_size_pop:
                size_selected_pop += fronts[last_front_index]
                last_front_index += 1

            last_front_index -= 1

            if size_selected_pop == new_size_pop:
                selected_indices = front_indices <= last_front_index
                np.copyto(self.__points_copy, new_pop_points[selected_indices])
                np.copyto(self.__fitnesses_copy, new_pop_fitnesses[selected_indices])
            else:
                indices_exclude_last_front = front_indices < last_front_index
                indices_last_front = front_indices == last_front_index

                amount_to_choose = new_size_pop - size_selected_pop

                new_pop_normalized_fitnesses = scipy.vstack((self.__normalized_fitnesses, self.__children_normalized_fitness))

                # Ideal point must be in the first Pareto front.
                self._find_ideal_point(new_pop_fitnesses[front_indices == front_indices.min()])

                self._normalize(new_pop_fitnesses, new_pop_normalized_fitnesses, amount_obj)

                closest_ref_points_and_distances =\
                    self._associate_and_niche_counting(new_pop_normalized_fitnesses[indices_exclude_last_front]
                                                       , new_pop_normalized_fitnesses[indices_last_front])

                indices_to_add = self._niching(amount_to_choose, closest_ref_points_and_distances
                                               , new_pop_normalized_fitnesses[indices_last_front])

                for (i, (point, fitness)) in enumerate(
                        zip(itertools.chain(new_pop_points[indices_exclude_last_front]
                                            , new_pop_points[indices_last_front][indices_to_add])
                            , itertools.chain(new_pop_fitnesses[indices_exclude_last_front]
                                              , new_pop_fitnesses[indices_last_front][indices_to_add]))):

                    self.__points_copy[i] = point
                    self.__fitnesses_copy[i] = fitness

            np.copyto(self.__points, self.__points_copy)
            np.copyto(self.__fitnesses, self.__fitnesses_copy)
            np.copyto(self.__normalized_fitnesses, self.__fitnesses)

        return self.__points.copy(), self.__fitnesses.copy()
