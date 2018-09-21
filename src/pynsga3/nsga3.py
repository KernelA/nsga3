"""The module contains implementation of algorithm NSGA-3.

"""
import random
import itertools
import sys
from typing import Tuple, Any, Union, Iterable, Sequence, Type

from .utils import convhull, tools
from . import bproblem

import numpy as np
import scipy
from scipy import linalg

import warnings

try:
    from nds import ndomsort
except ImportError:
    warnings.warn("You need to install nds package. You can download it from:\nhttps://github.com/KernelA/nds-py")
    raise

__all__ = ["NSGA3"]


_EPS = sys.float_info.epsilon * 100
_REL_TOL = sys.float_info.dig - 2 if sys.float_info.dig - 2 > 0 else sys.float_info.dig


class _RefPoint:
    """The class represents a reference point.
    """

    __slots__ = ["fitness", "fitness_on_hyperplane"]

    def __init__(self, fitness: np.ndarray):
        self.fitness = fitness
        self.fitness_on_hyperplane = fitness.copy()

    def map_on_hyperplane(self, ideal_point: np.ndarray, intercepts: np.ndarray) -> None:
        """Maps a reference point on a hyperplane. The hyperplane defined as x_1 + .... + x_n = 1,
        where n - dimension of objective space, x_i >= 0, i in {1,...,n}.
        """
        np.copyto(self.fitness_on_hyperplane, self.fitness)

        self.fitness_on_hyperplane -= ideal_point

        indices_with_close_values = np.isclose(ideal_point, intercepts, rtol=_REL_TOL, atol=_EPS)
        diff = (intercepts - ideal_point)[~indices_with_close_values]

        self.fitness_on_hyperplane[:, indices_with_close_values] /= _EPS
        self.fitness_on_hyperplane[:, ~indices_with_close_values] /= diff

    def __str__(self) -> str:
        return "Fitness: {0}\nNormalized fitness: {1}".format(self.fitness, self.fitness_on_hyperplane)


class NSGA3:
    """The algorithm described in:

    Deb, Kalyanmoy & Jain, Himanshu. (2014).
    An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
     Part I: Solving Problems With Box Constraints.
    Evolutionary Computation, IEEE Transactions on. 18. 577-601. 10.1109/TEVC.2013.2281535.

    """

    # Division of axis when need to create reference points each axis divided on n parts.
    # Keys are number of objectives.
    # Values are p parameter in the original algorithm.
    # For example, if two-objective problems chosen,
    # then p = 4 and we will get C_{2 + 4 - 1}^4 = 5 reference points.
    # If a value is a two number, then we use two-layered reference points.
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
        """Create NSGA-3 algorithm.

        --------------------
        Args:
            'crossover_op': A crossover operator. See class CrossoverOp.
            'mutation_operator': A mutation operator. See class MutationOp.
        """
        self._ref_points = []
        self.__points = None
        self.__fitnesses = None
        self.__children_points = None
        self.__children_fitness = None
        self.__all_pop_points = None
        self.__all_pop_fitnesses = None
        self.__all_pop_normalized_fitnesses = None
        self.__prev_params = {}

        self._ideal_point = None
        self._niche_counts = None

        self._crossover_op = crossover_op
        self._mut_op = mutation_operator

        self._is_ref_points_int = False
        self._is_ref_points_str = False

        self._is_size_pop_str = False
        
    def _generate_init_pop(self, size_pop: int, amount_obj: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        """Generate an initial population uniformly on a hyperrectangle  area.
        """
        dim_decision = len(lower_bounds)

        if self.__fitnesses is None:
            self.__fitnesses = np.zeros((size_pop, amount_obj))
        else:
            self.__fitnesses.resize((size_pop, amount_obj))

        self.__prev_params["size_pop"] = size_pop

        self.__points = (upper_bounds - lower_bounds) * np.random.rand(size_pop, dim_decision) + lower_bounds

    def _gen_ref_points(self, divisions: Tuple[int], amount_obj: int):
        """Generate the reference points on the hyperplane.

        The hyperplane defined as x_1 + .... + x_n = 1,
        where n - dimension of objective space, x_i >= 0, i in {1,...,n}.

        --------------------
        Args:
            'divisions':
            'amount_obj':

        """

        def compute_point_on_hyperplane(coord_base_vector, vec_coeff, amount_obj: int) -> np.ndarray:
            """

            --------------------
            Args:
                'coord_base_vector': A coordinate of base vector in the convex hull.
                                     For example, if 'coord_base_vector' = 0.25
                                     then convex hull based on vectors:
                                     (0.25, 0, ..., 0), (0, 0.25, ..., 0), ... ,(0, 0, ..., 0.25).
                                     The number of vectors and its dimension is equal to 'amount_obj'.
                'vec_coeff': A vector of coefficients of the convex hull.
                'amount_obj': Dimension of objective space.

            """
            point_on_hyperplane = np.zeros(amount_obj)

            for coord_ind in range(len(point_on_hyperplane)):
                point_on_hyperplane[coord_ind] = coord_base_vector * vec_coeff[coord_ind]
            return point_on_hyperplane

        prev_divisions = self.__prev_params.get("divisions", None)

        is_gen_ref_point = True

        if prev_divisions is not None and self._ref_points is not None:
            # The reference points are same as in the previous run.
            if prev_divisions == divisions and len(self._ref_points[0].fitness) == amount_obj:
                is_gen_ref_point = False

        if is_gen_ref_point:
            step = 1 / len(divisions)

            for i in range(len(divisions)):
                coord_base_vector = (i + 1) * step
                cvh_coeffs = convhull.generate_coeff_convex_hull(amount_obj, divisions[i] + 1)

                self._ref_points += [_RefPoint(compute_point_on_hyperplane(coord_base_vector, vec_coeff, amount_obj))
                                     for vec_coeff in cvh_coeffs]

            # Update the previous parameter.
            self.__prev_params["divisions"] = divisions

    def _gen_ref_points_from_sap(self, supplied_asp_points: Iterable[Sequence[Any]]):
        """Remember user-supplied reference points.

        --------------------
        Args:
            'supplied_asp_points': User-supplied reference points (aspiration points).
        """

        self._ref_points = [_RefPoint(np.array(sap, dtype=float)) for sap in supplied_asp_points]
        self.__prev_params['divisions'] = None

    def _check_params(self, num_pop: int, amount_objs: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                      size_pop: int, ref_points, supplied_asp_points):

        assert amount_objs > 1, "The length of 'objectives' must be > 1."
        assert num_pop > 0, "'num_pop' must be > 0."
        assert len(lower_bounds) > 0, "The length of 'lower_bounds' must be > 0."
        assert len(upper_bounds) > 0, "The length of 'upper_bounds' must be > 0."
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
            for i, sal in enumerate(supplied_asp_points):
                is_empty = False
                assert len(sal) == len(amount_objs), "The length of the supplied aspiration point at position {0}" \
                                                     " is not equals length 'objectives'.".format(i)
            assert not is_empty, "The supplied aspiration points are empty."
 
    def _find_pop_size(self, old_size_pop: int):
        """Such rule is not really needed. We keep population size as described in the algorithm.
        """
        new_size_pop = old_size_pop

        if self._is_size_pop_str:
            new_size_pop = len(self._ref_points) + 1
            while new_size_pop % 4 != 0:
                new_size_pop += 1

        return new_size_pop

    def _find_ideal_point(self, fitnesses: np.ndarray):
        fitnesses.min(axis=0, out=self._ideal_point)

    @staticmethod
    def _find_extreme_points(normalized_fitnesses: np.ndarray, amount_obj: int):
        weights = np.full(amount_obj, 1E-6)
        extreme_points = np.zeros((amount_obj, amount_obj))

        for num_obj in range(amount_obj):
            weights[num_obj - 1] = 1E-6
            weights[num_obj] = 1

            # Here 'normalized_fitness' is the translated fitness.
            # The ideal point has been subtracting from the fitness at the moment.
            min_normalized_fitness = min(normalized_fitnesses, key=lambda x: tools.asf(x, weights))
            extreme_points[num_obj] = min_normalized_fitness
      
        return extreme_points

    def _find_divisions(self, amount_obj, ref_points):
        if self._is_ref_points_str:
            if amount_obj in NSGA3.__divisions_axis:
                divisions = NSGA3.__divisions_axis[amount_obj]
            else:
                divisions = NSGA3.__divisions_axis[max(NSGA3.__divisions_axis.keys())]
        elif self._is_ref_points_int:
            divisions = (ref_points,)
        else:
            divisions = ref_points

        return divisions

    @staticmethod
    def _find_intercepts(fitnesses: np.ndarray, extreme_points: np.ndarray):
        is_linalg_error = False

        try:
            b = np.ones(len(extreme_points))

            solution = linalg.solve(extreme_points, b, overwrite_a=True, overwrite_b=True)

            for i in range(len(solution)):
                solution[i] = 1 / solution[i]

        except linalg.LinAlgError:
            is_linalg_error = True
        except linalg.LinAlgWarning:
            is_linalg_error = True

        if is_linalg_error:
            solution = np.zeros(len(extreme_points))

        # Find the maximum values for the all coordinates of fitness in the population.
        if is_linalg_error or np.any(solution < 0):
            fitnesses.max(axis=0, out=solution)

        return solution

    @staticmethod
    def _compute_distance(direction: np.ndarray, point: np.ndarray):
        """Compute distance between reference line and point.

        The reference line is line which joining the reference point with the origin.

        --------------------
        Args:
            'direction':
            'point':

        Returns:

        """
        dot_prod = scipy.dot(direction, point)
        squared_norm = scipy.power(direction, 2).sum()

        coeff = dot_prod / squared_norm

        return linalg.norm(point - direction * coeff)

    def _associate_and_niche_counting(self, normalized_fitnesses_exclude_last_front: np.ndarray
                                      , normalized_fitnesses_last_front: np.ndarray):
        """Assign to each fitness in population closest reference points.
        """

        len_pop = len(normalized_fitnesses_exclude_last_front) + len(normalized_fitnesses_last_front)

        self._niche_counts.fill(0)

        closest_ref_points_and_distances = {i: {"distance": 0, "indices_ref_points": set()} for i in range(len_pop)}

        distances = np.zeros(len(self._ref_points))

        for index_pop, ind in enumerate(itertools.chain(normalized_fitnesses_exclude_last_front, normalized_fitnesses_last_front)):
            for index_rp, ref_point in enumerate(self._ref_points):
                distances[index_rp] = NSGA3._compute_distance(ref_point.fitness_on_hyperplane, ind)

            min_dist = distances.min()
            closest_ref_points_and_distances[index_pop]["distance"] = min_dist
            indices = (distances == min_dist).nonzero()[0]

            for i in indices:
                if index_pop < len(normalized_fitnesses_exclude_last_front):
                    self._niche_counts[i] += 1
                closest_ref_points_and_distances[index_pop]["indices_ref_points"].add(i)

        return closest_ref_points_and_distances

    def _niching(self, amount_to_choose: int, closest_ref_points_and_distances: dict, pop_last_front: np.ndarray):
        """Choose members from 'pop_last_front' in amount 'amount_to_choose'.
        """
        k = 1
        indices_last_front = set(range(len(pop_last_front)))

        indices_to_add = []

        indices_ref_points = np.array([True] * len(self._ref_points), dtype=np.bool_)

        type_info = np.iinfo(self._niche_counts.dtype)

        # Each key of dictionary 'closest_ref_points_and_distances' is index of member of population.
        # Keys include indices of 'pop_last_front'.
        # So 0 index in 'pop_last_front' corresponds of diff_len + 0 key in 'closest_ref_points_and_distances'.
        diff_len = len(closest_ref_points_and_distances) - len(pop_last_front) 

        while k <= amount_to_choose:
            min_niche_count = self._niche_counts.min()

            indices_min = (self._niche_counts == min_niche_count).nonzero()[0]

            random_index = random.choice(indices_min)

            indices_pop_closest_to_ref_point = [diff_len + index for index in indices_last_front
                                                if random_index in
                                                closest_ref_points_and_distances[diff_len + index]["indices_ref_points"]]

            if indices_pop_closest_to_ref_point:
                if self._niche_counts[random_index] == 0:
                    index_min = min(indices_pop_closest_to_ref_point
                                    , key=lambda index: closest_ref_points_and_distances[index]["distance"])
                    # Return to zero-based index because we choose from 'pop_last_front' in the end.
                    index_for_add = index_min - diff_len
                    indices_to_add.append(index_for_add)
                else:
                    index_for_add = random.choice(indices_pop_closest_to_ref_point) - diff_len
                    indices_to_add.append(index_for_add)

                self._niche_counts[random_index] += 1
                # Remove a member from further choosing.
                indices_last_front.remove(index_for_add)
                k += 1
            else:
                # Delete a reference point.
                self._niche_counts[random_index] = type_info.max
        return indices_to_add

    def _cross_mutate_and_eval(self, problem: bproblem.MOProblem, params: dict):
        """Apply crossover and mutation operator. Get values of objectives in a new population.
        """
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

        for i, child in enumerate(new_children):
            for j, child_val in enumerate(child):
                self.__children_points[i, j] = tools.clip_random(child_val, lower_bounds[j], upper_bounds[j])

            is_mutaded = self._mut_op.mutate(self.__children_points[i], **params)

            if is_mutaded:
                indices_less_bounds = self.__children_points[i] < lower_bounds
                indices_greater_bounds = self.__children_points[i] > upper_bounds

                for index in (indices_less_bounds | indices_greater_bounds).nonzero()[0]:
                    self.__children_points[i, index] = tools.clip_random(self.__children_points[i, index]
                                                                               , lower_bounds[index], upper_bounds[index])

            for j, val in enumerate(problem.eval(self.__children_points[i])):
                self.__children_fitness[i, j] = val

    def _normalize(self, fitnesses: np.ndarray, normalized_fitnesses: np.ndarray, amount_obj: int):
        normalized_fitnesses -= self._ideal_point

        extreme_points = NSGA3._find_extreme_points(normalized_fitnesses, amount_obj)

        intercepts = NSGA3._find_intercepts(fitnesses, extreme_points)

        indices_with_close_values = np.isclose(self._ideal_point, intercepts, rtol=_REL_TOL, atol=_EPS)
        diff = (intercepts - self._ideal_point)[~indices_with_close_values]

        normalized_fitnesses[:, indices_with_close_values] /= _EPS
        normalized_fitnesses[:, ~indices_with_close_values] /= diff

        # Users-defined reference points. It must be mapped on hyperplane.
        if self.__prev_params["divisions"] is None:
            for ref_point in self._ref_points:
                ref_point.map_on_hyperplane(self._ideal_point, intercepts)

    def _init_vectors(self, amount_obj: int):
        if self._ideal_point is None:
            self._ideal_point = np.zeros(amount_obj)
        else:
            self._ideal_point.resize(amount_obj)

        if self._niche_counts is None:
            self._niche_counts = np.zeros(len(self._ref_points), dtype=np.uint32)
        else:
            self._niche_counts.resize(len(self._ref_points))

    def _init_all_pop(self):
        """Joining two ndarrays to one ndarray (decisions, fitnesses and normalized fitnesses).
        """
        if self.__all_pop_points is None:
            self.__all_pop_points = scipy.vstack((self.__points, self.__children_points))
        else:
            row, column = self.__points.shape
            self.__all_pop_points.resize(row + self.__children_points.shape[0], column)
            for index, point in enumerate(itertools.chain(self.__points, self.__children_points)):
                self.__all_pop_points[index] = point

        if self.__all_pop_fitnesses is None:
            self.__all_pop_fitnesses = scipy.vstack((self.__fitnesses, self.__children_fitness))
        else:
            row, column = self.__fitnesses.shape
            self.__all_pop_fitnesses.resize(row + self.__children_fitness.shape[0], column)
            for index, fitness in enumerate(itertools.chain(self.__fitnesses, self.__children_fitness)):
                self.__all_pop_fitnesses[index] = fitness

        if self.__all_pop_normalized_fitnesses is None:
            self.__all_pop_normalized_fitnesses = self.__all_pop_fitnesses.copy()
        else:
            self.__all_pop_normalized_fitnesses.resize(self.__all_pop_fitnesses.shape)
            np.copyto(self.__all_pop_normalized_fitnesses, self.__all_pop_fitnesses)

    def minimize(self, num_pop: int, problem: Type[bproblem.MOProblem], size_pop: Union[int, str] = 'auto'
                 , num_ref_points: Union[int, str, Tuple[int]] = 'auto'
                 , supplied_asp_points: Iterable[Sequence[Any]] = None) -> Tuple[np.ndarray]:
        """Run NSGA-3 algorithm.

        --------------------
        Args:
            num_pop: A total number of populations which replace one another (it is a number of iterations).
            problem: A multiobjective optimization problem. Its must be subclass of bproblem.MOProblem.
            size_pop: A number of decisions which generating on each iteration.
                      If 'size_pop' is equal to 'auto' then it is the smallest multiple of four
                      and greater than a total number of reference points.
            num_ref_points: The parameter used for calculation of a total number of reference points.
                            It is a number of division of axis on parts in objective space.
                            If it is int then is a number of division of axis on parts in objective space.
                            If it is 'auto'. A number of division of axis is stored in dictionary __divisions_axis.
                            If it is tuple. Two-layered or more reference points will be used.
                            If 'supplied_asp_points' is not None then the parameter is not used.
            supplied_asp_points: User-defined reference points.

        --------------------
        Returns:
            Tuple. The first item is decisions. The second item is fitnesses.
        """

        lower_bounds = np.array(problem.lower_bounds, dtype=float)
        upper_bounds = np.array(problem.upper_bounds, dtype=float)

        self._check_params(num_pop, problem.amount_objs, lower_bounds, upper_bounds
                           , size_pop, num_ref_points, supplied_asp_points)

        amount_obj = problem.amount_objs

        if supplied_asp_points is None:
            divisions = self._find_divisions(amount_obj, num_ref_points)
            self._gen_ref_points(divisions, amount_obj)
        else:
            self._gen_ref_points_from_sap(supplied_asp_points)

        self._init_vectors(amount_obj)

        new_size_pop = self._find_pop_size(size_pop)

        self._generate_init_pop(new_size_pop, amount_obj, lower_bounds, upper_bounds)

        params = {"iter": 1, "lower_bounds": lower_bounds, "upper_bounds": upper_bounds}

        for i, point in enumerate(self.__points):
            for j, val in enumerate(problem.eval(point)):
                self.__fitnesses[i, j] = val

        for num_iter in range(2, num_pop + 1):
            params["iter"] = num_iter

            if num_iter % 50 == 0:
                print("Iteration: ", num_iter)

            self._cross_mutate_and_eval(problem, params)
            self._init_all_pop()

            front_indices = np.array(ndomsort.non_domin_sort(self.__all_pop_fitnesses, only_front_indices=True))
            fronts, counts = np.unique(front_indices, return_counts=True)

            size_selected_pop = 0

            # First front index. 'fronts' is sorted array.
            last_front_index = fronts[0]
            amount_exclude_last_front = 0

            for front, count in zip(fronts, counts):
                if size_selected_pop >= new_size_pop:
                    break
                else:
                    amount_exclude_last_front = size_selected_pop
                    size_selected_pop += count
                    last_front_index = front

            amount_to_choose = new_size_pop - amount_exclude_last_front

            if size_selected_pop == new_size_pop:
                selected_indices = front_indices <= last_front_index
                np.copyto(self.__points, self.__all_pop_points[selected_indices])
                np.copyto(self.__fitnesses, self.__all_pop_fitnesses[selected_indices])
            else:
                indices_exclude_last_front = front_indices < last_front_index
                indices_last_front = front_indices == last_front_index

                # Ideal point must be in the first Pareto front.
                self.__all_pop_fitnesses[front_indices == fronts[0]].min(axis=0, out=self._ideal_point)

                self._normalize(self.__all_pop_fitnesses, self.__all_pop_normalized_fitnesses, amount_obj)

                closest_ref_points_and_distances =\
                    self._associate_and_niche_counting(self.__all_pop_normalized_fitnesses[indices_exclude_last_front]
                                                       , self.__all_pop_normalized_fitnesses[indices_last_front])

                indices_to_add = self._niching(amount_to_choose, closest_ref_points_and_distances
                                               , self.__all_pop_normalized_fitnesses[indices_last_front])

                for i, (point, fitness) in enumerate(
                        zip(itertools.chain(self.__all_pop_points[indices_exclude_last_front]
                                            , self.__all_pop_points[indices_last_front][indices_to_add])
                            , itertools.chain(self.__all_pop_fitnesses[indices_exclude_last_front]
                                              , self.__all_pop_fitnesses[indices_last_front][indices_to_add]))):

                    self.__points[i] = point
                    self.__fitnesses[i] = fitness

        return self.__points.copy(), self.__fitnesses.copy()
