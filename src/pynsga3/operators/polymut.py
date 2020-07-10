"""The module contains bounded polynomial mutation operator.

"""
import random
import math

from . import bmut

import numpy as np

__all__ = ["PolynomialMutationBound"]


class PolynomialMutationBound(bmut.MutationOp):
    """The bounded polynomial mutation operator.

    The operator described:
        Deb, Kalyanmoy & Goyal, Mayank. (1999).
        A Combined Genetic Adaptive Search (GeneAS) for Engineering Design.
        Computer Science and Informatics. 26.

    The algorithm of the bounded operator was taken from:
        http://www.iitk.ac.in/kangal/codes.shtml (Multi-objective NSGA-II code in C)
    """

    def __init__(self, prob_mut: float, dist_index: float):
        """Create bounded polynomial mutation operator.

        --------------------
        Args:
            prob_mut: The probability of mutation of real value.
            dist_index: The distribution index.
        """
        assert 0 <= prob_mut <= 1, "'prob_mut' must be in [0; ]."
        assert dist_index >= 0, "'dist_index' must be >= 0."
        self._prob_mut = prob_mut
        self._dist_index = dist_index

    def mutate(self, individual: np.ndarray, **kwargs) -> bool:
        """Mutate 'individual'.

        --------------------
        Args:
            individual:
            kwargs: Additional arguments.
                       {"lower_bounds" (np.array): the lower bounds of decision space,
                        "upper_bounds" (np.array): the upper bounds of decision space}

        --------------------
        Returns:
            True if 'individual' was mutated otherwise False.

        """
        lower_bounds = kwargs["lower_bounds"]
        upper_bounds = kwargs["upper_bounds"]
        is_mutated = False

        for i in range(len(individual)):
            if random.uniform(0, 1) < self._prob_mut:
                is_mutated = True
                x = individual[i]
                delta1 = (x - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i])
                delta2 = (upper_bounds[i] - x) / (upper_bounds[i] - lower_bounds[i])

                mut_pow = self._dist_index + 1

                uniform_var = random.uniform(0, 1)

                if uniform_var <= 0.5:
                    var_x = 1 - delta1
                    val = 2 * uniform_var + (1 - 2 * uniform_var) * math.pow(var_x, mut_pow)
                    delta_q = math.pow(val, 1 / mut_pow) - 1
                else:
                    var_x = 1 - delta2
                    val = 2 * (1 - uniform_var) + 2 * (uniform_var - 0.5) * math.pow(var_x, mut_pow)
                    delta_q = 1 - math.pow(val, 1 / mut_pow)

                individual[i] += delta_q * (upper_bounds[i] - lower_bounds[i])
        return is_mutated
