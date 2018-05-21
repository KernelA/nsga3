import random
import math
from typing import Sequence, List
from abc import ABC, abstractmethod

import numpy as np

__all__ = ["CrossoverOp", "SBXBound"]


class CrossoverOp(ABC):
    @abstractmethod
    def cross(self, parents: Sequence[np.ndarray], **kwargs):
        pass


class SBXBound(CrossoverOp):
    def __init__(self, crossover_prob: float, distr_index: float):
        assert crossover_prob >= 0 and crossover_prob <= 1, "'crossover_prob' must be in [0;1]."
        assert distr_index >= 0, "'distr_index' must be >= 0"
        self.distr_index = distr_index
        self.cross_prob = crossover_prob

    def cross(self, parents: Sequence[np.ndarray], **kwargs) -> List[np.ndarray]:
        lower_bounds = kwargs["lower_bounds"]
        upper_bounds = kwargs["upper_bounds"]

        children = []

        for i in range(len(parents) // 2):
            father = parents[random.randint(0, len(parents) - 1)]
            mother = parents[random.randint(0, len(parents) - 1)]

            child1 = np.array(father, dtype=float)
            child2 = np.array(mother, dtype=float)

            if random.uniform(0, 1) < self._cross_prob:
                for coord_num in range(len(father)):
                    if math.isclose(father[coord_num], mother[coord_num]):
                        continue

                    if random.uniform(0, 1) <= 0.5:
                        x1 = min(father[coord_num], mother[coord_num])
                        x2 = max(father[coord_num], mother[coord_num])
                        rand = random.uniform(0, 1)
                        power = self._distr_index + 1

                        beta = 1.0 + (2.0 * (x1 - lower_bounds[coord_num]) / (x2 - x1))
                        alpha = 2.0 - math.pow(beta, -power)

                        if rand <= 1.0 / alpha:
                            beta_q = math.pow(rand * alpha, 1.0 / power)
                        else:
                            beta_q = math.pow(1.0 / (2.0 - rand * alpha), 1.0 / power)

                        c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                        beta = 1.0 + (2.0 * (upper_bounds[coord_num] - x2) / (x2 - x1))
                        alpha = 2.0 - math.pow(beta, -power)
                        if rand <= 1.0 / alpha:
                            beta_q = math.pow(rand * alpha, 1.0 / power)
                        else:
                            beta_q = math.pow(1.0 / (2.0 - rand * alpha), 1.0 / power)
                        c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                        child1[coord_num] = c1
                        child2[coord_num] = c2

                        if random.uniform(0, 1) <= 0.5:
                            child1[coord_num] = c2
                            child2[coord_num] = c1

            children.append(child1)
            children.append(child2)

        return children
