"""The module contains a base class for all crossover operators.

"""

from typing import List, Iterable
from abc import ABC, abstractmethod

import numpy as np

__all__ = ["CrossoverOp"]


class CrossoverOp(ABC):
    """The base class for all crossover operators.
    """

    @abstractmethod
    def cross(self, parents: np.ndarray, **kwargs) -> List[Iterable[float]]:
        """The crossing of parents.

        --------------------
        Args:
            parents: The parents. A size of the array is a number of parents by a dimension of decision space.
            kwargs: Additional arguments.
                       {"lower_bounds" (np.array): the lower bounds of decision space,
                        "upper_bounds" (np.array): the upper bounds of decision space}

        --------------------
        Returns:
            The children.

        """
        pass
