"""The module contains  a bse class for all mutation operators.
"""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["MutationOp"]


class MutationOp(ABC):
    """The base class for all mutation operators.

    """

    @abstractmethod
    def mutate(self, individual: np.ndarray, **kwargs) -> bool:
        """Mutate an individual.

        --------------------
        Args:
            individual: An individual which must be mutated.
            kwargs:   Additional arguments.
                       {"lower_bounds" (np.array): the lower bounds of decision space,
                        "upper_bounds" (np.array): the upper bounds of decision space}

        --------------------
        Returns:
            True if individual was mutated, otherwise False
        """
        pass
