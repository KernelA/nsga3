"""A base class for the multiobjective problems.

"""

from abc import ABC, abstractmethod
from typing import Iterable, Any

import numpy as np

__all__ = ["MOProblem"]


class MOProblem(ABC):
    """The base class for the all multiobjective problems.
    """

    @property
    @abstractmethod
    def lower_bounds(self):
        """Return the lower bounds of decision space.

        A decision space is a rectangle area.
        """
        pass

    @property
    @abstractmethod
    def upper_bounds(self):
        """Return the upper bounds of decision space.

        A decision space is a rectangle area.
        """
        pass

    @property
    @abstractmethod
    def amount_objs(self):
        """Return number of objectives.
        """
        pass

    @abstractmethod
    def eval(self, point: np.ndarray) -> Iterable[Any]:
        """Maps a decision 'point' into a objectives space.

        --------------------
        Args:
            'point': A decision.

        """
        pass

