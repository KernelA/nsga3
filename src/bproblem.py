"""A base class for the multiobjective problems.

"""

from abc import ABC, abstractmethod

__all__ = ["MOProblem"]

class MOProblem(ABC):

    @property
    @abstractmethod
    def lower_bounds(self):
        pass

    @property
    @abstractmethod
    def upper_bounds(self):
        pass

    @property
    @abstractmethod
    def amount_objs(self):
        pass

    @abstractmethod
    def eval(self, point):
        pass

