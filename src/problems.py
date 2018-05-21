import math
from abc import ABC, abstractmethod

import bproblem

__all__ = ["ZDT1", "ZDT2", "ZDT3"]


class BaseZDT(bproblem.MOProblem):
    def __init__(self, amount_variables):
        assert amount_variables > 1
        self.__lower_bounds = (0,) * amount_variables
        self.__upper_bounds = (1,) * amount_variables

    def _g(self, x):
        return 1 + 9 / (len(self.__lower_bounds) - 1) * sum(x[1:])

    @abstractmethod
    def f1(self, x):
        pass

    @abstractmethod
    def f2(self, x):
        pass

    def eval(self, x):
        return self.f1(x), self.f2(x)

    @property
    def amount_objs(self):
        return 2

    @property
    def lower_bounds(self):
        return self.__lower_bounds

    @property
    def upper_bounds(self):
        return self.__upper_bounds

    def __str__(self):
        return "{0}_num_var={1}".format(self.__class__.__name__, len(self.constraints))


class ZDT1(BaseZDT):

    def __init__(self, amount_var):
        super().__init__(amount_var)

    def f1(self, x):
        return x[0]

    def f2(self, x):
        g = self._g(x)
        return g * (1 - math.sqrt(x[0] / g))


class ZDT2(BaseZDT):
    def __init__(self, amount_var):
        super().__init__(amount_var)

    def f1(self, x):
        return x[0]

    def f2(self, x):
        g = self._g(x)
        return g * (1 - (x[0] / g) ** 2)


class ZDT3(BaseZDT):
    def __init__(self, amount_var):
        super().__init__(amount_var)

    def f1(self, x):
        return x[0]

    def f2(self, x):
        g = self._g(x)
        return g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))

