import math
from abc import abstractmethod
import scipy

from pynsga3 import bproblem

__all__ = ["DTLZ1"]


class _BaseDTLZ(bproblem.MOProblem):

    def __init__(self, amount_dec: int, amount_objs: int):
        assert amount_objs > 1
        self.__amount_objs = amount_objs
        self.__lower_bounds = (0,) * amount_dec
        self.__upper_bounds = (1,) * amount_dec

    @abstractmethod
    def eval(self, x):
        pass

    @property
    def amount_objs(self):
        return self.__amount_objs

    @property
    def lower_bounds(self):
        return self.__lower_bounds

    @property
    def upper_bounds(self):
        return self.__upper_bounds

    def __str__(self):
        return "{0}_num_var={1}".format(self.__class__.__name__, len(self.__lower_bounds))


class DTLZ1(_BaseDTLZ):

    __PI_20 = 20 * math.pi

    __K = 5

    def __init__(self, amount_objs: int):
        super().__init__(amount_objs + DTLZ1.__K - 1, amount_objs)
        self.__res = [0] * amount_objs

    def _g(self, x):
        temp = x - 0.5
        return 100 * (DTLZ1.__K + (temp ** 2).sum() - scipy.cos(DTLZ1.__PI_20 * temp).sum())

    def eval(self, x):
        g = self._g(x[self.amount_objs - 1:])

        num_obj = 0

        for i in range(self.amount_objs - 1, -1, -1):
            product = scipy.prod(x[:i])

            if num_obj != 0:
                product *= (1 - x[i])

            self.__res[num_obj] = 0.5 * product * (1 + g)
            num_obj += 1

        return self.__res
