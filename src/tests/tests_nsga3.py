import unittest
import os

import scipy
import pandas

from pynsga3 import nsga3
from pynsga3.operators import sbx, polymut
from . import dtlzproblems


class TestNSGA3(unittest.TestCase):
    def setUp(self):
        sbx_op = sbx.SBXBound(1, 30)
        pol_mut = polymut.PolynomialMutationBound(1 / 7, 20)
        self.opt = nsga3.NSGA3(sbx_op, pol_mut)
        self.path = os.path.join(".", "DTLZ1(3)")
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def test_minimize(self):
        dtlz1 = dtlzproblems.DTLZ1(3)
        for i in range(10):
            print("Run ", i)
            points, fitnesses = self.opt.minimize(400, dtlz1, ref_points=12)
            res = pandas.DataFrame(scipy.hstack((points, fitnesses)), columns=["x" + str(i) for i in range(1, points.shape[1] + 1)] +
                                   ["obj" + str(i) for i in range(1, fitnesses.shape[1] + 1)])
            res.to_csv(os.path.join(self.path, f"Solution_DTLZ1(3)_id_0_run_{i}.csv"), index=False)


if __name__ == "__main__":
    unittest.main()
