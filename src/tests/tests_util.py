import unittest
import random
import math

from pynsga3 import utils


def _binomial(n, k):
    if k > n:
        return 0
    prod = 1
    for i in range(k + 1, n + 1):
        prod *= i
    return prod // math.factorial(n - k)


class TestStools(unittest.TestCase):

    def test_random_clip(self):
        value = 0.5
        low_b = -1
        upp_b = 1

        self.assertEqual(value, utils.tools.clip_random(value, low_b, upp_b))

    def test_gen_convex_hull(self):
        dim = tuple(range(1,6))
        count = tuple(range(2, 6))

        for d in dim:
            for c in count:
                coefficients = utils.tools.convhull.generate_coeff_convex_hull(d, c)
                self.assertEqual(_binomial(d + c - 2, c - 1), len(coefficients))
                for vec in coefficients:
                    self.assertAlmostEqual(1.0, sum(vec), places=10)
                    for coeff in vec:
                        self.assertGreaterEqual(coeff, 0)
                        self.assertLessEqual(coeff, 1)

        
if __name__ == '__main__':
    unittest.main()

