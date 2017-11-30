import unittest
import random
import math

import stools as st

def _binomial(n, k):
    if k > n:
        return 0

    prod = 1
    for i in range(k + 1, n + 1):
        prod *= i
    return prod // math.factorial(n - k)

class TestStools(unittest.TestCase):

    def setUp(self):

        self.dom_pairs_seq = (
            ((0,0), (1,1)),
            ((1,0), (1,1)),
            ((1,1), (1,1)),
            ((2,1), (1,1)),
            ((2,2), (1,1)),
            ((0,1,0), (1,0,0)),
            )
        self.dom_asnwers = (True, True, False, False, False, False)


    def test_is_dominate(self):
            i = 0
            for (left, right) in self.dom_pairs_seq:
                self.assertEqual(st.is_dominate(left, right), self.dom_asnwers[i])
                i += 1
    def test_find_median(self):
        for size in range(1, 31):
            seq = [random.uniform(-100,100) for i in range(size)]
            sort_seq = sorted(seq)
            self.assertEqual(st.find_low_median(seq), sort_seq[(size-1) // 2])
    def test_random_clip(self):
        value = 0.5
        low_b = -1
        upp_b = 1

        self.assertEqual(value, st.clip_random(value, low_b, upp_b))

    def test_gen_convex_hull(self):
        dim = tuple(range(1,6))
        count = tuple(range(2, 6))

        for d in dim:
            for c in count:
                coefficients = st.generate_coeff_convex_hull(d, c)
                self.assertEqual(binomial(d + c - 2, c - 1), len(coefficients))
                for vec in coefficients:
                    self.assertAlmostEqual(1.0, sum(vec), places = 10)
                    for coeff in vec:
                        self.assertGreaterEqual(coeff, 0)
                        self.assertLessEqual(coeff, 1)

        
if __name__ == '__main__':
    unitest.main()

