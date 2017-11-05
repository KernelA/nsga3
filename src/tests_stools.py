import unittest
import stools as st
import random

class Teststools(unittest.TestCase):

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
        
if __name__ == '__main__':
    unitest.main()
