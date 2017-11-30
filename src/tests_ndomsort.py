import unittest
import random

import ndomsort as nds
import stools as st

class TestNdomsort(unittest.TestCase):


    def test_non_domin_sort_many_fronts(self):
        seq = [(i,) * 4 for i in range(20)]

        res = nds.non_domin_sort(seq)

        self.assertEqual(len(res), 20)
        self.assertSetEqual(set(range(20)), set(res.keys()))

        for front in res:
            for res_seq in res[front]:
                self.assertTupleEqual(res_seq, seq[front])

    def test_non_domin_sort_one_elem(self):
        seq = [(2,3,4)]

        res = nds.non_domin_sort(seq)

        self.assertEqual(len(res), 1)

        self.assertSetEqual(set(range(1)), set(res.keys()))

        self.assertTupleEqual(seq[0], res[0][0])

    def test_non_domin_sort_one_front(self):

        seq = ((1,1), (1,1), (1,1))

        res = nds.non_domin_sort(seq)

        self.assertEqual(len(res), 1)

        self.assertSetEqual(set(range(1)), set(res.keys()))

        for res_seq in res[0]:
            self.assertTupleEqual(res_seq, seq[0])

    def test_non_domin_sort_two_front(self):
        seq = ((1,0,1), (0,1,1), (-2, -3, 0))

        res = nds.non_domin_sort(seq)

        self.assertEqual(len(res), 2)

        self.assertSetEqual(set(range(2)), set(res.keys()))

        for res_seq in res[1]:
            self.assertIn(res_seq, seq[:2])

    def test_non_domin_sort_random_elem(self):

        for dim in range(2, 5):    
            seq = [[1] * dim for i in range(51)]

            for s in seq:
                for i in range(len(s)):
                    s[i] = random.randint(-10,10)
    
            fronts = nds.non_domin_sort(seq)
    
            self.assertSetEqual(set(fronts.keys()), set(range(len(fronts))))
    
            for front_index in range(len(fronts) - 1, 0, -1):
                front_index_prev = front_index - 1
                for seq in fronts[front_index]:
                    is_dominated = False
                    for seq_prev_front in fronts[front_index_prev]:
                        is_dominated = st.is_dominate(seq_prev_front, seq)
                        if is_dominated:
                            break
                    self.assertTrue(is_dominated)




if __name__ == "__main__":
    unittest.main()