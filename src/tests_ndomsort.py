import unittest
import ndomsort as nds

def identity(x):
    return x

class Testndomsort(unittest.TestCase):


    def test_non_domin_sort_many_fronts(self):
        seq = [(i,) * 4 for i in range(20)]

        res = nds.non_domin_sort(seq, identity)

        self.assertEqual(len(res), 20)

        for front in res:
            for res_seq in res[front]:
                self.assertTupleEqual(res_seq, seq[front])

    def test_non_domin_sort_one_elem(self):
        seq = [(2,3,4)]

        res = nds.non_domin_sort(seq, identity)

        self.assertTupleEqual(seq[0], res[0][0])

    def test_non_domin_sort_one_front(self):

        seq = ((1,1), (1,1), (1,1))

        res = nds.non_domin_sort(seq, identity)

        self.assertEqual(len(res), 1)

        for res_seq in res[0]:
            self.assertTupleEqual(res_seq, seq[0])

    def test_non_domin_sort_two_front(self):
        seq = ((1,0,1), (0,1,1), (-2, -3, 0))

        res = nds.non_domin_sort(seq, identity)

        self.assertEqual(len(res), 2)

        for res_seq in res[1]:
            self.assertIn(res_seq, seq[:2])

    def test_non_domin_sort_one_fitness(self):

        seq =  ( (1,), (1,), (2,), (-4,), (0,) )

        res = nds.non_domin_sort(seq, identity)
        
        self.assertEqual(len(res), 4)

        seq_list = sorted(set(seq))

        for front in res:
            for res_seq in res[front]:
                self.assertTupleEqual(res_seq, seq_list[front])

if __name__ == "__main__":
    unittest.main()