import unittest
import numpy as np
import topos.fattree as ft
import embeddings.defaults as edf


class TestFatTree(unittest.TestCase):
    def setUp(self) -> None:
        self.k = 4
        self.graph = ft.make_topo(self.k)
        self.mapping = {
            'h-0000': np.array([0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,0]),
            'h-0001': np.array([0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,1]),
            'h-0002': np.array([0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,0]),
            'h-0003': np.array([0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,1]),

            'h-0004': np.array([0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,0]),
            'h-0005': np.array([0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,1]),
            'h-0006': np.array([0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,0]),
            'h-0007': np.array([0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,1]),

            'h-0008': np.array([0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,0]),
            'h-0009': np.array([0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,1]),
            'h-0010': np.array([0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,0]),
            'h-0011': np.array([0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,1]),

            'h-0012': np.array([0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,0]),
            'h-0013': np.array([0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,1]),
            'h-0014': np.array([0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,0]),
            'h-0015': np.array([0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,1]),

            'tor-0000': np.array([0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,1]),
            'tor-0001': np.array([0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,1]),

            'tor-0002': np.array([0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,1]),
            'tor-0003': np.array([0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,1]),

            'tor-0004': np.array([0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,1]),
            'tor-0005': np.array([0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,1]),

            'tor-0006': np.array([0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,1]),
            'tor-0007': np.array([0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,1,  0,0,0,0,0,0,0,1]),

            'agg-0000': np.array([0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,1]),
            'agg-0001': np.array([0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,1]),

            'agg-0002': np.array([0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,1]),
            'agg-0003': np.array([0,0,0,0,0,0,0,1,  0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,1]),

            'agg-0004': np.array([0,0,0,0,0,0,1,0,  0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,1]),
            'agg-0005': np.array([0,0,0,0,0,0,1,0,  0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,1]),

            'agg-0006': np.array([0,0,0,0,0,0,1,1,  0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,1]),
            'agg-0007': np.array([0,0,0,0,0,0,1,1,  0,0,0,0,0,0,1,1,  0,0,0,0,0,0,0,1]),

            'core-0000': np.array([0,0,0,0,0,1,0,0,  0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,1]),
            'core-0001': np.array([0,0,0,0,0,1,0,0,  0,0,0,0,0,0,0,1, 0,0,0,0,0,0,1,0]),
            'core-0002': np.array([0,0,0,0,0,1,0,0,  0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1]),
            'core-0003': np.array([0,0,0,0,0,1,0,0,  0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0])
        }

    def test_embedding(self):
        embd = edf.fat_tree_ip_scheme(self.graph, self.k)
        for k, v in embd.items():
            self.assertEqual(
                first=np.sum(np.equal(v, self.mapping[k])),
                second=24,
                msg="Arrays do not match for node {}. Expected {} got {}".format(k, str(self.mapping[k]), str(v))
            )