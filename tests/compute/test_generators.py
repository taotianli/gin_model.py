import dgl
import backend as F
import numpy as np
import unittest

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU random choice not implemented")
def test_rand_graph():
    g = dgl.rand_graph(10000, 100000)
    assert g.number_of_nodes() == 10000
    assert g.number_of_edges() == 100000
    # test random seed
    dgl.random.seed(42)
    g1 = dgl.rand_graph(100, 30)
    dgl.random.seed(42)
    g2 = dgl.rand_graph(100, 30)
    u1, v1 = g1.edges()
    u2, v2 = g2.edges()
    assert F.array_equal(u1, u2)
    assert F.array_equal(v1, v2)

if __name__ == '__main__':
    test_rand_graph()
