import dgl
import backend as F
import numpy as np
import unittest
from test_utils import parametrize_idtype
from dgl.utils import Filter

def test_graph_filter():
    g = dgl.DGLGraph().to(F.ctx())
    g.add_nodes(4)
    g.add_edges([0,1,2,3], [1,2,3,0])

    n_repr = np.zeros((4, 5))
    e_repr = np.zeros((4, 5))
    n_repr[[1, 3]] = 1
    e_repr[[1, 3]] = 1
    n_repr = F.copy_to(F.zerocopy_from_numpy(n_repr), F.ctx())
    e_repr = F.copy_to(F.zerocopy_from_numpy(e_repr), F.ctx())

    g.ndata['a'] = n_repr
    g.edata['a'] = e_repr

    def predicate(r):
        return F.max(r.data['a'], 1) > 0

    # full node filter
    n_idx = g.filter_nodes(predicate)
    assert set(F.zerocopy_to_numpy(n_idx)) == {1, 3}

    # partial node filter
    n_idx = g.filter_nodes(predicate, [0, 1])
    assert set(F.zerocopy_to_numpy(n_idx)) == {1}

    # full edge filter
    e_idx = g.filter_edges(predicate)
    assert set(F.zerocopy_to_numpy(e_idx)) == {1, 3}

    # partial edge filter
    e_idx = g.filter_edges(predicate, [0, 1])
    assert set(F.zerocopy_to_numpy(e_idx)) == {1}

@unittest.skipIf(F._default_context_str == 'cpu',
                 reason="CPU not yet supported")
@parametrize_idtype
def test_array_filter(idtype):
    f = Filter(F.copy_to(F.tensor([0,1,9,4,6,5,7], dtype=idtype), F.ctx()))
    x = F.copy_to(F.tensor([0,3,9,11], dtype=idtype), F.ctx())
    y = F.copy_to(F.tensor([0,19,0,28,3,9,11,4,5], dtype=idtype), F.ctx())

    xi_act = f.find_included_indices(x)
    xi_exp = F.copy_to(F.tensor([0,2], dtype=idtype), F.ctx())
    assert F.array_equal(xi_act, xi_exp)
    xe_act = f.find_excluded_indices(x)
    xe_exp = F.copy_to(F.tensor([1,3], dtype=idtype), F.ctx())
    assert F.array_equal(xe_act, xe_exp)

    yi_act = f.find_included_indices(y)
    yi_exp = F.copy_to(F.tensor([0,2,5,7,8], dtype=idtype), F.ctx())
    assert F.array_equal(yi_act, yi_exp)
    ye_act = f.find_excluded_indices(y)
    ye_exp = F.copy_to(F.tensor([1,3,4,6], dtype=idtype), F.ctx())
    assert F.array_equal(ye_act, ye_exp)

if __name__ == '__main__':
    test_graph_filter()
    test_array_filter()
