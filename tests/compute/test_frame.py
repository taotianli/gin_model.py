import dgl
import dgl.ndarray as nd
from dgl.frame import Column
import numpy as np
import backend as F
import unittest
from test_utils import parametrize_idtype

def test_column_subcolumn():
    data = F.copy_to(F.tensor([[1., 1., 1., 1.],
                               [0., 2., 9., 0.],
                               [3., 2., 1., 0.],
                               [1., 1., 1., 1.],
                               [0., 2., 4., 0.]]), F.ctx())
    original = Column(data)

    # subcolumn from cpu context 
    i1 = F.tensor([0, 2, 1, 3], dtype=F.int64)
    l1 = original.subcolumn(i1)

    assert len(l1) == i1.shape[0]
    assert F.array_equal(l1.data, F.gather_row(data, i1))

    # next subcolumn from target context
    i2 = F.copy_to(F.tensor([0, 2], dtype=F.int64), F.ctx())
    l2 = l1.subcolumn(i2)

    assert len(l2) == i2.shape[0]
    i1i2 = F.copy_to(F.gather_row(i1, F.copy_to(i2, F.context(i1))), F.ctx())
    assert F.array_equal(l2.data, F.gather_row(data,i1i2))

    # next subcolumn also from target context
    i3 = F.copy_to(F.tensor([1], dtype=F.int64), F.ctx())
    l3 = l2.subcolumn(i3)

    assert len(l3) == i3.shape[0]
    i1i2i3 = F.copy_to(F.gather_row(i1i2, F.copy_to(i3, F.context(i1i2))), F.ctx())
    assert F.array_equal(l3.data, F.gather_row(data, i1i2i3))

