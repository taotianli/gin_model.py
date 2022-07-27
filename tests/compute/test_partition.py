from dgl.partition import NDArrayPartition
from dgl.distributed import graph_partition_book as gpb
import unittest
import backend as F
from test_utils import parametrize_idtype



@unittest.skipIf(F._default_context_str == 'cpu', reason="NDArrayPartition only works on GPU.")
@parametrize_idtype
def test_get_node_partition_from_book(idtype):
    node_map = {
        "type_n": F.tensor([
            [0,3],
            [4,5],
            [6,10]
        ], dtype=idtype)}
    edge_map = {
        "type_e": F.tensor([
            [0,9],
            [10,15],
            [16,25]
        ], dtype=idtype)}
    book = gpb.RangePartitionBook(0, 3, node_map, edge_map,
                                  {"type_n": 0}, {"type_e": 0})
    partition = gpb.get_node_partition_from_book(book, F.ctx())
    assert partition.num_parts() == 3
    assert partition.array_size() == 11

    test_ids = F.copy_to(F.tensor([0, 2, 6, 7, 10], dtype=idtype), F.ctx())
    act_ids = partition.map_to_local(test_ids)
    exp_ids = F.copy_to(F.tensor([0, 2, 0, 1, 4], dtype=idtype), F.ctx())
    assert F.array_equal(act_ids, exp_ids)

    test_ids = F.copy_to(F.tensor([0, 2], dtype=idtype), F.ctx())
    act_ids = partition.map_to_global(test_ids, 0)
    exp_ids = F.copy_to(F.tensor([0, 2], dtype=idtype), F.ctx())
    assert F.array_equal(act_ids, exp_ids)

    test_ids = F.copy_to(F.tensor([0, 1], dtype=idtype), F.ctx())
    act_ids = partition.map_to_global(test_ids, 1)
    exp_ids = F.copy_to(F.tensor([4, 5], dtype=idtype), F.ctx())
    assert F.array_equal(act_ids, exp_ids)

    test_ids = F.copy_to(F.tensor([0, 1, 4], dtype=idtype), F.ctx())
    act_ids = partition.map_to_global(test_ids, 2)
    exp_ids = F.copy_to(F.tensor([6, 7, 10], dtype=idtype), F.ctx())
    assert F.array_equal(act_ids, exp_ids)
 
