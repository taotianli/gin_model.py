"""Sampling utilities"""
from collections.abc import Mapping
import numpy as np

from ..utils import recursive_apply, recursive_apply_pair
from ..base import EID
from .. import backend as F
from .. import transforms, utils

def _locate_eids_to_exclude(frontier_parent_eids, exclude_eids):
    """Find the edges whose IDs in parent graph appeared in exclude_eids.

    Note that both arguments are numpy arrays or numpy dicts.
    """
    func = lambda x, y: np.isin(x, y).nonzero()[0]
    result = recursive_apply_pair(frontier_parent_eids, exclude_eids, func)
    return recursive_apply(result, F.zerocopy_from_numpy)

class EidExcluder(object):
    """Class that finds the edges whose IDs in parent graph appeared in exclude_eids.

    The edge IDs can be both CPU and GPU tensors.
    """
    def __init__(self, exclude_eids):
        device = None
        if isinstance(exclude_eids, Mapping):
            for _, v in exclude_eids.items():
                if device is None:
                    device = F.context(v)
                    break
        else:
            device = F.context(exclude_eids)
        self._exclude_eids = None
        self._filter = None

        if device == F.cpu():
            # TODO(nv-dlasalle): Once Filter is implemented for the CPU, we
            # should just use that irregardless of the device.
            self._exclude_eids = (
                recursive_apply(exclude_eids, F.zerocopy_to_numpy)
                if exclude_eids is not None else None)
        else:
            self._filter = recursive_apply(exclude_eids, utils.Filter)

    def _find_indices(self, parent_eids):
        """ Find the set of edge indices to remove.
        """
        if self._exclude_eids is not None:
            parent_eids_np = recursive_apply(parent_eids, F.zerocopy_to_numpy)
            return _locate_eids_to_exclude(parent_eids_np, self._exclude_eids)
        else:
            assert self._filter is not None
            func = lambda x, y: x.find_included_indices(y)
            return recursive_apply_pair(self._filter, parent_eids, func)

    def __call__(self, frontier):
        parent_eids = frontier.edata[EID]
        located_eids = self._find_indices(parent_eids)

        if not isinstance(located_eids, Mapping):
            # (BarclayII) If frontier already has a EID field and located_eids is empty,
            # the returned graph will keep EID intact.  Otherwise, EID will change
            # to the mapping from the new graph to the old frontier.
            # So we need to test if located_eids is empty, and do the remapping ourselves.
            if len(located_eids) > 0:
                frontier = transforms.remove_edges(
                    frontier, located_eids, store_ids=True)
                frontier.edata[EID] = F.gather_row(parent_eids, frontier.edata[EID])
        else:
            # (BarclayII) remove_edges only accepts removing one type of edges,
            # so I need to keep track of the edge IDs left one by one.
            new_eids = parent_eids.copy()
            for k, v in located_eids.items():
                if len(v) > 0:
                    frontier = transforms.remove_edges(
                        frontier, v, etype=k, store_ids=True)
                    new_eids[k] = F.gather_row(parent_eids[k], frontier.edges[k].data[EID])
            frontier.edata[EID] = new_eids
        return frontier
