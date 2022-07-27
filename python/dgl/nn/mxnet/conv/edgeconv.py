"""MXNet Module for EdgeConv Layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
from mxnet.gluon import nn

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair


class EdgeConv(nn.Block):
    r"""EdgeConv layer from `Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__

    It can be described as follows:

    .. math::
       h_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} (
       \Theta \cdot (h_j^{(l)} - h_i^{(l)}) + \Phi \cdot h_i^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    :math:`\Theta` and :math:`\Phi` are linear layers.

    .. note::

       The original formulation includes a ReLU inside the maximum operator.
       This is equivalent to first applying a maximum operator then applying
       the ReLU.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    batch_norm : bool
        Whether to include batch normalization on messages. Default: ``False``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> from dgl.nn import EdgeConv
    >>>
    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = mx.nd.ones((6, 10))
    >>> conv = EdgeConv(10, 2)
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> res = conv(g, feat)
    >>> res
    [[1.0517545 0.8091326]
    [1.0517545 0.8091326]
    [1.0517545 0.8091326]
    [1.0517545 0.8091326]
    [1.0517545 0.8091326]
    [1.0517545 0.8091326]]
    <NDArray 6x2 @cpu(0)>

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_fea = mx.nd.random.randn(2, 5)
    >>> v_fea = mx.nd.random.randn(4, 5)
    >>> conv = EdgeConv(5, 2, 3)
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    [[-3.4617817   0.84700686]
    [ 1.3170856  -1.5731761 ]
    [-2.0761423   0.56653017]
    [-1.015364    0.78919804]]
    <NDArray 4x2 @cpu(0)>
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        with self.name_scope():
            self.theta = nn.Dense(out_feat, in_units=in_feat,
                                  weight_initializer=mx.init.Xavier())
            self.phi = nn.Dense(out_feat, in_units=in_feat,
                                weight_initializer=mx.init.Xavier())

            if batch_norm:
                self.bn = nn.BatchNorm(in_channels=out_feat)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, g, h):
        """

        Description
        -----------
        Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : mxnet.NDArray or pair of mxnet.NDArray
            :math:`(N, D)` where :math:`N` is the number of nodes and
            :math:`D` is the number of feature dimensions.

            If a pair of mxnet.NDArray is given, the graph must be a uni-bipartite graph
            with only one edge type, and the two tensors must have the same
            dimensionality on all except the first axis.

        Returns
        -------
        mxnet.NDArray
            New node features.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if g.in_degrees().min() == 0:
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            h_src, h_dst = expand_as_pair(h, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            g.apply_edges(fn.v_sub_u('x', 'x', 'theta'))
            g.edata['theta'] = self.theta(g.edata['theta'])
            g.dstdata['phi'] = self.phi(g.dstdata['x'])
            if not self.batch_norm:
                g.update_all(fn.e_add_v('theta', 'phi', 'e'), fn.max('e', 'x'))
            else:
                g.apply_edges(fn.e_add_v('theta', 'phi', 'e'))
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'm'), fn.max('m', 'x'))
            return g.dstdata['x']
