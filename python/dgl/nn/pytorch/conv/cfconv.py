"""Torch modules for interaction blocks in SchNet"""
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np
import torch.nn as nn

from .... import function as fn

class ShiftedSoftplus(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    Attributes
    ----------
    beta : int
        :math:`\beta` value for the mathematical formulation. Default to 1.
    shift : int
        :math:`\text{shift}` value for the mathematical formulation. Default to 2.
    """
    def __init__(self, beta=1, shift=2, threshold=20):
        super(ShiftedSoftplus, self).__init__()

        self.shift = shift
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, inputs):
        """

        Description
        -----------
        Applies the activation function.

        Parameters
        ----------
        inputs : float32 tensor of shape (N, *)
            * denotes any number of additional dimensions.

        Returns
        -------
        float32 tensor of shape (N, *)
            Result of applying the activation function to the input.
        """
        return self.softplus(inputs) - np.log(float(self.shift))

class CFConv(nn.Module):
    r"""CFConv from `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__

    It combines node and edge features in message passing and updates node representations.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} h_j^{l} \circ W^{(l)}e_ij

    where :math:`\circ` represents element-wise multiplication and for :math:`\text{SPP}` :

    .. math::
        \text{SSP}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) - \log(\text{shift})

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features :math:`h_j^{(l)}`.
    edge_in_feats : int
        Size for the input edge features :math:`e_ij`.
    hidden_feats : int
        Size for the hidden representations.
    out_feats : int
        Size for the output representations :math:`h_j^{(l+1)}`.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import CFConv
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> nfeat = th.ones(6, 10)
    >>> efeat = th.ones(6, 5)
    >>> conv = CFConv(10, 5, 3, 2)
    >>> res = conv(g, nfeat, efeat)
    >>> res
    tensor([[-0.1209, -0.2289],
            [-0.1209, -0.2289],
            [-0.1209, -0.2289],
            [-0.1135, -0.2338],
            [-0.1209, -0.2289],
            [-0.1283, -0.2240]], grad_fn=<SubBackward0>)
    """
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats, out_feats):
        super(CFConv, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Linear(edge_in_feats, hidden_feats),
            ShiftedSoftplus(),
            nn.Linear(hidden_feats, hidden_feats),
            ShiftedSoftplus()
        )
        self.project_node = nn.Linear(node_in_feats, hidden_feats)
        self.project_out = nn.Sequential(
            nn.Linear(hidden_feats, out_feats),
            ShiftedSoftplus()
        )

    def forward(self, g, node_feats, edge_feats):
        """

        Description
        -----------
        Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        node_feats : torch.Tensor or pair of torch.Tensor
            The input node features. If a torch.Tensor is given, it represents the input
            node feature of shape :math:`(N, D_{in})` where :math:`D_{in}` is size of
            input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph,
            the pair must contain two tensors of shape :math:`(N_{src}, D_{in_{src}})` and
            :math:`(N_{dst}, D_{in_{dst}})` separately for the source and destination nodes.

        edge_feats : torch.Tensor
            The input edge feature of shape :math:`(E, edge_in_feats)`
            where :math:`E` is the number of edges.

        Returns
        -------
        torch.Tensor
            The output node feature of shape :math:`(N_{out}, out_feats)`
            where :math:`N_{out}` is the number of destination nodes.
        """
        with g.local_scope():
            if isinstance(node_feats, tuple):
                node_feats_src, _ = node_feats
            else:
                node_feats_src = node_feats
            g.srcdata['hv'] = self.project_node(node_feats_src)
            g.edata['he'] = self.project_edge(edge_feats)
            g.update_all(fn.u_mul_e('hv', 'he', 'm'), fn.sum('m', 'h'))
            return self.project_out(g.dstdata['h'])
