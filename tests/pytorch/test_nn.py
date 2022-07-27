import io
import torch as th
import networkx as nx
import dgl
import dgl.nn.pytorch as nn
import dgl.function as fn
import backend as F
import pytest
from test_utils.graph_cases import get_cases, random_graph, random_bipartite, random_dglgraph
from test_utils import parametrize_idtype
from copy import deepcopy
import pickle

import scipy as sp

tmp_buffer = io.BytesIO()

def _AXWb(A, X, W, b):
    X = th.matmul(X, W)
    Y = th.matmul(A, X.view(X.shape[0], -1)).view_as(X)
    return Y + b

@pytest.mark.parametrize('out_dim', [1, 2])
def test_graph_conv0(out_dim):
    g = dgl.DGLGraph(nx.path_graph(3)).to(F.ctx())
    ctx = F.ctx()
    adj = g.adjacency_matrix(transpose=True, ctx=ctx)

    conv = nn.GraphConv(5, out_dim, norm='none', bias=True)
    conv = conv.to(ctx)
    print(conv)

    # test pickle
    th.save(conv, tmp_buffer)


    # test#1: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert F.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))
    # test#2: more-dim
    h0 = F.ones((3, 5, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    assert F.allclose(h1, _AXWb(adj, h0, conv.weight, conv.bias))

    conv = nn.GraphConv(5, out_dim)
    conv = conv.to(ctx)
    # test#3: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # test#4: basic
    h0 = F.ones((3, 5, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0

    conv = nn.GraphConv(5, out_dim)
    conv = conv.to(ctx)
    # test#3: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    # test#4: basic
    h0 = F.ones((3, 5, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0

    # test rest_parameters
    old_weight = deepcopy(conv.weight.data)
    conv.reset_parameters()
    new_weight = conv.weight.data
    assert not F.allclose(old_weight, new_weight)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'bipartite'], exclude=['zero-degree', 'dglgraph']))
@pytest.mark.parametrize('norm', ['none', 'both', 'right', 'left'])
@pytest.mark.parametrize('weight', [True, False])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('out_dim', [1, 2])
def test_graph_conv(idtype, g, norm, weight, bias, out_dim):
    # Test one tensor input
    g = g.astype(idtype).to(F.ctx())
    conv = nn.GraphConv(5, out_dim, norm=norm, weight=weight, bias=bias).to(F.ctx())
    ext_w = F.randn((5, out_dim)).to(F.ctx())
    nsrc = g.number_of_src_nodes()
    ndst = g.number_of_dst_nodes()
    h = F.randn((nsrc, 5)).to(F.ctx())
    if weight:
        h_out = conv(g, h)
    else:
        h_out = conv(g, h, weight=ext_w)
    assert h_out.shape == (ndst, out_dim)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['has_scalar_e_feature'], exclude=['zero-degree', 'dglgraph']))
@pytest.mark.parametrize('norm', ['none', 'both', 'right'])
@pytest.mark.parametrize('weight', [True, False])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('out_dim', [1, 2])
def test_graph_conv_e_weight(idtype, g, norm, weight, bias, out_dim):
    g = g.astype(idtype).to(F.ctx())
    conv = nn.GraphConv(5, out_dim, norm=norm, weight=weight, bias=bias).to(F.ctx())
    ext_w = F.randn((5, out_dim)).to(F.ctx())
    nsrc = g.number_of_src_nodes()
    ndst = g.number_of_dst_nodes()
    h = F.randn((nsrc, 5)).to(F.ctx())
    e_w = g.edata['scalar_w']
    if weight:
        h_out = conv(g, h, edge_weight=e_w)
    else:
        h_out = conv(g, h, weight=ext_w, edge_weight=e_w)
    assert h_out.shape == (ndst, out_dim)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['has_scalar_e_feature'], exclude=['zero-degree', 'dglgraph']))
@pytest.mark.parametrize('norm', ['none', 'both', 'right'])
@pytest.mark.parametrize('weight', [True, False])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('out_dim', [1, 2])
def test_graph_conv_e_weight_norm(idtype, g, norm, weight, bias, out_dim):
    g = g.astype(idtype).to(F.ctx())
    conv = nn.GraphConv(5, out_dim, norm=norm, weight=weight, bias=bias).to(F.ctx())

    # test pickle
    th.save(conv, tmp_buffer)

    ext_w = F.randn((5, out_dim)).to(F.ctx())
    nsrc = g.number_of_src_nodes()
    ndst = g.number_of_dst_nodes()
    h = F.randn((nsrc, 5)).to(F.ctx())
    edgenorm = nn.EdgeWeightNorm(norm=norm)
    norm_weight = edgenorm(g, g.edata['scalar_w'])
    if weight:
        h_out = conv(g, h, edge_weight=norm_weight)
    else:
        h_out = conv(g, h, weight=ext_w, edge_weight=norm_weight)
    assert h_out.shape == (ndst, out_dim)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree', 'dglgraph']))
@pytest.mark.parametrize('norm', ['none', 'both', 'right'])
@pytest.mark.parametrize('weight', [True, False])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('out_dim', [1, 2])
def test_graph_conv_bi(idtype, g, norm, weight, bias, out_dim):
    # Test a pair of tensor inputs
    g = g.astype(idtype).to(F.ctx())
    conv = nn.GraphConv(5, out_dim, norm=norm, weight=weight, bias=bias).to(F.ctx())

    # test pickle
    th.save(conv, tmp_buffer)

    ext_w = F.randn((5, out_dim)).to(F.ctx())
    nsrc = g.number_of_src_nodes()
    ndst = g.number_of_dst_nodes()
    h = F.randn((nsrc, 5)).to(F.ctx())
    h_dst = F.randn((ndst, out_dim)).to(F.ctx())
    if weight:
        h_out = conv(g, (h, h_dst))
    else:
        h_out = conv(g, (h, h_dst), weight=ext_w)
    assert h_out.shape == (ndst, out_dim)

def _S2AXWb(A, N, X, W, b):
    X1 = X * N
    X1 = th.matmul(A, X1.view(X1.shape[0], -1))
    X1 = X1 * N
    X2 = X1 * N
    X2 = th.matmul(A, X2.view(X2.shape[0], -1))
    X2 = X2 * N
    X = th.cat([X, X1, X2], dim=-1)
    Y = th.matmul(X, W.rot90())

    return Y + b

@pytest.mark.parametrize('out_dim', [1, 2])
def test_tagconv(out_dim):
    g = dgl.DGLGraph(nx.path_graph(3))
    g = g.to(F.ctx())
    ctx = F.ctx()
    adj = g.adjacency_matrix(transpose=True, ctx=ctx)
    norm = th.pow(g.in_degrees().float(), -0.5)

    conv = nn.TAGConv(5, out_dim, bias=True)
    conv = conv.to(ctx)
    print(conv)

    # test pickle
    th.save(conv, tmp_buffer)

    # test#1: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert len(g.ndata) == 0
    assert len(g.edata) == 0
    shp = norm.shape + (1,) * (h0.dim() - 1)
    norm = th.reshape(norm, shp).to(ctx)

    assert F.allclose(h1, _S2AXWb(adj, norm, h0, conv.lin.weight, conv.lin.bias))

    conv = nn.TAGConv(5, out_dim)
    conv = conv.to(ctx)

    # test#2: basic
    h0 = F.ones((3, 5))
    h1 = conv(g, h0)
    assert h1.shape[-1] == out_dim

    # test reset_parameters
    old_weight = deepcopy(conv.lin.weight.data)
    conv.reset_parameters()
    new_weight = conv.lin.weight.data
    assert not F.allclose(old_weight, new_weight)

def test_set2set():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(10))
    g = g.to(F.ctx())

    s2s = nn.Set2Set(5, 3, 3) # hidden size 5, 3 iters, 3 layers
    s2s = s2s.to(ctx)
    print(s2s)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = s2s(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 and h1.dim() == 2

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(11)).to(F.ctx())
    g2 = dgl.DGLGraph(nx.path_graph(5)).to(F.ctx())
    bg = dgl.batch([g, g1, g2])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = s2s(bg, h0)
    assert h1.shape[0] == 3 and h1.shape[1] == 10 and h1.dim() == 2

def test_glob_att_pool():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(10))
    g = g.to(F.ctx())

    gap = nn.GlobalAttentionPooling(th.nn.Linear(5, 1), th.nn.Linear(5, 10))
    gap = gap.to(ctx)
    print(gap)

    # test pickle
    th.save(gap, tmp_buffer)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    h1 = gap(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 and h1.dim() == 2

    # test#2: batched graph
    bg = dgl.batch([g, g, g, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = gap(bg, h0)
    assert h1.shape[0] == 4 and h1.shape[1] == 10 and h1.dim() == 2

def test_simple_pool():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(15))
    g = g.to(F.ctx())

    sum_pool = nn.SumPooling()
    avg_pool = nn.AvgPooling()
    max_pool = nn.MaxPooling()
    sort_pool = nn.SortPooling(10) # k = 10
    print(sum_pool, avg_pool, max_pool, sort_pool)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 5))
    sum_pool = sum_pool.to(ctx)
    avg_pool = avg_pool.to(ctx)
    max_pool = max_pool.to(ctx)
    sort_pool = sort_pool.to(ctx)
    h1 = sum_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.sum(h0, 0))
    h1 = avg_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.mean(h0, 0))
    h1 = max_pool(g, h0)
    assert F.allclose(F.squeeze(h1, 0), F.max(h0, 0))
    h1 = sort_pool(g, h0)
    assert h1.shape[0] == 1 and h1.shape[1] == 10 * 5 and h1.dim() == 2

    # test#2: batched graph
    g_ = dgl.DGLGraph(nx.path_graph(5)).to(F.ctx())
    bg = dgl.batch([g, g_, g, g_, g])
    h0 = F.randn((bg.number_of_nodes(), 5))
    h1 = sum_pool(bg, h0)
    truth = th.stack([F.sum(h0[:15], 0),
                      F.sum(h0[15:20], 0),
                      F.sum(h0[20:35], 0),
                      F.sum(h0[35:40], 0),
                      F.sum(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = avg_pool(bg, h0)
    truth = th.stack([F.mean(h0[:15], 0),
                      F.mean(h0[15:20], 0),
                      F.mean(h0[20:35], 0),
                      F.mean(h0[35:40], 0),
                      F.mean(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = max_pool(bg, h0)
    truth = th.stack([F.max(h0[:15], 0),
                      F.max(h0[15:20], 0),
                      F.max(h0[20:35], 0),
                      F.max(h0[35:40], 0),
                      F.max(h0[40:55], 0)], 0)
    assert F.allclose(h1, truth)

    h1 = sort_pool(bg, h0)
    assert h1.shape[0] == 5 and h1.shape[1] == 10 * 5 and h1.dim() == 2

def test_set_trans():
    ctx = F.ctx()
    g = dgl.DGLGraph(nx.path_graph(15))

    st_enc_0 = nn.SetTransformerEncoder(50, 5, 10, 100, 2, 'sab')
    st_enc_1 = nn.SetTransformerEncoder(50, 5, 10, 100, 2, 'isab', 3)
    st_dec = nn.SetTransformerDecoder(50, 5, 10, 100, 2, 4)
    st_enc_0 = st_enc_0.to(ctx)
    st_enc_1 = st_enc_1.to(ctx)
    st_dec = st_dec.to(ctx)
    print(st_enc_0, st_enc_1, st_dec)

    # test#1: basic
    h0 = F.randn((g.number_of_nodes(), 50))
    h1 = st_enc_0(g, h0)
    assert h1.shape == h0.shape
    h1 = st_enc_1(g, h0)
    assert h1.shape == h0.shape
    h2 = st_dec(g, h1)
    assert h2.shape[0] == 1 and h2.shape[1] == 200 and h2.dim() == 2

    # test#2: batched graph
    g1 = dgl.DGLGraph(nx.path_graph(5))
    g2 = dgl.DGLGraph(nx.path_graph(10))
    bg = dgl.batch([g, g1, g2])
    h0 = F.randn((bg.number_of_nodes(), 50))
    h1 = st_enc_0(bg, h0)
    assert h1.shape == h0.shape
    h1 = st_enc_1(bg, h0)
    assert h1.shape == h0.shape

    h2 = st_dec(bg, h1)
    assert h2.shape[0] == 3 and h2.shape[1] == 200 and h2.dim() == 2

@parametrize_idtype
@pytest.mark.parametrize('O', [1, 8, 32])
def test_rgcn(idtype, O):
    ctx = F.ctx()
    etype = []
    g = dgl.from_scipy(sp.sparse.random(100, 100, density=0.1))
    g = g.astype(idtype).to(F.ctx())
    # 5 etypes
    R = 5
    for i in range(g.number_of_edges()):
        etype.append(i % 5)
    B = 2
    I = 10

    h = th.randn((100, I)).to(ctx)
    r = th.tensor(etype).to(ctx)
    norm = th.rand((g.number_of_edges(), 1)).to(ctx)
    sorted_r, idx = th.sort(r)
    sorted_g = dgl.reorder_graph(g, edge_permute_algo='custom', permute_config={'edges_perm' : idx.to(idtype)})
    sorted_norm = norm[idx]

    rgc = nn.RelGraphConv(I, O, R).to(ctx)
    th.save(rgc, tmp_buffer)  # test pickle
    rgc_basis = nn.RelGraphConv(I, O, R, "basis", B).to(ctx)
    th.save(rgc_basis, tmp_buffer)  # test pickle
    if O % B == 0:
        rgc_bdd = nn.RelGraphConv(I, O, R, "bdd", B).to(ctx)
        th.save(rgc_bdd, tmp_buffer)  # test pickle

    # basic usage
    h_new = rgc(g, h, r)
    assert h_new.shape == (100, O)
    h_new_basis = rgc_basis(g, h, r)
    assert h_new_basis.shape == (100, O)
    if O % B == 0:
        h_new_bdd = rgc_bdd(g, h, r)
        assert h_new_bdd.shape == (100, O)

    # sorted input
    h_new_sorted = rgc(sorted_g, h, sorted_r, presorted=True)
    assert th.allclose(h_new, h_new_sorted, atol=1e-4, rtol=1e-4)
    h_new_basis_sorted = rgc_basis(sorted_g, h, sorted_r, presorted=True)
    assert th.allclose(h_new_basis, h_new_basis_sorted, atol=1e-4, rtol=1e-4)
    if O % B == 0:
        h_new_bdd_sorted = rgc_bdd(sorted_g, h, sorted_r, presorted=True)
        assert th.allclose(h_new_bdd, h_new_bdd_sorted, atol=1e-4, rtol=1e-4)

    # norm input
    h_new = rgc(g, h, r, norm)
    assert h_new.shape == (100, O)
    h_new = rgc_basis(g, h, r, norm)
    assert h_new.shape == (100, O)
    if O % B == 0:
        h_new = rgc_bdd(g, h, r, norm)
        assert h_new.shape == (100, O)


@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 5])
@pytest.mark.parametrize('num_heads', [1, 4])
def test_gat_conv(g, idtype, out_dim, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gat = nn.GATConv(5, out_dim, num_heads)
    feat = F.randn((g.number_of_src_nodes(), 5))
    gat = gat.to(ctx)
    h = gat(g, feat)

    # test pickle
    th.save(gat, tmp_buffer)

    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_dim)
    _, a = gat(g, feat, get_attention=True)
    assert a.shape == (g.number_of_edges(), num_heads, 1)

    # test residual connection
    gat = nn.GATConv(5, out_dim, num_heads, residual=True)
    gat = gat.to(ctx)
    h = gat(g, feat)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
@pytest.mark.parametrize('num_heads', [1, 4])
def test_gat_conv_bi(g, idtype, out_dim, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gat = nn.GATConv(5, out_dim, num_heads)
    feat = (F.randn((g.number_of_src_nodes(), 5)), F.randn((g.number_of_dst_nodes(), 5)))
    gat = gat.to(ctx)
    h = gat(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_dim)
    _, a = gat(g, feat, get_attention=True)
    assert a.shape == (g.number_of_edges(), num_heads, 1)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 5])
@pytest.mark.parametrize('num_heads', [1, 4])
def test_gatv2_conv(g, idtype, out_dim, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gat = nn.GATv2Conv(5, out_dim, num_heads)
    feat = F.randn((g.number_of_src_nodes(), 5))
    gat = gat.to(ctx)
    h = gat(g, feat)

    # test pickle
    th.save(gat, tmp_buffer)

    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_dim)
    _, a = gat(g, feat, get_attention=True)
    assert a.shape == (g.number_of_edges(), num_heads, 1)

    # test residual connection
    gat = nn.GATConv(5, out_dim, num_heads, residual=True)
    gat = gat.to(ctx)
    h = gat(g, feat)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
@pytest.mark.parametrize('num_heads', [1, 4])
def test_gatv2_conv_bi(g, idtype, out_dim, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gat = nn.GATv2Conv(5, out_dim, num_heads)
    feat = (F.randn((g.number_of_src_nodes(), 5)), F.randn((g.number_of_dst_nodes(), 5)))
    gat = gat.to(ctx)
    h = gat(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_dim)
    _, a = gat(g, feat, get_attention=True)
    assert a.shape == (g.number_of_edges(), num_heads, 1)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_node_feats', [1, 5])
@pytest.mark.parametrize('out_edge_feats', [1, 5])
@pytest.mark.parametrize('num_heads', [1, 4])
def test_egat_conv(g, idtype, out_node_feats, out_edge_feats, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    egat = nn.EGATConv(in_node_feats=10,
                       in_edge_feats=5,
                       out_node_feats=out_node_feats,
                       out_edge_feats=out_edge_feats,
                       num_heads=num_heads)
    nfeat = F.randn((g.number_of_nodes(), 10))
    efeat = F.randn((g.number_of_edges(), 5))
    egat = egat.to(ctx)
    h, f = egat(g, nfeat, efeat)
    
    th.save(egat, tmp_buffer)

    assert h.shape == (g.number_of_nodes(), num_heads, out_node_feats)
    assert f.shape == (g.number_of_edges(), num_heads, out_edge_feats)
    _, _, attn = egat(g, nfeat, efeat, True)
    assert attn.shape == (g.number_of_edges(), num_heads, 1)
    
@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_node_feats', [1, 5])
@pytest.mark.parametrize('out_edge_feats', [1, 5])
@pytest.mark.parametrize('num_heads', [1, 4])
def test_egat_conv_bi(g, idtype, out_node_feats, out_edge_feats, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    egat = nn.EGATConv(in_node_feats=(10,15),
                       in_edge_feats=7,
                       out_node_feats=out_node_feats,
                       out_edge_feats=out_edge_feats,
                       num_heads=num_heads)
    nfeat = (F.randn((g.number_of_src_nodes(), 10)), F.randn((g.number_of_dst_nodes(), 15)))
    efeat = F.randn((g.number_of_edges(), 7))
    egat = egat.to(ctx)
    h, f = egat(g, nfeat, efeat)
    
    th.save(egat, tmp_buffer)

    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_node_feats)
    assert f.shape == (g.number_of_edges(), num_heads, out_edge_feats)
    _, _, attn = egat(g, nfeat, efeat, True)
    assert attn.shape == (g.number_of_edges(), num_heads, 1)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite']))
@pytest.mark.parametrize('aggre_type', ['mean', 'pool', 'gcn', 'lstm'])
def test_sage_conv(idtype, g, aggre_type):
    g = g.astype(idtype).to(F.ctx())
    sage = nn.SAGEConv(5, 10, aggre_type)
    feat = F.randn((g.number_of_src_nodes(), 5))
    sage = sage.to(F.ctx())
    # test pickle
    th.save(sage, tmp_buffer)
    h = sage(g, feat)
    assert h.shape[-1] == 10

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite']))
@pytest.mark.parametrize('aggre_type', ['mean', 'pool', 'gcn', 'lstm'])
@pytest.mark.parametrize('out_dim', [1, 2])
def test_sage_conv_bi(idtype, g, aggre_type, out_dim):
    g = g.astype(idtype).to(F.ctx())
    dst_dim = 5 if aggre_type != 'gcn' else 10
    sage = nn.SAGEConv((10, dst_dim), out_dim, aggre_type)
    feat = (F.randn((g.number_of_src_nodes(), 10)), F.randn((g.number_of_dst_nodes(), dst_dim)))
    sage = sage.to(F.ctx())
    h = sage(g, feat)
    assert h.shape[-1] == out_dim
    assert h.shape[0] == g.number_of_dst_nodes()

@parametrize_idtype
@pytest.mark.parametrize('out_dim', [1, 2])
def test_sage_conv2(idtype, out_dim):
    # TODO: add test for blocks
    # Test the case for graphs without edges
    g = dgl.heterograph({('_U', '_E', '_V'): ([], [])}, {'_U': 5, '_V': 3})
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    sage = nn.SAGEConv((3, 3), out_dim, 'gcn')
    feat = (F.randn((5, 3)), F.randn((3, 3)))
    sage = sage.to(ctx)
    h = sage(g, (F.copy_to(feat[0], F.ctx()), F.copy_to(feat[1], F.ctx())))
    assert h.shape[-1] == out_dim
    assert h.shape[0] == 3
    for aggre_type in ['mean', 'pool', 'lstm']:
        sage = nn.SAGEConv((3, 1), out_dim, aggre_type)
        feat = (F.randn((5, 3)), F.randn((3, 1)))
        sage = sage.to(ctx)
        h = sage(g, feat)
        assert h.shape[-1] == out_dim
        assert h.shape[0] == 3

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
def test_sgc_conv(g, idtype, out_dim):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    # not cached
    sgc = nn.SGConv(5, out_dim, 3)

    # test pickle
    th.save(sgc, tmp_buffer)

    feat = F.randn((g.number_of_nodes(), 5))
    sgc = sgc.to(ctx)

    h = sgc(g, feat)
    assert h.shape[-1] == out_dim

    # cached
    sgc = nn.SGConv(5, out_dim, 3, True)
    sgc = sgc.to(ctx)
    h_0 = sgc(g, feat)
    h_1 = sgc(g, feat + 1)
    assert F.allclose(h_0, h_1)
    assert h_0.shape[-1] == out_dim

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_appnp_conv(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    appnp = nn.APPNPConv(10, 0.1)
    feat = F.randn((g.number_of_nodes(), 5))
    appnp = appnp.to(ctx)

    # test pickle
    th.save(appnp, tmp_buffer)

    h = appnp(g, feat)
    assert h.shape[-1] == 5


@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_appnp_conv_e_weight(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    appnp = nn.APPNPConv(10, 0.1)
    feat = F.randn((g.number_of_nodes(), 5))
    eweight = F.ones((g.num_edges(), ))
    appnp = appnp.to(ctx)

    h = appnp(g, feat, edge_weight=eweight)
    assert h.shape[-1] == 5

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_gcn2conv_e_weight(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    gcn2conv = nn.GCN2Conv(5, layer=2, alpha=0.5,
                           project_initial_features=True)
    feat = F.randn((g.number_of_nodes(), 5))
    eweight = F.ones((g.num_edges(), ))
    gcn2conv = gcn2conv.to(ctx)
    res = feat
    h = gcn2conv(g, res, feat, edge_weight=eweight)
    assert h.shape[-1] == 5


@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_sgconv_e_weight(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    sgconv = nn.SGConv(5, 5, 3)
    feat = F.randn((g.number_of_nodes(), 5))
    eweight = F.ones((g.num_edges(), ))
    sgconv = sgconv.to(ctx)
    h = sgconv(g, feat, edge_weight=eweight)
    assert h.shape[-1] == 5

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_tagconv_e_weight(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    conv = nn.TAGConv(5, 5, bias=True)
    conv = conv.to(ctx)
    feat = F.randn((g.number_of_nodes(), 5))
    eweight = F.ones((g.num_edges(), ))
    conv = conv.to(ctx)
    h = conv(g, feat, edge_weight=eweight)
    assert h.shape[-1] == 5

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('aggregator_type', ['mean', 'max', 'sum'])
def test_gin_conv(g, idtype, aggregator_type):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gin = nn.GINConv(
        th.nn.Linear(5, 12),
        aggregator_type
    )
    th.save(gin, tmp_buffer)
    feat = F.randn((g.number_of_src_nodes(), 5))
    gin = gin.to(ctx)
    h = gin(g, feat)

    # test pickle
    th.save(gin, tmp_buffer)

    assert h.shape == (g.number_of_dst_nodes(), 12)

    gin = nn.GINConv(None, aggregator_type)
    th.save(gin, tmp_buffer)
    gin = gin.to(ctx)
    h = gin(g, feat)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite']))
def test_gine_conv(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    gine = nn.GINEConv(
        th.nn.Linear(5, 12)
    )
    th.save(gine, tmp_buffer)
    nfeat = F.randn((g.number_of_src_nodes(), 5))
    efeat = F.randn((g.num_edges(), 5))
    gine = gine.to(ctx)
    h = gine(g, nfeat, efeat)

    # test pickle
    th.save(gine, tmp_buffer)
    assert h.shape == (g.number_of_dst_nodes(), 12)

    gine = nn.GINEConv(None)
    th.save(gine, tmp_buffer)
    gine = gine.to(ctx)
    h = gine(g, nfeat, efeat)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('aggregator_type', ['mean', 'max', 'sum'])
def test_gin_conv_bi(g, idtype, aggregator_type):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gin = nn.GINConv(
        th.nn.Linear(5, 12),
        aggregator_type
    )
    feat = (F.randn((g.number_of_src_nodes(), 5)), F.randn((g.number_of_dst_nodes(), 5)))
    gin = gin.to(ctx)
    h = gin(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), 12)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree']))
def test_agnn_conv(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    agnn = nn.AGNNConv(1)
    feat = F.randn((g.number_of_src_nodes(), 5))
    agnn = agnn.to(ctx)
    h = agnn(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), 5)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
def test_agnn_conv_bi(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    agnn = nn.AGNNConv(1)
    feat = (F.randn((g.number_of_src_nodes(), 5)), F.randn((g.number_of_dst_nodes(), 5)))
    agnn = agnn.to(ctx)
    h = agnn(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), 5)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_gated_graph_conv(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    ggconv = nn.GatedGraphConv(5, 10, 5, 3)
    etypes = th.arange(g.number_of_edges()) % 3
    feat = F.randn((g.number_of_nodes(), 5))
    ggconv = ggconv.to(ctx)
    etypes = etypes.to(ctx)

    h = ggconv(g, feat, etypes)
    # current we only do shape check
    assert h.shape[-1] == 10

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_gated_graph_conv_one_etype(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    ggconv = nn.GatedGraphConv(5, 10, 5, 1)
    etypes = th.zeros(g.number_of_edges())
    feat = F.randn((g.number_of_nodes(), 5))
    ggconv = ggconv.to(ctx)
    etypes = etypes.to(ctx)

    h = ggconv(g, feat, etypes)
    h2 = ggconv(g, feat)
    # current we only do shape check
    assert F.allclose(h, h2)
    assert h.shape[-1] == 10

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree']))
def test_nn_conv(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    edge_func = th.nn.Linear(4, 5 * 10)
    nnconv = nn.NNConv(5, 10, edge_func, 'mean')
    feat = F.randn((g.number_of_src_nodes(), 5))
    efeat = F.randn((g.number_of_edges(), 4))
    nnconv = nnconv.to(ctx)
    h = nnconv(g, feat, efeat)
    # currently we only do shape check
    assert h.shape[-1] == 10

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
def test_nn_conv_bi(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    edge_func = th.nn.Linear(4, 5 * 10)
    nnconv = nn.NNConv((5, 2), 10, edge_func, 'mean')
    feat = F.randn((g.number_of_src_nodes(), 5))
    feat_dst = F.randn((g.number_of_dst_nodes(), 2))
    efeat = F.randn((g.number_of_edges(), 4))
    nnconv = nnconv.to(ctx)
    h = nnconv(g, (feat, feat_dst), efeat)
    # currently we only do shape check
    assert h.shape[-1] == 10

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_gmm_conv(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gmmconv = nn.GMMConv(5, 10, 3, 4, 'mean')
    feat = F.randn((g.number_of_nodes(), 5))
    pseudo = F.randn((g.number_of_edges(), 3))
    gmmconv = gmmconv.to(ctx)
    h = gmmconv(g, feat, pseudo)
    # currently we only do shape check
    assert h.shape[-1] == 10

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite', 'block-bipartite'], exclude=['zero-degree']))
def test_gmm_conv_bi(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    gmmconv = nn.GMMConv((5, 2), 10, 3, 4, 'mean')
    feat = F.randn((g.number_of_src_nodes(), 5))
    feat_dst = F.randn((g.number_of_dst_nodes(), 2))
    pseudo = F.randn((g.number_of_edges(), 3))
    gmmconv = gmmconv.to(ctx)
    h = gmmconv(g, (feat, feat_dst), pseudo)
    # currently we only do shape check
    assert h.shape[-1] == 10

@parametrize_idtype
@pytest.mark.parametrize('norm_type', ['both', 'right', 'none'])
@pytest.mark.parametrize('g', get_cases(['homo', 'bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
def test_dense_graph_conv(norm_type, g, idtype, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    # TODO(minjie): enable the following option after #1385
    adj = g.adjacency_matrix(transpose=True, ctx=ctx).to_dense()
    conv = nn.GraphConv(5, out_dim, norm=norm_type, bias=True)
    dense_conv = nn.DenseGraphConv(5, out_dim, norm=norm_type, bias=True)
    dense_conv.weight.data = conv.weight.data
    dense_conv.bias.data = conv.bias.data
    feat = F.randn((g.number_of_src_nodes(), 5))
    conv = conv.to(ctx)
    dense_conv = dense_conv.to(ctx)
    out_conv = conv(g, feat)
    out_dense_conv = dense_conv(adj, feat)
    assert F.allclose(out_conv, out_dense_conv)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'bipartite']))
@pytest.mark.parametrize('out_dim', [1, 2])
def test_dense_sage_conv(g, idtype, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    adj = g.adjacency_matrix(transpose=True, ctx=ctx).to_dense()
    sage = nn.SAGEConv(5, out_dim, 'gcn')
    dense_sage = nn.DenseSAGEConv(5, out_dim)
    dense_sage.fc.weight.data = sage.fc_neigh.weight.data
    dense_sage.fc.bias.data = sage.bias.data
    if len(g.ntypes) == 2:
        feat = (
            F.randn((g.number_of_src_nodes(), 5)),
            F.randn((g.number_of_dst_nodes(), 5))
        )
    else:
        feat = F.randn((g.number_of_nodes(), 5))
    sage = sage.to(ctx)
    dense_sage = dense_sage.to(ctx)
    out_sage = sage(g, feat)
    out_dense_sage = dense_sage(adj, feat)
    assert F.allclose(out_sage, out_dense_sage), g

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
def test_edge_conv(g, idtype, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    edge_conv = nn.EdgeConv(5, out_dim).to(ctx)
    print(edge_conv)

    # test pickle
    th.save(edge_conv, tmp_buffer)

    h0 = F.randn((g.number_of_src_nodes(), 5))
    h1 = edge_conv(g, h0)
    assert h1.shape == (g.number_of_dst_nodes(), out_dim)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
def test_edge_conv_bi(g, idtype, out_dim):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    edge_conv = nn.EdgeConv(5, out_dim).to(ctx)
    print(edge_conv)
    h0 = F.randn((g.number_of_src_nodes(), 5))
    x0 = F.randn((g.number_of_dst_nodes(), 5))
    h1 = edge_conv(g, (h0, x0))
    assert h1.shape == (g.number_of_dst_nodes(), out_dim)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'block-bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
@pytest.mark.parametrize('num_heads', [1, 4])
def test_dotgat_conv(g, idtype, out_dim, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    dotgat = nn.DotGatConv(5, out_dim, num_heads)
    feat = F.randn((g.number_of_src_nodes(), 5))
    dotgat = dotgat.to(ctx)

    # test pickle
    th.save(dotgat, tmp_buffer)

    h = dotgat(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_dim)
    _, a = dotgat(g, feat, get_attention=True)
    assert a.shape == (g.number_of_edges(), num_heads, 1)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
@pytest.mark.parametrize('num_heads', [1, 4])
def test_dotgat_conv_bi(g, idtype, out_dim, num_heads):
    g = g.astype(idtype).to(F.ctx())
    ctx = F.ctx()
    dotgat = nn.DotGatConv((5, 5), out_dim, num_heads)
    feat = (F.randn((g.number_of_src_nodes(), 5)), F.randn((g.number_of_dst_nodes(), 5)))
    dotgat = dotgat.to(ctx)
    h = dotgat(g, feat)
    assert h.shape == (g.number_of_dst_nodes(), num_heads, out_dim)
    _, a = dotgat(g, feat, get_attention=True)
    assert a.shape == (g.number_of_edges(), num_heads, 1)

@pytest.mark.parametrize('out_dim', [1, 2])
def test_dense_cheb_conv(out_dim):
    for k in range(1, 4):
        ctx = F.ctx()
        g = dgl.DGLGraph(sp.sparse.random(100, 100, density=0.1), readonly=True)
        g = g.to(F.ctx())
        adj = g.adjacency_matrix(transpose=True, ctx=ctx).to_dense()
        cheb = nn.ChebConv(5, out_dim, k, None)
        dense_cheb = nn.DenseChebConv(5, out_dim, k)
        #for i in range(len(cheb.fc)):
        #    dense_cheb.W.data[i] = cheb.fc[i].weight.data.t()
        dense_cheb.W.data = cheb.linear.weight.data.transpose(-1, -2).view(k, 5, out_dim)
        if cheb.linear.bias is not None:
            dense_cheb.bias.data = cheb.linear.bias.data
        feat = F.randn((100, 5))
        cheb = cheb.to(ctx)
        dense_cheb = dense_cheb.to(ctx)
        out_cheb = cheb(g, feat, [2.0])
        out_dense_cheb = dense_cheb(adj, feat, 2.0)
        print(k, out_cheb, out_dense_cheb)
        assert F.allclose(out_cheb, out_dense_cheb)

def test_sequential():
    ctx = F.ctx()
    # Test single graph
    class ExampleLayer(th.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, graph, n_feat, e_feat):
            graph = graph.local_var()
            graph.ndata['h'] = n_feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            n_feat += graph.ndata['h']
            graph.apply_edges(fn.u_add_v('h', 'h', 'e'))
            e_feat += graph.edata['e']
            return n_feat, e_feat

    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    g = g.to(F.ctx())
    net = nn.Sequential(ExampleLayer(), ExampleLayer(), ExampleLayer())
    n_feat = F.randn((3, 4))
    e_feat = F.randn((9, 4))
    net = net.to(ctx)
    n_feat, e_feat = net(g, n_feat, e_feat)
    assert n_feat.shape == (3, 4)
    assert e_feat.shape == (9, 4)

    # Test multiple graph
    class ExampleLayer(th.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, graph, n_feat):
            graph = graph.local_var()
            graph.ndata['h'] = n_feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            n_feat += graph.ndata['h']
            return n_feat.view(graph.number_of_nodes() // 2, 2, -1).sum(1)

    g1 = dgl.DGLGraph(nx.erdos_renyi_graph(32, 0.05)).to(F.ctx())
    g2 = dgl.DGLGraph(nx.erdos_renyi_graph(16, 0.2)).to(F.ctx())
    g3 = dgl.DGLGraph(nx.erdos_renyi_graph(8, 0.8)).to(F.ctx())
    net = nn.Sequential(ExampleLayer(), ExampleLayer(), ExampleLayer())
    net = net.to(ctx)
    n_feat = F.randn((32, 4))
    n_feat = net([g1, g2, g3], n_feat)
    assert n_feat.shape == (4, 4)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
def test_atomic_conv(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    aconv = nn.AtomicConv(interaction_cutoffs=F.tensor([12.0, 12.0]),
                          rbf_kernel_means=F.tensor([0.0, 2.0]),
                          rbf_kernel_scaling=F.tensor([4.0, 4.0]),
                          features_to_use=F.tensor([6.0, 8.0]))

    ctx = F.ctx()
    if F.gpu_ctx():
        aconv = aconv.to(ctx)

    feat = F.randn((g.number_of_nodes(), 1))
    dist = F.randn((g.number_of_edges(), 1))

    h = aconv(g, feat, dist)

    # current we only do shape check
    assert h.shape[-1] == 4

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo', 'bipartite'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 3])
def test_cf_conv(g, idtype, out_dim):
    g = g.astype(idtype).to(F.ctx())
    cfconv = nn.CFConv(node_in_feats=2,
                       edge_in_feats=3,
                       hidden_feats=2,
                       out_feats=out_dim)

    ctx = F.ctx()
    if F.gpu_ctx():
        cfconv = cfconv.to(ctx)

    src_feats = F.randn((g.number_of_src_nodes(), 2))
    edge_feats = F.randn((g.number_of_edges(), 3))
    h = cfconv(g, src_feats, edge_feats)
    # current we only do shape check
    assert h.shape[-1] == out_dim

    # case for bipartite graphs
    dst_feats = F.randn((g.number_of_dst_nodes(), 3))
    h = cfconv(g, (src_feats, dst_feats), edge_feats)
    # current we only do shape check
    assert h.shape[-1] == out_dim

def myagg(alist, dsttype):
    rst = alist[0]
    for i in range(1, len(alist)):
        rst = rst + (i + 1) * alist[i]
    return rst

@parametrize_idtype
@pytest.mark.parametrize('agg', ['sum', 'max', 'min', 'mean', 'stack', myagg])
def test_hetero_conv(agg, idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 0, 2, 1], [1, 2, 1, 3]),
        ('user', 'plays', 'game'): ([0, 0, 0, 1, 2], [0, 2, 3, 0, 2]),
        ('store', 'sells', 'game'): ([0, 0, 1, 1], [0, 3, 1, 2])},
        idtype=idtype, device=F.ctx())
    conv = nn.HeteroGraphConv({
        'follows': nn.GraphConv(2, 3, allow_zero_in_degree=True),
        'plays': nn.GraphConv(2, 4, allow_zero_in_degree=True),
        'sells': nn.GraphConv(3, 4, allow_zero_in_degree=True)},
        agg)
    conv = conv.to(F.ctx())

    # test pickle
    th.save(conv, tmp_buffer)

    uf = F.randn((4, 2))
    gf = F.randn((4, 4))
    sf = F.randn((2, 3))

    h = conv(g, {'user': uf, 'game': gf, 'store': sf})
    assert set(h.keys()) == {'user', 'game'}
    if agg != 'stack':
        assert h['user'].shape == (4, 3)
        assert h['game'].shape == (4, 4)
    else:
        assert h['user'].shape == (4, 1, 3)
        assert h['game'].shape == (4, 2, 4)

    block = dgl.to_block(g.to(F.cpu()), {'user': [0, 1, 2, 3], 'game': [0, 1, 2, 3], 'store': []}).to(F.ctx())
    h = conv(block, ({'user': uf, 'game': gf, 'store': sf}, {'user': uf, 'game': gf, 'store': sf[0:0]}))
    assert set(h.keys()) == {'user', 'game'}
    if agg != 'stack':
        assert h['user'].shape == (4, 3)
        assert h['game'].shape == (4, 4)
    else:
        assert h['user'].shape == (4, 1, 3)
        assert h['game'].shape == (4, 2, 4)

    h = conv(block, {'user': uf, 'game': gf, 'store': sf})
    assert set(h.keys()) == {'user', 'game'}
    if agg != 'stack':
        assert h['user'].shape == (4, 3)
        assert h['game'].shape == (4, 4)
    else:
        assert h['user'].shape == (4, 1, 3)
        assert h['game'].shape == (4, 2, 4)

    # test with mod args
    class MyMod(th.nn.Module):
        def __init__(self, s1, s2):
            super(MyMod, self).__init__()
            self.carg1 = 0
            self.carg2 = 0
            self.s1 = s1
            self.s2 = s2
        def forward(self, g, h, arg1=None, *, arg2=None):
            if arg1 is not None:
                self.carg1 += 1
            if arg2 is not None:
                self.carg2 += 1
            return th.zeros((g.number_of_dst_nodes(), self.s2))
    mod1 = MyMod(2, 3)
    mod2 = MyMod(2, 4)
    mod3 = MyMod(3, 4)
    conv = nn.HeteroGraphConv({
        'follows': mod1,
        'plays': mod2,
        'sells': mod3},
        agg)
    conv = conv.to(F.ctx())
    mod_args = {'follows' : (1,), 'plays' : (1,)}
    mod_kwargs = {'sells' : {'arg2' : 'abc'}}
    h = conv(g, {'user' : uf, 'game': gf, 'store' : sf}, mod_args=mod_args, mod_kwargs=mod_kwargs)
    assert mod1.carg1 == 1
    assert mod1.carg2 == 0
    assert mod2.carg1 == 1
    assert mod2.carg2 == 0
    assert mod3.carg1 == 0
    assert mod3.carg2 == 1

    #conv on graph without any edges
    for etype in g.etypes:
        g = dgl.remove_edges(g, g.edges(form='eid', etype=etype), etype=etype)
    assert g.num_edges() == 0
    h = conv(g, {'user': uf, 'game': gf, 'store': sf})
    assert set(h.keys()) == {'user', 'game'}

    block = dgl.to_block(g.to(F.cpu()), {'user': [0, 1, 2, 3], 'game': [
                         0, 1, 2, 3], 'store': []}).to(F.ctx())
    h = conv(block, ({'user': uf, 'game': gf, 'store': sf},
             {'user': uf, 'game': gf, 'store': sf[0:0]}))
    assert set(h.keys()) == {'user', 'game'}

@pytest.mark.parametrize('out_dim', [1, 2, 100])
def test_hetero_linear(out_dim):
    in_feats = {
        'user': F.randn((2, 1)),
        ('user', 'follows', 'user'): F.randn((3, 2))
    }

    layer = nn.HeteroLinear({'user': 1, ('user', 'follows', 'user'): 2}, out_dim)
    layer = layer.to(F.ctx())
    out_feats = layer(in_feats)
    assert out_feats['user'].shape == (2, out_dim)
    assert out_feats[('user', 'follows', 'user')].shape == (3, out_dim)

@pytest.mark.parametrize('out_dim', [1, 2, 100])
def test_hetero_embedding(out_dim):
    layer = nn.HeteroEmbedding({'user': 2, ('user', 'follows', 'user'): 3}, out_dim)
    layer = layer.to(F.ctx())

    embeds = layer.weight
    assert embeds['user'].shape == (2, out_dim)
    assert embeds[('user', 'follows', 'user')].shape == (3, out_dim)

    embeds = layer({
        'user': F.tensor([0], dtype=F.int64),
        ('user', 'follows', 'user'): F.tensor([0, 2], dtype=F.int64)
    })
    assert embeds['user'].shape == (1, out_dim)
    assert embeds[('user', 'follows', 'user')].shape == (2, out_dim)

@parametrize_idtype
@pytest.mark.parametrize('g', get_cases(['homo'], exclude=['zero-degree']))
@pytest.mark.parametrize('out_dim', [1, 2])
def test_gnnexplainer(g, idtype, out_dim):
    g = g.astype(idtype).to(F.ctx())
    feat = F.randn((g.num_nodes(), 5))

    class Model(th.nn.Module):
        def __init__(self, in_feats, out_feats, graph=False):
            super(Model, self).__init__()
            self.linear = th.nn.Linear(in_feats, out_feats)
            if graph:
                self.pool = nn.AvgPooling()
            else:
                self.pool = None

        def forward(self, graph, feat, eweight=None):
            with graph.local_scope():
                feat = self.linear(feat)
                graph.ndata['h'] = feat
                if eweight is None:
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                else:
                    graph.edata['w'] = eweight
                    graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))

                if self.pool:
                    return self.pool(graph, graph.ndata['h'])
                else:
                    return graph.ndata['h']

    # Explain node prediction
    model = Model(5, out_dim)
    model = model.to(F.ctx())
    explainer = nn.GNNExplainer(model, num_hops=1)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(0, g, feat)

    # Explain graph prediction
    model = Model(5, out_dim, graph=True)
    model = model.to(F.ctx())
    explainer = nn.GNNExplainer(model, num_hops=1)
    feat_mask, edge_mask = explainer.explain_graph(g, feat)

def test_jumping_knowledge():
    ctx = F.ctx()
    num_layers = 2
    num_nodes = 3
    num_feats = 4

    feat_list = [th.randn((num_nodes, num_feats)).to(ctx) for _ in range(num_layers)]

    model = nn.JumpingKnowledge('cat').to(ctx)
    model.reset_parameters()
    assert model(feat_list).shape == (num_nodes, num_layers * num_feats)

    model = nn.JumpingKnowledge('max').to(ctx)
    model.reset_parameters()
    assert model(feat_list).shape == (num_nodes, num_feats)

    model = nn.JumpingKnowledge('lstm', num_feats, num_layers).to(ctx)
    model.reset_parameters()
    assert model(feat_list).shape == (num_nodes, num_feats)

@pytest.mark.parametrize('op', ['dot', 'cos', 'ele', 'cat'])
def test_edge_predictor(op):
    ctx = F.ctx()
    num_pairs = 3
    in_feats = 4
    out_feats = 5
    h_src = th.randn((num_pairs, in_feats)).to(ctx)
    h_dst = th.randn((num_pairs, in_feats)).to(ctx)

    pred = nn.EdgePredictor(op)
    if op in ['dot', 'cos']:
        assert pred(h_src, h_dst).shape == (num_pairs, 1)
    elif op == 'ele':
        assert pred(h_src, h_dst).shape == (num_pairs, in_feats)
    else:
        assert pred(h_src, h_dst).shape == (num_pairs, 2 * in_feats)
    pred = nn.EdgePredictor(op, in_feats, out_feats, bias=True).to(ctx)
    assert pred(h_src, h_dst).shape == (num_pairs, out_feats)


def test_ke_score_funcs():
    ctx = F.ctx()
    num_edges = 30
    num_rels = 3
    nfeats = 4

    h_src = th.randn((num_edges, nfeats)).to(ctx)
    h_dst = th.randn((num_edges, nfeats)).to(ctx)
    rels = th.randint(low=0, high=num_rels, size=(num_edges,)).to(ctx)

    score_func = nn.TransE(num_rels=num_rels, feats=nfeats).to(ctx)
    score_func.reset_parameters()
    score_func(h_src, h_dst, rels).shape == (num_edges)

    score_func = nn.TransR(num_rels=num_rels, rfeats=nfeats - 1, nfeats=nfeats).to(ctx)
    score_func.reset_parameters()
    score_func(h_src, h_dst, rels).shape == (num_edges)


def test_twirls():
    g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    feat = th.ones(6, 10)
    conv = nn.TWIRLSConv(10, 2, 128, prop_step = 64)
    res = conv(g , feat)
    assert ( res.size() == (6,2) )

@pytest.mark.parametrize('feat_size', [4, 32])
@pytest.mark.parametrize('regularizer,num_bases', [(None, None), ('basis', 4), ('bdd', 4)])
def test_typed_linear(feat_size, regularizer, num_bases):
    dev = F.ctx()
    num_types = 5
    lin = nn.TypedLinear(feat_size, feat_size * 2, 5, regularizer=regularizer, num_bases=num_bases).to(dev)
    print(lin)
    x = th.randn(100, feat_size).to(dev)
    x_type = th.randint(0, 5, (100,)).to(dev)
    x_type_sorted, idx = th.sort(x_type)
    _, rev_idx = th.sort(idx)
    x_sorted = x[idx]

    # test unsorted
    y = lin(x, x_type)
    assert y.shape == (100, feat_size * 2)
    # test sorted
    y_sorted = lin(x_sorted, x_type_sorted, sorted_by_type=True)
    assert y_sorted.shape == (100, feat_size * 2)

    assert th.allclose(y, y_sorted[rev_idx], atol=1e-4, rtol=1e-4)

@parametrize_idtype
@pytest.mark.parametrize('in_size', [4])
@pytest.mark.parametrize('num_heads', [1])
def test_hgt(idtype, in_size, num_heads):
    dev = F.ctx()
    num_etypes = 5
    num_ntypes = 2
    head_size = in_size // num_heads

    g = dgl.from_scipy(sp.sparse.random(100, 100, density=0.01))
    g = g.astype(idtype).to(dev)
    etype = th.tensor([i % num_etypes for i in range(g.num_edges())]).to(dev)
    ntype = th.tensor([i % num_ntypes for i in range(g.num_nodes())]).to(dev)
    x = th.randn(g.num_nodes(), in_size).to(dev)

    m = nn.HGTConv(in_size, head_size, num_heads, num_ntypes, num_etypes).to(dev)

    y = m(g, x, ntype, etype)
    assert y.shape == (g.num_nodes(), head_size * num_heads)
    # presorted
    sorted_ntype, idx_nt = th.sort(ntype)
    sorted_etype, idx_et = th.sort(etype)
    _, rev_idx = th.sort(idx_nt)
    g.ndata['t'] = ntype
    g.ndata['x'] = x
    g.edata['t'] = etype
    sorted_g = dgl.reorder_graph(g, node_permute_algo='custom', edge_permute_algo='custom',
                                 permute_config={'nodes_perm' : idx_nt.to(idtype), 'edges_perm' : idx_et.to(idtype)})
    print(sorted_g.ndata['t'])
    print(sorted_g.edata['t'])
    sorted_x = sorted_g.ndata['x']
    sorted_y = m(sorted_g, sorted_x, sorted_ntype, sorted_etype, presorted=False)
    assert sorted_y.shape == (g.num_nodes(), head_size * num_heads)
    # TODO(minjie): enable the following check
    #assert th.allclose(y, sorted_y[rev_idx], atol=1e-4, rtol=1e-4)

@pytest.mark.parametrize('self_loop', [True, False])
@pytest.mark.parametrize('get_distances', [True, False])
def test_radius_graph(self_loop, get_distances):
    pos = th.tensor([[0.1, 0.3, 0.4],
                     [0.5, 0.2, 0.1],
                     [0.7, 0.9, 0.5],
                     [0.3, 0.2, 0.5],
                     [0.2, 0.8, 0.2],
                     [0.9, 0.2, 0.1],
                     [0.7, 0.4, 0.4],
                     [0.2, 0.1, 0.6],
                     [0.5, 0.3, 0.5],
                     [0.4, 0.2, 0.6]])

    rg = nn.RadiusGraph(0.3, self_loop=self_loop)

    if get_distances:
        g, dists = rg(pos, get_distances=get_distances)
    else:
        g = rg(pos)

    if self_loop:
        src_target = th.tensor([0, 0, 1, 2, 3, 3, 3, 3, 3, 4, 5, 6, 6, 7, 7, 7,
                                8, 8, 8, 8, 9, 9, 9, 9])
        dst_target = th.tensor([0, 3, 1, 2, 0, 3, 7, 8, 9, 4, 5, 6, 8, 3, 7, 9,
                                3, 6, 8, 9, 3, 7, 8, 9])

        if get_distances:
            dists_target = th.tensor([[0.0000],
                                      [0.2449],
                                      [0.0000],
                                      [0.0000],
                                      [0.2449],
                                      [0.0000],
                                      [0.1732],
                                      [0.2236],
                                      [0.1414],
                                      [0.0000],
                                      [0.0000],
                                      [0.0000],
                                      [0.2449],
                                      [0.1732],
                                      [0.0000],
                                      [0.2236],
                                      [0.2236],
                                      [0.2449],
                                      [0.0000],
                                      [0.1732],
                                      [0.1414],
                                      [0.2236],
                                      [0.1732],
                                      [0.0000]])
    else:
        src_target = th.tensor([0, 3, 3, 3, 3, 6, 7, 7, 8, 8, 8, 9, 9, 9])
        dst_target = th.tensor([3, 0, 7, 8, 9, 8, 3, 9, 3, 6, 9, 3, 7, 8])

        if get_distances:
            dists_target = th.tensor([[0.2449],
                                      [0.2449],
                                      [0.1732],
                                      [0.2236],
                                      [0.1414],
                                      [0.2449],
                                      [0.1732],
                                      [0.2236],
                                      [0.2236],
                                      [0.2449],
                                      [0.1732],
                                      [0.1414],
                                      [0.2236],
                                      [0.1732]])

    src, dst = g.edges()

    assert th.equal(src, src_target)
    assert th.equal(dst, dst_target)

    if get_distances:
        assert th.allclose(dists, dists_target, rtol=1e-03)

@parametrize_idtype
def test_group_rev_res(idtype):
    dev = F.ctx()

    num_nodes = 5
    num_edges = 20
    feats = 32
    groups = 2
    g = dgl.rand_graph(num_nodes, num_edges).to(dev)
    h = th.randn(num_nodes, feats).to(dev)
    conv = nn.GraphConv(feats // groups, feats // groups)
    model = nn.GroupRevRes(conv, groups).to(dev)
    model(g, h)

@pytest.mark.parametrize('in_size', [16, 32])
@pytest.mark.parametrize('hidden_size', [16, 32])
@pytest.mark.parametrize('out_size', [16, 32])
@pytest.mark.parametrize('edge_feat_size', [16, 10, 0])
def test_egnn_conv(in_size, hidden_size, out_size, edge_feat_size):
    dev = F.ctx()
    num_nodes = 5
    num_edges = 20
    g = dgl.rand_graph(num_nodes, num_edges).to(dev)
    h = th.randn(num_nodes, in_size).to(dev)
    x = th.randn(num_nodes, 3).to(dev)
    e = th.randn(num_edges, edge_feat_size).to(dev)
    model = nn.EGNNConv(in_size, hidden_size, out_size, edge_feat_size).to(dev)
    model(g, h, x, e)

@pytest.mark.parametrize('in_size', [16, 32])
@pytest.mark.parametrize('out_size', [16, 32])
@pytest.mark.parametrize('aggregators',
    [['mean', 'max', 'sum'], ['min', 'std', 'var'], ['moment3', 'moment4', 'moment5']])
@pytest.mark.parametrize('scalers', [['identity'], ['amplification', 'attenuation']])
@pytest.mark.parametrize('delta', [2.5, 7.4])
@pytest.mark.parametrize('dropout', [0., 0.1])
@pytest.mark.parametrize('num_towers', [1, 4])
@pytest.mark.parametrize('edge_feat_size', [16, 0])
@pytest.mark.parametrize('residual', [True, False])
def test_pna_conv(in_size, out_size, aggregators, scalers, delta,
    dropout, num_towers, edge_feat_size, residual):
    dev = F.ctx()
    num_nodes = 5
    num_edges = 20
    g = dgl.rand_graph(num_nodes, num_edges).to(dev)
    h = th.randn(num_nodes, in_size).to(dev)
    e = th.randn(num_edges, edge_feat_size).to(dev)
    model = nn.PNAConv(in_size, out_size, aggregators, scalers, delta, dropout,
        num_towers, edge_feat_size, residual).to(dev)
    model(g, h, edge_feat=e)

@pytest.mark.parametrize('k', [3, 5])
@pytest.mark.parametrize('alpha', [0., 0.5, 1.])
@pytest.mark.parametrize('norm_type', ['sym', 'row'])
@pytest.mark.parametrize('clamp', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('reset', [True, False])
def test_label_prop(k, alpha, norm_type, clamp, normalize, reset):
    dev = F.ctx()
    num_nodes = 5
    num_edges = 20
    num_classes = 4
    g = dgl.rand_graph(num_nodes, num_edges).to(dev)
    labels = th.tensor([0, 2, 1, 3, 0]).long().to(dev)
    ml_labels = th.rand(num_nodes, num_classes).to(dev) > 0.7
    mask = th.tensor([0, 1, 1, 1, 0]).bool().to(dev)
    model = nn.LabelPropagation(k, alpha, norm_type, clamp, normalize, reset)
    model(g, labels, mask)
    # multi-label case
    model(g, ml_labels, mask)

@pytest.mark.parametrize('in_size', [16, 32])
@pytest.mark.parametrize('out_size', [16, 32])
@pytest.mark.parametrize('aggregators',
    [['mean', 'max', 'dir2-av'], ['min', 'std', 'dir1-dx'], ['moment3', 'moment4', 'dir3-av']])
@pytest.mark.parametrize('scalers', [['identity'], ['amplification', 'attenuation']])
@pytest.mark.parametrize('delta', [2.5, 7.4])
@pytest.mark.parametrize('dropout', [0., 0.1])
@pytest.mark.parametrize('num_towers', [1, 4])
@pytest.mark.parametrize('edge_feat_size', [16, 0])
@pytest.mark.parametrize('residual', [True, False])
def test_dgn_conv(in_size, out_size, aggregators, scalers, delta,
    dropout, num_towers, edge_feat_size, residual):
    dev = F.ctx()
    num_nodes = 5
    num_edges = 20
    g = dgl.rand_graph(num_nodes, num_edges).to(dev)
    h = th.randn(num_nodes, in_size).to(dev)
    e = th.randn(num_edges, edge_feat_size).to(dev)
    transform = dgl.LaplacianPE(k=3, feat_name='eig')
    g = transform(g)
    eig = g.ndata['eig']
    model = nn.DGNConv(in_size, out_size, aggregators, scalers, delta, dropout,
        num_towers, edge_feat_size, residual).to(dev)
    model(g, h, edge_feat=e, eig_vec=eig)

    aggregators_non_eig = [aggr for aggr in aggregators if not aggr.startswith('dir')]
    model = nn.DGNConv(in_size, out_size, aggregators_non_eig, scalers, delta, dropout,
        num_towers, edge_feat_size, residual).to(dev)
    model(g, h, edge_feat=e)
