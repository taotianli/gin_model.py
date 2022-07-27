"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import argparse
import gc
import logging
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl.nn import RelGraphConv

# import sys
# import os
# dir_path = Path(os.path.dirname(__file__))
# sys.path.insert(0, dir_path.parent)

from .. import utils
# import utils


class EntityClassify(nn.Module):
    def __init__(self,
                 device,
                 num_nodes,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=True,
                 layer_norm=False):
        super(EntityClassify, self).__init__()
        self.device = th.device(device if device >= 0 else 'cpu')
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout, layer_norm=layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout, layer_norm=layer_norm))
        # h2o
        self.layers.append(RelGraphConv(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop,
            low_mem=self.low_mem, layer_norm=layer_norm))

    def forward(self, blocks, feats, norm=None):
        if blocks is None:
            # full graph training
            blocks = [self.g] * len(self.layers)
        h = feats
        for layer, block in zip(self.layers, blocks):
            block = block.to(self.device)
            h = layer(block, h, block.edata['etype'], block.edata['norm'])
        return h


def gen_norm(g):
    _, v, eid = g.all_edges(form='all')
    _, inverse_index, count = th.unique(
        v, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    norm = th.ones(eid.shape[0], device=eid.device) / degrees
    norm = norm.unsqueeze(1)
    g.edata['norm'] = norm


class NeighborSampler:

    def __init__(self, g, target_idx, fanouts):
        self.g = g
        self.target_idx = target_idx
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        blocks = []
        etypes = []
        norms = []
        ntypes = []
        seeds = th.tensor(seeds).long()
        cur = self.target_idx[seeds]
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur)
            gen_norm(block)
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks


@utils.thread_wrapped_func
def run(proc_id, n_gpus, n_cpus, args, devices, dataset, split, queue=None):
    from .rgcn_model import RelGraphEmbedLayer
    dev_id = devices[proc_id]
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, \
        train_idx, val_idx, test_idx, labels = dataset
    labels = labels.cuda(dev_id)
    if split is not None:
        train_seed, val_seed, test_seed = split
        train_idx = train_idx[train_seed]
        # val_idx = val_idx[val_seed]
        # test_idx = test_idx[test_seed]

    fanouts = args.fanout
    node_tids = g.ndata[dgl.NTYPE]
    sampler = NeighborSampler(g, target_idx, fanouts)
    loader = DataLoader(dataset=train_idx.numpy(),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=args.num_workers)

    world_size = n_gpus

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    backend = 'nccl'

    # using sparse embedding or usig mix_cpu_gpu model (embedding model can not be stored in GPU)
    if args.dgl_sparse is False:
        backend = 'gloo'
    print("backend using {}".format(backend))
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=dev_id)

    # node features
    # None for one-hot feature, if not none, it should be the feature tensor.
    #
    embed_layer = RelGraphEmbedLayer(dev_id,
                                     g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     node_feats,
                                     args.n_hidden,
                                     dgl_sparse=args.dgl_sparse)

    # create model
    # all model params are in device.
    model = EntityClassify(dev_id,
                           g.number_of_nodes(),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           low_mem=args.low_mem,
                           layer_norm=args.layer_norm)

    model.cuda(dev_id)
    model = DistributedDataParallel(
        model, device_ids=[dev_id], output_device=dev_id)
    if args.dgl_sparse:
        embed_layer.cuda(dev_id)
        if len(list(embed_layer.parameters())) > 0:
            embed_layer = DistributedDataParallel(
                embed_layer, device_ids=[dev_id], output_device=dev_id)
    else:
        if len(list(embed_layer.parameters())) > 0:
            embed_layer = DistributedDataParallel(
                embed_layer, device_ids=None, output_device=None)

    # optimizer
    dense_params = list(model.parameters())
    if args.node_feats:
        if n_gpus > 1:
            dense_params += list(embed_layer.module.embeds.parameters())
        else:
            dense_params += list(embed_layer.embeds.parameters())
    optimizer = th.optim.Adam(dense_params, lr=args.lr,
                              weight_decay=args.l2norm)

    if args.dgl_sparse:
        all_params = list(model.parameters()) + list(embed_layer.parameters())
        optimizer = th.optim.Adam(
            all_params, lr=args.lr, weight_decay=args.l2norm)
        if n_gpus > 1 and isinstance(embed_layer, DistributedDataParallel):
            dgl_emb = embed_layer.module.dgl_emb
        else:
            dgl_emb = embed_layer.dgl_emb
        emb_optimizer = dgl.optim.SparseAdam(
            params=dgl_emb, lr=args.sparse_lr, eps=1e-8) if len(dgl_emb) > 0 else None
    else:
        if n_gpus > 1:
            embs = list(embed_layer.module.node_embeds.parameters())
        else:
            embs = list(embed_layer.node_embeds.parameters())
        emb_optimizer = th.optim.SparseAdam(
            embs, lr=args.sparse_lr) if len(embs) > 0 else None

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    train_time = 0
    validation_time = 0
    test_time = 0
    last_val_acc = 0.0
    do_test = False
    if n_gpus > 1 and n_cpus - args.num_workers > 0:
        th.set_num_threads(n_cpus-args.num_workers)
    steps = 0
    time_records = []
    model.train()
    embed_layer.train()

    # Warm up
    for i, sample_data in enumerate(loader):
        seeds, blocks = sample_data
        t0 = time.time()
        feats = embed_layer(blocks[0].srcdata[dgl.NID],
                            blocks[0].srcdata['ntype'],
                            blocks[0].srcdata['type_id'],
                            node_feats)
        logits = model(blocks, feats)
        loss = F.cross_entropy(logits, labels[seeds])
        t1 = time.time()
        optimizer.zero_grad()
        if emb_optimizer is not None:
            emb_optimizer.zero_grad()

        loss.backward()
        if emb_optimizer is not None:
            emb_optimizer.step()
        optimizer.step()
        gc.collect()
        if i >= 3:
            break
    
    # real time
    for i, sample_data in enumerate(loader):
        seeds, blocks = sample_data
        t0 = time.time()
        feats = embed_layer(blocks[0].srcdata[dgl.NID],
                            blocks[0].srcdata['ntype'],
                            blocks[0].srcdata['type_id'],
                            node_feats)
        logits = model(blocks, feats)
        loss = F.cross_entropy(logits, labels[seeds])
        t1 = time.time()
        optimizer.zero_grad()
        if emb_optimizer is not None:
            emb_optimizer.zero_grad()

        loss.backward()
        if emb_optimizer is not None:
            emb_optimizer.step()
        optimizer.step()
        th.distributed.barrier()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        time_records.append(t2 - t0)

        gc.collect()
        if i >= 10:
            break

    if proc_id == 0:
        queue.put(np.array(time_records))


@utils.skip_if_not_4gpu()
@utils.benchmark('time', timeout=600)
@utils.parametrize('data', ['am', 'ogbn-mag'])
@utils.parametrize('low_mem', [True, False])
@utils.parametrize('dgl_sparse', [True, False])
def track_time(data, low_mem, dgl_sparse):
    # load graph data
    dataset = utils.process_data(data)
    args = config()
    devices = [0, 1, 2, 3]
    args.low_mem = low_mem
    args.dgl_sparse = dgl_sparse
    args.dataset = dataset
    ogb_dataset = False

    if data == 'am':
        args.n_bases = 40
        args.l2norm = 5e-4
    elif data == 'ogbn-mag':
        args.n_bases = 2
        args.l2norm = 0
    else:
        raise ValueError()

    if ogb_dataset is True:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]['paper']
        val_idx = split_idx["valid"]['paper']
        test_idx = split_idx["test"]['paper']
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
        labels = labels['paper'].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        if args.dataset == 'ogbn-mag':
            category = 'paper'
        print('Number of relations: {}'.format(num_rels))
        print('Number of class: {}'.format(num_classes))
        print('Number of train: {}'.format(len(train_idx)))
        print('Number of valid: {}'.format(len(val_idx)))
        print('Number of test: {}'.format(len(test_idx)))

    else:
        # Load from hetero-graph
        hg = dataset[0]

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        category = dataset.predict_category
        num_classes = dataset.num_classes
        train_mask = hg.nodes[category].data.pop('train_mask')
        test_mask = hg.nodes[category].data.pop('test_mask')
        labels = hg.nodes[category].data.pop('labels')
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

        # AIFB, MUTAG, BGS and AM datasets do not provide validation set split.
        # Split train set into train and validation if args.validation is set
        # otherwise use train set as the validation set.
        if args.validation:
            val_idx = train_idx[:len(train_idx) // 5]
            train_idx = train_idx[len(train_idx) // 5:]
        else:
            val_idx = train_idx

    node_feats = []
    for ntype in hg.ntypes:
        if len(hg.nodes[ntype].data) == 0 or args.node_feats is False:
            node_feats.append(hg.number_of_nodes(ntype))
        else:
            assert len(hg.nodes[ntype].data) == 1
            feat = hg.nodes[ntype].data.pop('feat')
            node_feats.append(feat.share_memory_())

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
        print('{}:{}'.format(i, ntype))

    g = dgl.to_homogeneous(hg)
    g.ndata['ntype'] = g.ndata[dgl.NTYPE]
    g.ndata['ntype'].share_memory_()
    g.edata['etype'] = g.edata[dgl.ETYPE]
    g.edata['etype'].share_memory_()
    g.ndata['type_id'] = g.ndata[dgl.NID]
    g.ndata['type_id'].share_memory_()
    node_ids = th.arange(g.number_of_nodes())

    # find out the target node ids
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    target_idx.share_memory_()
    train_idx.share_memory_()
    val_idx.share_memory_()
    test_idx.share_memory_()
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()

    n_gpus = len(devices)
    n_cpus = mp.cpu_count()

    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    procs = []
    num_train_seeds = train_idx.shape[0]
    num_valid_seeds = val_idx.shape[0]
    num_test_seeds = test_idx.shape[0]
    train_seeds = th.randperm(num_train_seeds)
    valid_seeds = th.randperm(num_valid_seeds)
    test_seeds = th.randperm(num_test_seeds)
    tseeds_per_proc = num_train_seeds // n_gpus
    vseeds_per_proc = num_valid_seeds // n_gpus
    tstseeds_per_proc = num_test_seeds // n_gpus

    for proc_id in range(n_gpus):
        # we have multi-gpu for training, evaluation and testing
        # so split trian set, valid set and test set into num-of-gpu parts.
        proc_train_seeds = train_seeds[proc_id * tseeds_per_proc:
                                       (proc_id + 1) * tseeds_per_proc
                                       if (proc_id + 1) * tseeds_per_proc < num_train_seeds
                                       else num_train_seeds]
        proc_valid_seeds = valid_seeds[proc_id * vseeds_per_proc:
                                       (proc_id + 1) * vseeds_per_proc
                                       if (proc_id + 1) * vseeds_per_proc < num_valid_seeds
                                       else num_valid_seeds]
        proc_test_seeds = test_seeds[proc_id * tstseeds_per_proc:
                                     (proc_id + 1) * tstseeds_per_proc
                                     if (proc_id + 1) * tstseeds_per_proc < num_test_seeds
                                     else num_test_seeds]

        p = ctx.Process(target=run, args=(proc_id, n_gpus, n_cpus // n_gpus, args, devices,
                                          (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
                                           train_idx, val_idx, test_idx, labels),
                                          (proc_train_seeds,
                                           proc_valid_seeds, proc_test_seeds),
                                          queue))
        p.start()

        procs.append(p)
    for p in procs:
        p.join()
    time_records = queue.get(block=False)
    num_exclude = 10 # exclude first 10 iterations
    if len(time_records) < 15:
        # exclude less if less records
        num_exclude = int(len(time_records)*0.3)
    return np.mean(time_records[num_exclude:])


def config():
    # parser = argparse.ArgumentParser(description='RGCN')
    args = SimpleNamespace(
        dropout=0,
        n_hidden=16,
        gpu="0,1,2,3",
        lr=1e-2,
        sparse_lr=2e-2,
        n_bases=-1,
        n_layers=2,
        dataset=None,
        l2norm=0,
        fanout=[10, 25],
        use_self_loop=True,
        batch_size=100,
        layer_norm=False,
        validation=False,
        node_feats=False,
        num_workers=0,
        dgl_sparse=False,
        low_mem=False,
    )
    # parser.add_argument("--dropout", type=float, default=0,
    #         help="dropout probability")
    # parser.add_argument("--n-hidden", type=int, default=16,
    #         help="number of hidden units")
    # parser.add_argument("--gpu", type=str, default='0',
    #         help="gpu")
    # parser.add_argument("--lr", type=float, default=1e-2,
    #         help="learning rate")
    # parser.add_argument("--sparse-lr", type=float, default=2e-2,
    #         help="sparse embedding learning rate")
    # parser.add_argument("--n-bases", type=int, default=-1,
    #         help="number of filter weight matrices, default: -1 [use all]")
    # parser.add_argument("--n-layers", type=int, default=2,
    #         help="number of propagation rounds")
    # parser.add_argument("-e", "--n-epochs", type=int, default=50,
    #         help="number of training epochs")
    # parser.add_argument("-d", "--dataset", type=str, required=True,
    #         help="dataset to use")
    # parser.add_argument("--l2norm", type=float, default=0,
    #         help="l2 norm coef")
    # parser.add_argument("--fanout", type=str, default="4, 4",
    #         help="Fan-out of neighbor sampling.")
    # parser.add_argument("--use-self-loop", default=False, action='store_true',
    #         help="include self feature as a special relation")
    # fp = parser.add_mutually_exclusive_group(required=False)
    # parser.add_argument("--batch-size", type=int, default=100,
    #         help="Mini-batch size. ")
    # parser.add_argument("--eval-batch-size", type=int, default=32,
    #         help="Mini-batch size. ")
    # parser.add_argument("--num-workers", type=int, default=0,
    #         help="Number of workers for dataloader.")
    # parser.add_argument("--low-mem", default=False, action='store_true',
    #         help="Whether use low mem RelGraphCov")
    # parser.add_argument("--dgl-sparse", default=False, action='store_true',
    #         help='Use sparse embedding for node embeddings.')
    # parser.add_argument('--node-feats', default=False, action='store_true',
    #         help='Whether use node features')
    # parser.add_argument('--layer-norm', default=False, action='store_true',
    #         help='Use layer norm')
    # parser.set_defaults(validation=True)
    # args = parser.parse_args()
    return args


if __name__ == '__main__':
    track_time('am')
