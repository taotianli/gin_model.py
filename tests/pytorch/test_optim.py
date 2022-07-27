import time
import torch.multiprocessing as mp
import unittest, os
import pytest

import torch as th
import backend as F

from dgl.nn import NodeEmbedding
from dgl.optim import SparseAdam, SparseAdagrad

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_sparse_adam():
    num_embs = 10
    emb_dim = 4
    device=F.ctx()
    dgl_emb = NodeEmbedding(num_embs, emb_dim, 'test')
    torch_emb = th.nn.Embedding(num_embs, emb_dim, sparse=True)
    th.manual_seed(0)
    th.nn.init.uniform_(torch_emb.weight, 0, 1.0)
    th.manual_seed(0)
    th.nn.init.uniform_(dgl_emb.weight, 0, 1.0)

    dgl_adam = SparseAdam(params=[dgl_emb], lr=0.01)
    torch_adam = th.optim.SparseAdam(list(torch_emb.parameters()), lr=0.01)

    # first step
    idx = th.randint(0, num_embs, size=(4,))
    dgl_value = dgl_emb(idx, device).to(th.device('cpu'))
    torch_value = torch_emb(idx)
    labels = th.ones((4,)).long()

    dgl_adam.zero_grad()
    torch_adam.zero_grad()
    dgl_loss = th.nn.functional.cross_entropy(dgl_value, labels)
    torch_loss = th.nn.functional.cross_entropy(torch_value, labels)
    dgl_loss.backward()
    torch_loss.backward()

    dgl_adam.step()
    torch_adam.step()
    assert F.allclose(dgl_emb.weight, torch_emb.weight)

    # Can not test second step
    # Pytorch sparseAdam maintains a global step
    # DGL sparseAdam use a per embedding step

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_sparse_adam_zero_step():
    num_embs = 10
    emb_dim = 4
    device=F.ctx()
    dgl_emb = NodeEmbedding(num_embs, emb_dim, 'test')
    torch_emb = th.nn.Embedding(num_embs, emb_dim, sparse=True)
    dgl_emb_zero = NodeEmbedding(num_embs, emb_dim, 'test2')
    torch_emb_zero = th.nn.Embedding(num_embs, emb_dim, sparse=True)
    th.manual_seed(0)
    th.nn.init.uniform_(torch_emb.weight, 0, 1.0)
    th.nn.init.uniform_(torch_emb_zero.weight, 0, 1.0)
    th.manual_seed(0)
    th.nn.init.uniform_(dgl_emb.weight, 0, 1.0)
    th.nn.init.uniform_(dgl_emb_zero.weight, 0, 1.0)

    dgl_adam = SparseAdam(params=[dgl_emb, dgl_emb_zero], lr=0.01)
    torch_adam = th.optim.SparseAdam(
        list(torch_emb.parameters()) + list(torch_emb_zero.parameters()), lr=0.01)

    # first step
    idx = th.randint(0, num_embs, size=(4,))
    dgl_value = dgl_emb(idx, device).to(th.device('cpu'))
    torch_value = torch_emb(idx)
    labels = th.ones((4,)).long()

    dgl_adam.zero_grad()
    torch_adam.zero_grad()
    dgl_loss = th.nn.functional.cross_entropy(dgl_value, labels)
    torch_loss = th.nn.functional.cross_entropy(torch_value, labels)
    dgl_loss.backward()
    torch_loss.backward()

    dgl_adam.step()
    torch_adam.step()
    assert F.allclose(dgl_emb.weight, torch_emb.weight)

def initializer(emb):
    th.manual_seed(0)
    emb.uniform_(-1.0, 1.0)
    return emb

def start_sparse_adam_worker(rank, device, world_size, weight, tensor_dev='cpu', has_zero_grad=False,
                             backend='gloo', num_embs=128, emb_dim=10):
    print('start sparse worker for adam {}'.format(rank))
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')

    if device.type == 'cuda':
        th.cuda.set_device(device)

    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=rank)

    init_weight = th.empty((num_embs, emb_dim))
    th.manual_seed(0)
    th.nn.init.uniform_(init_weight, -1.0, 1.0)
    dgl_emb = NodeEmbedding(num_embs, emb_dim, 'test', init_func=initializer, device=tensor_dev)
    dgl_emb.all_set_embedding(init_weight)

    if has_zero_grad:
        dgl_emb_zero = NodeEmbedding(num_embs, emb_dim, 'zero', init_func=initializer, device=tensor_dev)
        dgl_adam = SparseAdam(params=[dgl_emb, dgl_emb_zero], lr=0.01)
    else:
        dgl_adam = SparseAdam(params=[dgl_emb], lr=0.01)

    start = (num_embs // world_size) * rank
    end = (num_embs // world_size) * (rank + 1)
    th.manual_seed(rank)
    idx = th.randint(start, end, size=(4,)).to(tensor_dev)
    dgl_value = dgl_emb(idx, device)
    labels = th.ones((4,)).long().to(device)
    dgl_loss = th.nn.functional.cross_entropy(dgl_value, labels)
    dgl_adam.zero_grad()
    dgl_loss.backward()
    dgl_adam.step()
    th.distributed.barrier()
    dgl_weight = dgl_emb.all_get_embedding().detach()
    after_step = dgl_emb(idx, device).cpu()

    if rank == 0:
        dgl_value = dgl_value.detach().cpu()
        assert F.allclose(dgl_value, after_step) is False
        weight[:] = dgl_weight[:]
    th.distributed.barrier()

def start_torch_adam_worker(rank, world_size, weight, has_zero_grad=False,
                            num_embs=128, emb_dim=10):
    print('start sparse worker for adam {}'.format(rank))
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    backend='gloo'

    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=rank)

    torch_emb = th.nn.Embedding(num_embs, emb_dim, sparse=True)
    th.manual_seed(0)
    th.nn.init.uniform_(torch_emb.weight, -1.0, 1.0)
    torch_emb = th.nn.parallel.DistributedDataParallel(torch_emb)
    if has_zero_grad:
        torch_emb_zero = th.nn.Embedding(num_embs, emb_dim, sparse=True)
        torch_emb_zero = torch_emb_zero.to(tensor_dev)
        th.manual_seed(0)
        th.nn.init.uniform_(torch_emb_zero.weight, -1.0, 1.0)
        torch_emb_zero = th.nn.parallel.DistributedDataParallel(torch_emb_zero)
        torch_adam = th.optim.SparseAdam(
            list(torch_emb.module.parameters()) + list(torch_emb_zero.module.parameters()),
            lr=0.01)
    else:
        torch_adam = th.optim.SparseAdam(list(torch_emb.module.parameters()), lr=0.01)

    start = (num_embs // world_size) * rank
    end = (num_embs // world_size) * (rank + 1)
    th.manual_seed(rank)
    idx = th.randint(start, end, size=(4,))
    labels = th.ones((4,)).long()
    torch_value = torch_emb(idx)
    torch_loss = th.nn.functional.cross_entropy(torch_value, labels)
    torch_adam.zero_grad()
    torch_loss.backward()
    torch_adam.step()
    th.distributed.barrier()

    if rank == 0:
        weight[:] = torch_emb.module.weight.cpu()[:]
    th.distributed.barrier()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F.ctx().type != 'cpu', reason='cpu only test')
@pytest.mark.parametrize("num_workers", [2, 4])
def test_multiprocess_cpu_sparse_adam(num_workers):
    backend = 'gloo'
    worker_list = []
    num_embs=128
    emb_dim=10
    dgl_weight = th.empty((num_embs, emb_dim))
    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        device = F.ctx()
        p = ctx.Process(target=start_sparse_adam_worker,
                        args=(i, device, num_workers, dgl_weight, th.device('cpu'), True, backend))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    worker_list = []
    torch_weight = th.empty((num_embs, emb_dim))
    for i in range(num_workers):
        p = ctx.Process(target=start_torch_adam_worker,
                        args=(i, num_workers, torch_weight, False))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    assert F.allclose(dgl_weight, torch_weight)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F.ctx().type == 'cpu', reason='gpu only test')
@pytest.mark.parametrize("num_workers", [2, 4, 8])
@pytest.mark.parametrize("backend", ['nccl', 'gloo'])
def test_multiprocess_sparse_adam(num_workers, backend):
    if F.ctx().type == 'cuda' and th.cuda.device_count() < num_workers:
        pytest.skip("Not enough GPUs to run test.")

    worker_list = []
    num_embs=128
    emb_dim=10
    dgl_weight = th.empty((num_embs, emb_dim))
    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        device = F.ctx()
        if device.type == 'cuda':
            # make sure each process has a unique GPU
            device = th.device(i)
        p = ctx.Process(target=start_sparse_adam_worker,
                        args=(i, device, num_workers, dgl_weight, th.device('cpu'), True, backend))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    worker_list = []
    torch_weight = th.empty((num_embs, emb_dim))
    for i in range(num_workers):
        p = ctx.Process(target=start_torch_adam_worker,
                        args=(i, num_workers, torch_weight, False))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    assert F.allclose(dgl_weight, torch_weight)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F.ctx().type == 'cpu', reason='cuda tensor is not supported for cpu')
@pytest.mark.parametrize("num_workers", [2, 4, 8])
def test_multiprocess_sparse_adam_cuda_tensor(num_workers):
    if F.ctx().type == 'cpu':
        pytest.skip("Do not test CPU")
    if F.ctx().type == 'cuda' and th.cuda.device_count() < num_workers:
        pytest.skip("Not enough GPUs to run test.")

    backend = 'nccl'
    worker_list = []
    num_embs=128
    emb_dim=10
    dgl_weight = th.empty((num_embs, emb_dim))
    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        device = th.device(i)
        p = ctx.Process(target=start_sparse_adam_worker,
                        args=(i, device, num_workers, dgl_weight, device, False, backend))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    worker_list = []
    torch_weight = th.empty((num_embs, emb_dim))
    for i in range(num_workers):
        p = ctx.Process(target=start_torch_adam_worker,
                        args=(i, num_workers, torch_weight, False))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    assert F.allclose(dgl_weight, torch_weight)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F.ctx().type != 'cpu', reason='cpu only test')
@pytest.mark.parametrize("num_workers", [2, 4])
def test_multiprocess_sparse_adam_cpu_zero_step(num_workers):
    backend = 'gloo'

    worker_list = []
    num_embs=128
    emb_dim=10
    dgl_weight = th.empty((num_embs, emb_dim))
    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        device = F.ctx()
        p = ctx.Process(target=start_sparse_adam_worker,
                        args=(i, device, num_workers, dgl_weight, th.device('cpu'), True, backend))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    worker_list = []
    torch_weight = th.empty((num_embs, emb_dim))
    for i in range(num_workers):
        p = ctx.Process(target=start_torch_adam_worker,
                        args=(i, num_workers, torch_weight, False))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    assert F.allclose(dgl_weight, torch_weight)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F.ctx().type == 'cpu', reason='gpu only test')
@pytest.mark.parametrize("num_workers", [2, 4, 8])
@pytest.mark.parametrize("backend", ['nccl', 'gloo'])
def test_multiprocess_sparse_adam_zero_step(num_workers, backend):
    if F.ctx().type == 'cuda' and th.cuda.device_count() < num_workers:
        pytest.skip("Not enough GPUs to run test.")

    worker_list = []
    num_embs=128
    emb_dim=10
    dgl_weight = th.empty((num_embs, emb_dim))
    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        device = F.ctx()
        if device.type == 'cuda':
            # make sure each process has a unique GPU
            device = th.device(i)
        p = ctx.Process(target=start_sparse_adam_worker,
                        args=(i, device, num_workers, dgl_weight, th.device('cpu'), True, backend))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    worker_list = []
    torch_weight = th.empty((num_embs, emb_dim))
    for i in range(num_workers):
        p = ctx.Process(target=start_torch_adam_worker,
                        args=(i, num_workers, torch_weight, False))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    assert F.allclose(dgl_weight, torch_weight)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
@unittest.skipIf(F.ctx().type == 'cpu', reason='cuda tensor is not supported for cpu')
@pytest.mark.parametrize("num_workers", [2, 4, 8])
def test_multiprocess_sparse_adam_zero_step_cuda_tensor(num_workers):
    if F.ctx().type == 'cuda' and th.cuda.device_count() < num_workers:
        pytest.skip("Not enough GPUs to run test.")

    backend = 'nccl'
    worker_list = []
    num_embs=128
    emb_dim=10
    dgl_weight = th.empty((num_embs, emb_dim))
    ctx = mp.get_context('spawn')
    for i in range(num_workers):
        device = th.device(i)
        p = ctx.Process(target=start_sparse_adam_worker,
                        args=(i, device, num_workers, dgl_weight, device, True, backend))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    worker_list = []
    torch_weight = th.empty((num_embs, emb_dim))
    for i in range(num_workers):
        p = ctx.Process(target=start_torch_adam_worker,
                        args=(i, num_workers, torch_weight, False))
        p.start()
        worker_list.append(p)
    for p in worker_list:
        p.join()

    assert F.allclose(dgl_weight, torch_weight)

if __name__ == '__main__':
    test_sparse_adam()
    test_sparse_adam_zero_step()

    test_multiprocess_cpu_sparse_adam(2)
    test_multiprocess_cpu_sparse_adam(4)
    test_multiprocess_cpu_sparse_adam(8)
    test_multiprocess_sparse_adam_cpu_zero_step(2)

    test_multiprocess_sparse_adam(2, backend='gloo')
    test_multiprocess_sparse_adam(4, backend='gloo')
    test_multiprocess_sparse_adam(8, backend='gloo')
    test_multiprocess_sparse_adam(2, backend='nccl')
    test_multiprocess_sparse_adam(4, backend='nccl')
    test_multiprocess_sparse_adam(8, backend='nccl')

    test_multiprocess_sparse_adam_zero_step(2, backend='gloo')
    test_multiprocess_sparse_adam_zero_step(4, backend='nccl')

    test_multiprocess_sparse_adam_cuda_tensor(2)
    test_multiprocess_sparse_adam_zero_step_cuda_tensor(4)
