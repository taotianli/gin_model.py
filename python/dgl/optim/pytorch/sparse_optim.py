"""Node embedding optimizers"""
import abc
from abc import abstractmethod
import torch as th

from ...utils import get_shared_mem_array, create_shared_mem_array
from ...nn.pytorch import NodeEmbedding
from ...cuda import nccl
from ...partition import NDArrayPartition

class SparseGradOptimizer(abc.ABC):
    r''' The abstract sparse optimizer.

    Note: dgl sparse optimizer only work with dgl.NodeEmbedding

    Parameters
    ----------
    params : list of NodeEmbedding
        The list of NodeEmbeddings.
    lr : float
        The learning rate.
    '''
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self._rank = None
        self._world_size = None
        self._shared_cache = {}
        self._clean_grad = False
        self._opt_meta = {}
        self._comm = None
        self._first_step = True
        self._device = None
        # hold released shared memory to let other process to munmap it first
        # otherwise it will crash the training
        self.shmem_buffer_holder = []

        assert len(params) > 0, 'Empty parameters'
        # if we are using shared memory for communication
        for emb in params:
            assert isinstance(emb, NodeEmbedding), \
                'DGL SparseOptimizer only supports dgl.nn.NodeEmbedding'

            if self._rank is None:
                self._rank = emb.rank
                self._world_size = emb.world_size
            else:
                assert self._rank == emb.rank, \
                    'MultiGPU rank for each embedding should be same.'
                assert self._world_size == emb.world_size, \
                    'MultiGPU world_size for each embedding should be same.'
        assert not self._rank is None
        assert not self._world_size is None
        self._nccl_root_id = 'SparseGradOptimizer.nccl_root_id'

    def step(self):
        ''' The step function.

        The step function is invoked at the end of every batch to update embeddings
        '''
        # on the first step, check to see if the grads are on the GPU
        if self._first_step:
            for emb in self._params:
                for _, data in emb._trace:
                    if data.grad.data.device.type == 'cuda':
                        # create a communicator
                        if self._device:
                            assert self._device == data.grad.device, \
                                "All gradients must be on the same device"
                        else:
                            self._device = data.grad.device
                    else:
                        assert not self._device, \
                            "All gradients must be on the same device"

            # distributed backend use nccl
            if self._device and \
                (not th.distributed.is_initialized() or th.distributed.get_backend() == 'nccl'):
                # device is only set if the grads are on a GPU
                self._comm_setup()
            else:
                self._shared_setup()
            self.setup(self._params)
            self._first_step = False

        if self._comm:
            self._comm_step()
        else:
            self._shared_step()

    def setup(self, params):
        ''' This is function where subclasses can perform any setup they need
            to. It will be called during the first step, and communicators or
            shared memory will have been setup before this call.

            Parameters
            ----------
            params : list of NodeEmbedding
                The list of NodeEmbeddings.
        '''


    def _comm_setup(self):
        # find a store to communicate the unique id through
        if len(self._params) > 0:
            store = self._params[0].store

            if self._rank < 0:
                self._comm = nccl.Communicator(1, 0, nccl.UniqueId())
            else:
                th.cuda.set_device(self._device)
                if self._rank == 0:
                    # root process broadcasts nccl id
                    nccl_id = nccl.UniqueId()
                    uid = str(nccl_id)
                    store.set(self._nccl_root_id, uid)
                else:
                    uid = store.get(self._nccl_root_id)
                    nccl_id = nccl.UniqueId(uid)
                # needs to be set for nccl to work
                self._comm = nccl.Communicator(self._world_size,
                                               self._rank,
                                               nccl_id)
                th.distributed.barrier()

    def _shared_setup(self):
        for emb in self._params:
            emb_name = emb.name
            if self._rank == 0: # the master gpu process
                opt_meta = create_shared_mem_array(emb_name+'_opt_meta', \
                    (self._world_size, self._world_size), th.int32).zero_()

            if self._rank == 0:
                emb.store.set(emb_name+'_opt_meta', emb_name)
                self._opt_meta[emb_name] = opt_meta
            elif self._rank > 0:
                # receive
                emb.store.wait([emb_name+'_opt_meta'])
                opt_meta = get_shared_mem_array(emb_name+'_opt_meta', \
                    (self._world_size, self._world_size), th.int32)
                self._opt_meta[emb_name] = opt_meta

    def _comm_step(self):
        comm = self._comm
        with th.no_grad():
            idx_in = {}
            grad_in = {}
            for emb in self._params: # pylint: disable=too-many-nested-blocks
                emb_name = emb.name
                partition = emb.partition

                if not partition:
                    # use default partitioning
                    partition = NDArrayPartition(
                        emb.num_embeddings,
                        self._world_size if self._world_size > 0 else 1,
                        mode='remainder')

                # we need to combine gradients from multiple forward paths
                if len(emb._trace) == 0:
                    idx = th.zeros((0,), dtype=th.long, device=self._device)
                    grad = th.zeros((0, emb.embedding_dim),
                                    dtype=th.float32,
                                    device=self._device)
                elif len(emb._trace) == 1:
                    # the special case where we can use the tensors as is
                    # without any memcpy's
                    idx, grad = emb._trace[0]
                    grad = grad.grad.data
                else:
                    idx = []
                    grad = []
                    for i, data in emb._trace:
                        idx.append(i)
                        grad.append(data.grad.data)
                    idx = th.cat(idx, dim=0)
                    grad = th.cat(grad, dim=0)

                idx_in[emb_name], grad_in[emb_name] = \
                    comm.sparse_all_to_all_push(
                        idx, grad, partition=partition)
                if emb.partition:
                    # if the embedding is partitioned, map back to indexes
                    # into the local tensor
                    idx_in[emb_name] = partition.map_to_local(idx_in[emb_name])

            if self._clean_grad:
                # clean gradient track
                for emb in self._params:
                    emb.reset_trace()
                self._clean_grad = False

            for emb in self._params:
                emb_name = emb.name
                idx = idx_in[emb_name]
                grad = grad_in[emb_name]
                self.update(idx, grad, emb)


    def _shared_step(self):
        with th.no_grad():
            # Frequently alloc and free shared memory to hold intermediate tensor is expensive
            # We cache shared memory buffers in shared_emb.
            shared_emb = {emb.name: ([], []) for emb in self._params}

            # Go through all sparse embeddings
            for emb in self._params: # pylint: disable=too-many-nested-blocks
                emb_name = emb.name

                # we need to combine gradients from multiple forward paths
                idx = []
                grad = []
                for i, data in emb._trace:
                    idx.append(i)
                    grad.append(data.grad.data)
                # If the sparse embedding is not used in the previous forward step
                # The idx and grad will be empty, initialize them as empty tensors to
                # avoid crashing the optimizer step logic.
                #
                # Note: we cannot skip the gradient exchange and update steps as other
                # working processes may send gradient update requests corresponding
                # to certain embedding to this process.
                idx = th.cat(idx, dim=0) if len(idx) != 0 else \
                    th.zeros((0,), dtype=th.long, device=th.device('cpu'))
                grad = th.cat(grad, dim=0) if len(grad) != 0 else \
                    th.zeros((0, emb.embedding_dim), dtype=th.float32, device=th.device('cpu'))

                device = grad.device
                idx_dtype = idx.dtype
                grad_dtype = grad.dtype
                grad_dim = grad.shape[1]
                if self._world_size > 1:
                    if emb_name not in self._shared_cache:
                        self._shared_cache[emb_name] = {}

                    # Each training process takes the resposibility of updating a range
                    # of node embeddings, thus we can parallel the gradient update.
                    # The overall progress includes:
                    #   1. In each training process:
                    #     1.a Deciding which process a node embedding belongs to according
                    #         to the formula: process_id = node_idx mod num_of_process(N)
                    #     1.b Split the node index tensor and gradient tensor into N parts
                    #         according to step 1.
                    #     1.c Write each node index sub-tensor and gradient sub-tensor into
                    #         different DGL shared memory buffers.
                    #   2. Cross training process synchronization
                    #   3. In each traning process:
                    #     3.a Collect node index sub-tensors and gradient sub-tensors
                    #     3.b Do gradient update
                    #   4. Done
                    idx_split = th.remainder(idx, self._world_size).long()
                    for i in range(self._world_size):
                        mask = idx_split == i
                        idx_i = idx[mask]
                        grad_i = grad[mask]

                        if i == self._rank:
                            shared_emb[emb_name][0].append(idx_i)
                            shared_emb[emb_name][1].append(grad_i)
                        else:
                            # currently nccl does not support Alltoallv operation
                            # we need to use CPU shared memory to share gradient
                            # across processes
                            idx_i = idx_i.to(th.device('cpu'))
                            grad_i = grad_i.to(th.device('cpu'))
                            idx_shmem_name = 'idx_{}_{}_{}'.format(emb_name, self._rank, i)
                            grad_shmem_name = 'grad_{}_{}_{}'.format(emb_name, self._rank, i)

                            # Create shared memory to hold temporary index and gradient tensor for
                            # cross-process send and recv.
                            if idx_shmem_name not in self._shared_cache[emb_name] or \
                                self._shared_cache[emb_name][idx_shmem_name].shape[0] \
                                    < idx_i.shape[0]:

                                if idx_shmem_name in self._shared_cache[emb_name]:
                                    self.shmem_buffer_holder.append(
                                        self._shared_cache[emb_name][idx_shmem_name])
                                    self.shmem_buffer_holder.append(
                                        self._shared_cache[emb_name][grad_shmem_name])

                                # The total number of buffers is the number of NodeEmbeddings *
                                # world_size * (world_size - 1). The minimun buffer size is 128.
                                #
                                # We extend the buffer by idx_i.shape[0] * 2 to avoid
                                # frequent shared memory allocation.
                                # The overall buffer cost will be smaller than three times
                                # the maximum memory requirement for sharing gradients.
                                buffer_size = 128 if idx_i.shape[0] < 128 else idx_i.shape[0] * 2
                                idx_shmem = create_shared_mem_array(
                                    '{}_{}'.format(idx_shmem_name, buffer_size), \
                                    (buffer_size,), idx_dtype)
                                grad_shmem = create_shared_mem_array(
                                    '{}_{}'.format(grad_shmem_name, buffer_size), \
                                    (buffer_size, grad_dim), grad_dtype)
                                self._shared_cache[emb_name][idx_shmem_name] = idx_shmem
                                self._shared_cache[emb_name][grad_shmem_name] = grad_shmem

                            # Fill shared memory with temporal index tensor and gradient tensor
                            self._shared_cache[emb_name][idx_shmem_name][:idx_i.shape[0]] \
                                = idx_i
                            self._shared_cache[emb_name][grad_shmem_name][:idx_i.shape[0]] \
                                = grad_i
                            self._opt_meta[emb_name][self._rank][i] = idx_i.shape[0]
                else:
                    shared_emb[emb_name][0].append(idx)
                    shared_emb[emb_name][1].append(grad)

            # make sure the idx shape is passed to each process through opt_meta
            if self._world_size > 1:
                th.distributed.barrier()
            for emb in self._params: # pylint: disable=too-many-nested-blocks
                emb_name = emb.name
                if self._world_size > 1:
                    # The first element in shared_emb[emb_name][0] is the local idx
                    device = shared_emb[emb_name][0][0].device
                    # gather gradients from all other processes
                    for i in range(self._world_size):
                        if i != self._rank:
                            idx_shmem_name = 'idx_{}_{}_{}'.format(emb_name, i, self._rank)
                            grad_shmem_name = 'grad_{}_{}_{}'.format(emb_name, i, self._rank)
                            size = self._opt_meta[emb_name][i][self._rank]

                            # Retrive shared memory holding the temporal index and gradient
                            # tensor that is sent to current training process
                            if idx_shmem_name not in self._shared_cache[emb_name] or \
                                self._shared_cache[emb_name][idx_shmem_name].shape[0] < size:
                                buffer_size = 128 if size < 128 else size * 2
                                idx_shmem = get_shared_mem_array(
                                    '{}_{}'.format(idx_shmem_name, buffer_size), \
                                    (buffer_size,), idx_dtype)
                                grad_shmem = get_shared_mem_array(
                                    '{}_{}'.format(grad_shmem_name, buffer_size), \
                                    (buffer_size, grad_dim), grad_dtype)
                                self._shared_cache[emb_name][idx_shmem_name] = idx_shmem
                                self._shared_cache[emb_name][grad_shmem_name] = grad_shmem

                            idx_i = self._shared_cache[emb_name][idx_shmem_name][:size]
                            grad_i = self._shared_cache[emb_name][grad_shmem_name][:size]
                            shared_emb[emb_name][0].append(idx_i.to(device,
                                                                    non_blocking=True))
                            shared_emb[emb_name][1].append(grad_i.to(device,
                                                                     non_blocking=True))

            if self._clean_grad:
                # clean gradient track
                for emb in self._params:
                    emb.reset_trace()
                self._clean_grad = False

            for emb in self._params:
                emb_name = emb.name

                idx = th.cat(shared_emb[emb_name][0], dim=0)
                grad = th.cat(shared_emb[emb_name][1], dim=0)
                self.update(idx, grad, emb)

            # synchronized gradient update
            if self._world_size > 1:
                th.distributed.barrier()

    @abstractmethod
    def update(self, idx, grad, emb):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.nn.NodeEmbedding
            Sparse node embedding to update.
        """

    def zero_grad(self):
        """clean grad cache
        """
        self._clean_grad = True

class SparseAdagrad(SparseGradOptimizer):
    r''' Node embedding optimizer using the Adagrad algorithm.

    This optimizer implements a sparse version of Adagrad algorithm for
    optimizing :class:`dgl.nn.NodeEmbedding`. Being sparse means it only updates
    the embeddings whose gradients have updates, which are usually a very
    small portion of the total embeddings.

    Adagrad maintains a :math:`G_{t,i,j}` for every parameter in the embeddings, where
    :math:`G_{t,i,j}=G_{t-1,i,j} + g_{t,i,j}^2` and :math:`g_{t,i,j}` is the gradient of
    the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    NOTE: The support of sparse Adagrad optimizer is experimental.

    Parameters
    ----------
    params : list[dgl.nn.NodeEmbedding]
        The list of dgl.nn.NodeEmbedding.
    lr : float
        The learning rate.
    eps : float, Optional
        The term added to the denominator to improve numerical stability
        Default: 1e-10

    Examples
    --------
    >>> def initializer(emb):
            th.nn.init.xavier_uniform_(emb)
            return emb
    >>> emb = dgl.nn.NodeEmbedding(g.number_of_nodes(), 10, 'emb', init_func=initializer)
    >>> optimizer = dgl.optim.SparseAdagrad([emb], lr=0.001)
    >>> for blocks in dataloader:
    ...     ...
    ...     feats = emb(nids, gpu_0)
    ...     loss = F.sum(feats + 1, 0)
    ...     loss.backward()
    ...     optimizer.step()
    '''
    def __init__(self, params, lr, eps=1e-10):
        super(SparseAdagrad, self).__init__(params, lr)
        self._eps = eps

    def setup(self, params):
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(emb, NodeEmbedding), \
                'SparseAdagrad only supports dgl.nn.NodeEmbedding'

            emb_name = emb.name
            if th.device(emb.emb_tensor.device) == th.device('cpu'):
                # if our embedding is on the CPU, our state also has to be
                if self._rank < 0:
                    state = th.empty(
                        emb.weight.shape,
                        dtype=th.float32,
                        device=eth.device('cpu')).zero_()
                elif self._rank == 0:
                    state = create_shared_mem_array(emb_name+'_state', \
                        emb.weight.shape, th.float32).zero_()

                    if self._world_size > 1:
                        emb.store.set(emb_name+'_opt', emb_name)
                elif self._rank > 0:
                    # receive
                    emb.store.wait([emb_name+'_opt'])
                    state = get_shared_mem_array(emb_name+'_state', \
                        emb.weight.shape, th.float32)
            else:
                # distributed state on on gpu
                state = th.empty(
                    emb.emb_tensor.shape,
                    dtype=th.float32,
                    device=emb.emb_tensor.device).zero_()
            emb.set_optm_state(state)

    def update(self, idx, grad, emb):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.nn.NodeEmbedding
            Sparse embedding to update.
        """
        eps = self._eps
        clr = self._lr

        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(idx, return_inverse=True, return_counts=True)
        grad_values = th.zeros((grad_indices.shape[0], grad.shape[1]), device=grad.device)
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)

        grad_sum = (grad_values * grad_values)
        state = emb.optm_state
        state_dev = state.device
        state_idx = grad_indices.to(state_dev)
        grad_state = state[state_idx].to(grad.device)
        grad_state += grad_sum
        state[state_idx] = grad_state.to(state_dev)

        std_values = grad_state.add_(eps).sqrt_()
        tmp = clr * grad_values / std_values
        emb.weight[state_idx] -= tmp.to(state_dev)

class SparseAdam(SparseGradOptimizer):
    r''' Node embedding optimizer using the Adam algorithm.

    This optimizer implements a sparse version of Adagrad algorithm for
    optimizing :class:`dgl.nn.NodeEmbedding`. Being sparse means it only
    updates the embeddings whose gradients have updates, which are usually
    a very small portion of the total embeddings.

    Adam maintains a :math:`Gm_{t,i,j}` and `Gp_{t,i,j}` for every parameter
    in the embeddings, where
    :math:`Gm_{t,i,j}=beta1 * Gm_{t-1,i,j} + (1-beta1) * g_{t,i,j}`,
    :math:`Gp_{t,i,j}=beta2 * Gp_{t-1,i,j} + (1-beta2) * g_{t,i,j}^2`,
    :math:`g_{t,i,j} = lr * Gm_{t,i,j} / (1 - beta1^t) / \sqrt{Gp_{t,i,j} / (1 - beta2^t)}` and
    :math:`g_{t,i,j}` is the gradient of the dimension :math:`j` of embedding :math:`i`
    at step :math:`t`.

    NOTE: The support of sparse Adam optimizer is experimental.

    Parameters
    ----------
    params : list[dgl.nn.NodeEmbedding]
        The list of dgl.nn.NodeEmbeddings.
    lr : float
        The learning rate.
    betas : tuple[float, float], Optional
        Coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.999)
    eps : float, Optional
        The term added to the denominator to improve numerical stability
        Default: 1e-8

    Examples
    --------
    >>> def initializer(emb):
            th.nn.init.xavier_uniform_(emb)
            return emb
    >>> emb = dgl.nn.NodeEmbedding(g.number_of_nodes(), 10, 'emb', init_func=initializer)
    >>> optimizer = dgl.optim.SparseAdam([emb], lr=0.001)
    >>> for blocks in dataloader:
    ...     ...
    ...     feats = emb(nids, gpu_0)
    ...     loss = F.sum(feats + 1, 0)
    ...     loss.backward()
    ...     optimizer.step()
    '''
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-08):
        super(SparseAdam, self).__init__(params, lr)
        self._lr = lr
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._eps = eps

    def setup(self, params):
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(emb, NodeEmbedding), \
                'SparseAdam only supports dgl.nn.NodeEmbedding'
            emb_name = emb.name
            if th.device(emb.emb_tensor.device) == th.device('cpu'):
                # if our embedding is on the CPU, our state also has to be
                if self._rank < 0:
                    state_step = th.empty(
                        (emb.weight.shape[0],),
                        dtype=th.float32,
                        device=th.device('cpu')).zero_()
                    state_mem = th.empty(
                        emb.weight.shape,
                        dtype=th.float32,
                        device=th.device('cpu')).zero_()
                    state_power = th.empty(
                        emb.weight.shape,
                        dtype=th.float32,
                        device=th.device('cpu')).zero_()
                elif self._rank == 0:
                    state_step = create_shared_mem_array(emb_name+'_step', \
                        (emb.weight.shape[0],), th.float32).zero_()
                    state_mem = create_shared_mem_array(emb_name+'_mem', \
                        emb.weight.shape, th.float32).zero_()
                    state_power = create_shared_mem_array(emb_name+'_power', \
                        emb.weight.shape, th.float32).zero_()

                    if self._world_size > 1:
                        emb.store.set(emb_name+'_opt', emb_name)
                elif self._rank > 0:
                    # receive
                    emb.store.wait([emb_name+'_opt'])
                    state_step = get_shared_mem_array(emb_name+'_step', \
                        (emb.weight.shape[0],), th.float32)
                    state_mem = get_shared_mem_array(emb_name+'_mem', \
                        emb.weight.shape, th.float32)
                    state_power = get_shared_mem_array(emb_name+'_power', \
                        emb.weight.shape, th.float32)
            else:
                # distributed state on on gpu
                state_step = th.empty(
                    [emb.emb_tensor.shape[0]],
                    dtype=th.float32,
                    device=emb.emb_tensor.device).zero_()
                state_mem = th.empty(
                    emb.emb_tensor.shape,
                    dtype=th.float32,
                    device=emb.emb_tensor.device).zero_()
                state_power = th.empty(
                    emb.emb_tensor.shape,
                    dtype=th.float32,
                    device=emb.emb_tensor.device).zero_()
            state = (state_step, state_mem, state_power)
            emb.set_optm_state(state)

    def update(self, idx, grad, emb):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.nn.NodeEmbedding
            Sparse embedding to update.
        """
        with th.no_grad():
            beta1 = self._beta1
            beta2 = self._beta2
            eps = self._eps

            clr = self._lr
            state_step, state_mem, state_power = emb.optm_state
            exec_dev = grad.device
            state_dev = state_step.device

            # only perform async copies cpu -> gpu, or gpu-> gpu, but block
            # when copying to the cpu, so as to ensure the copy is finished
            # before operating on the data on the cpu
            state_block = state_dev == th.device('cpu') and exec_dev != state_dev

            # There can be duplicated indices due to sampling.
            # Thus unique them here and average the gradient here.
            grad_indices, inverse, cnt = th.unique(idx,
                                                   return_inverse=True,
                                                   return_counts=True)
            state_idx = grad_indices.to(state_dev)
            state_step[state_idx] += 1
            state_step = state_step[state_idx].to(exec_dev)
            orig_mem = state_mem[state_idx].to(exec_dev)
            orig_power = state_power[state_idx].to(exec_dev)

            grad_values = th.zeros((grad_indices.shape[0], grad.shape[1]), device=exec_dev)
            grad_values.index_add_(0, inverse, grad)
            grad_values = grad_values / cnt.unsqueeze(1)

            grad_mem = grad_values
            grad_power = grad_values * grad_values

            update_mem = beta1 * orig_mem + (1.-beta1) * grad_mem
            update_power = beta2 * orig_power + (1.-beta2) * grad_power
            update_mem_dst = update_mem.to(state_dev, non_blocking=True)
            update_power_dst = update_power.to(state_dev, non_blocking=True)
            if state_block:
                # use events to try and overlap CPU and GPU as much as possible
                update_event = th.cuda.Event()
                update_event.record()

            update_mem_corr = update_mem / (1. - th.pow(th.tensor(beta1, device=exec_dev),
                                                        state_step)).unsqueeze(1)
            update_power_corr = update_power / (1. - th.pow(th.tensor(beta2, device=exec_dev),
                                                            state_step)).unsqueeze(1)
            std_values = clr * update_mem_corr / (th.sqrt(update_power_corr) + eps)
            std_values_dst = std_values.to(state_dev, non_blocking=True)

            if state_block:
                std_event = th.cuda.Event()
                std_event.record()
                # wait for our transfers from exec_dev to state_dev to finish
                # before we can use them
                update_event.wait()
            state_mem[state_idx] = update_mem_dst
            state_power[state_idx] = update_power_dst

            if state_block:
                # wait for the transfer of std_values to finish before we
                # can use it
                std_event.wait()
            emb.weight[state_idx] -= std_values_dst
