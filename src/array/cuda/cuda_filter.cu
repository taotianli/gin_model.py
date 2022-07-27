/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cuda/cuda_filter.cc
 * \brief Object for selecting items in a set, or selecting items not in a set.
 */

#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../filter.h"
#include "../../runtime/cuda/cuda_hashtable.cuh"
#include "./dgl_cub.cuh"

using namespace dgl::runtime::cuda;

namespace dgl {
namespace array {

namespace {

// TODO(nv-dlasalle): Replace with getting the stream from the context
// when it's implemented.
constexpr cudaStream_t cudaDefaultStream = 0;

template<typename IdType, bool include>
__global__ void _IsInKernel(
    DeviceOrderedHashTable<IdType> table,
    const IdType * const array,
    const int64_t size,
    IdType * const mark) {
  const int64_t idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx < size) {
    mark[idx] = table.Contains(array[idx]) ^ (!include);
  }
}

template<typename IdType>
__global__ void _InsertKernel(
    const IdType * const prefix,
    const int64_t size,
    IdType * const result) {
  const int64_t idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx < size) {
    if (prefix[idx] != prefix[idx+1]) {
      result[prefix[idx]] = idx;
    }
  }
}

template<typename IdType, bool include>
IdArray _PerformFilter(
    const OrderedHashTable<IdType>& table,
    IdArray test) {
  const auto& ctx = test->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t size = test->shape[0];

  if (size == 0) {
    return test;
  }

  cudaStream_t stream = cudaDefaultStream;

  // we need two arrays: 1) to act as a prefixsum
  // for the number of entries that will be inserted, and
  // 2) to collect the included items.
  IdType * prefix = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*(size+1)));

  // will resize down later
  IdArray result = aten::NewIdArray(size, ctx, sizeof(IdType)*8);

  // mark each index based on it's existence in the hashtable
  {
    const dim3 block(256);
    const dim3 grid((size+block.x-1)/block.x);

    CUDA_KERNEL_CALL((_IsInKernel<IdType, include>),
        grid, block, 0, stream,
        table.DeviceHandle(),
        static_cast<const IdType*>(test->data),
        size,
        prefix);
  }

  // generate prefix-sum
  {
    size_t workspace_bytes;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr,
        workspace_bytes,
        static_cast<IdType*>(nullptr),
        static_cast<IdType*>(nullptr),
        size+1));
    void * workspace = device->AllocWorkspace(ctx, workspace_bytes);

    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
        workspace,
        workspace_bytes,
        prefix,
        prefix,
        size+1, stream));
    device->FreeWorkspace(ctx, workspace);
  }

  // copy number
  IdType num_unique;
  device->CopyDataFromTo(prefix+size, 0,
      &num_unique, 0,
      sizeof(num_unique),
      ctx,
      DGLContext{kDLCPU, 0},
      test->dtype,
      stream);

  // insert items into set
  {
    const dim3 block(256);
    const dim3 grid((size+block.x-1)/block.x);

    CUDA_KERNEL_CALL(_InsertKernel,
        grid, block, 0, stream,
        prefix,
        size,
        static_cast<IdType*>(result->data));
  }
  device->FreeWorkspace(ctx, prefix);

  return result.CreateView({num_unique}, result->dtype);
}


template<typename IdType>
class CudaFilterSet : public Filter {
 public:
  explicit CudaFilterSet(IdArray array) :
      table_(array->shape[0], array->ctx, cudaDefaultStream) {
    table_.FillWithUnique(
        static_cast<const IdType*>(array->data),
        array->shape[0],
        cudaDefaultStream);
  }

  IdArray find_included_indices(IdArray test) override {
    return _PerformFilter<IdType, true>(table_, test);
  }

  IdArray find_excluded_indices(IdArray test) override {
    return _PerformFilter<IdType, false>(table_, test);
  }

 private:
  OrderedHashTable<IdType> table_;
};

}  // namespace

template<DLDeviceType XPU, typename IdType>
FilterRef CreateSetFilter(IdArray set) {
  return FilterRef(std::make_shared<CudaFilterSet<IdType>>(set));
}

template FilterRef CreateSetFilter<kDLGPU, int32_t>(IdArray set);
template FilterRef CreateSetFilter<kDLGPU, int64_t>(IdArray set);

}  // namespace array
}  // namespace dgl
