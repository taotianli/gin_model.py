/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/utils.h
 * \brief Utilities for CUDA kernels.
 */
#ifndef DGL_ARRAY_CUDA_UTILS_H_
#define DGL_ARRAY_CUDA_UTILS_H_

#include <dmlc/logging.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/ndarray.h>
#include <dlpack/dlpack.h>
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace cuda {

#define CUDA_MAX_NUM_BLOCKS_X 0x7FFFFFFF
#define CUDA_MAX_NUM_BLOCKS_Y 0xFFFF
#define CUDA_MAX_NUM_BLOCKS_Z 0xFFFF
#define CUDA_MAX_NUM_THREADS 1024

#ifdef USE_FP16
#define SWITCH_BITS(bits, DType, ...)                           \
  do {                                                          \
    if ((bits) == 16) {                                         \
      typedef half DType;                                       \
      { __VA_ARGS__ }                                           \
    } else if ((bits) == 32) {                                  \
      typedef float DType;                                      \
      { __VA_ARGS__ }                                           \
    } else if ((bits) == 64) {                                  \
      typedef double DType;                                     \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Data type not recognized with bits " << bits; \
    }                                                           \
  } while (0)
#else  // USE_FP16
#define SWITCH_BITS(bits, DType, ...)                           \
  do {                                                          \
    if ((bits) == 32) {                                         \
      typedef float DType;                                      \
      { __VA_ARGS__ }                                           \
    } else if ((bits) == 64) {                                  \
      typedef double DType;                                     \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Data type not recognized with bits " << bits; \
    }                                                           \
  } while (0)
#endif  // USE_FP16

/*! \brief Calculate the number of threads needed given the dimension length.
 *
 * It finds the biggest number that is smaller than min(dim, max_nthrs)
 * and is also power of two.
 */
inline int FindNumThreads(int dim, int max_nthrs = CUDA_MAX_NUM_THREADS) {
  CHECK_GE(dim, 0);
  if (dim == 0)
    return 1;
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

/*
 * !\brief Find number of blocks is smaller than nblks and max_nblks
 * on the given axis ('x', 'y' or 'z').
 */
template <char axis>
inline int FindNumBlocks(int nblks, int max_nblks = -1) {
  int default_max_nblks = -1;
  switch (axis) {
    case 'x':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_X;
      break;
    case 'y':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_Y;
      break;
    case 'z':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_Z;
      break;
    default:
      LOG(FATAL) << "Axis " << axis << " not recognized";
      break;
  }
  if (max_nblks == -1)
    max_nblks = default_max_nblks;
  CHECK_NE(nblks, 0);
  if (nblks < max_nblks)
    return nblks;
  return max_nblks;
}

template <typename T>
__device__ __forceinline__ T _ldg(T* addr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(addr);
#else
  return *addr;
#endif
}

/*!
 * \brief Return true if the given bool flag array is all true.
 * The input bool array is in int8_t type so it is aligned with byte address.
 *
 * \param flags The bool array.
 * \param length The length.
 * \param ctx Device context.
 * \return True if all the flags are true.
 */
bool AllTrue(int8_t* flags, int64_t length, const DLContext& ctx);

/*!
 * \brief CUDA Kernel of filling the vector started from ptr of size length
 *        with val.
 * \note internal use only.
 */
template <typename DType>
__global__ void _FillKernel(DType* ptr, size_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = val;
    tx += stride_x;
  }
}

/*! \brief Fill the vector started from ptr of size length with val */
template <typename DType>
void _Fill(DType* ptr, size_t length, DType val) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = FindNumThreads(length);
  int nb = (length + nt - 1) / nt;  // on x-axis, no need to worry about upperbound.
  CUDA_KERNEL_CALL(cuda::_FillKernel, nb, nt, 0, thr_entry->stream, ptr, length, val);
}

/*!
 * \brief Search adjacency list linearly for each (row, col) pair and
 * write the data under the matched position in the indices array to the output.
 *
 * If there is no match, the value in \c filler is written.
 * If there are multiple matches, only the first match is written.
 * If the given data array is null, write the matched position to the output.
 */
template <typename IdType, typename DType>
__global__ void _LinearSearchKernel(
    const IdType* indptr, const IdType* indices, const IdType* data,
    const IdType* row, const IdType* col,
    int64_t row_stride, int64_t col_stride,
    int64_t length, const DType* weights, DType filler, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    int rpos = tx * row_stride, cpos = tx * col_stride;
    IdType v = -1;
    const IdType r = row[rpos], c = col[cpos];
    for (IdType i = indptr[r]; i < indptr[r + 1]; ++i) {
      if (indices[i] == c) {
        v = data ? data[i] : i;
        break;
      }
    }
    if (v == -1)
      out[tx] = filler;
    else
      out[tx] = weights ? weights[v] : v;
    tx += stride_x;
  }
}

template <typename DType>
inline DType GetCUDAScalar(
    runtime::DeviceAPI* device_api,
    DLContext ctx,
    const DType* cuda_ptr,
    cudaStream_t stream) {
  DType result;
  device_api->CopyDataFromTo(
      cuda_ptr, 0,
      &result, 0,
      sizeof(result),
      ctx,
      DLContext{kDLCPU, 0},
      DLDataTypeTraits<DType>::dtype,
      stream);
  return result;
}

/*!
 * \brief Given a sorted array and a value this function returns the index
 * of the first element which compares greater than value.
 *
 * This function assumes 0-based index
 * @param A: ascending sorted array
 * @param n: size of the A
 * @param x: value to search in A
 * @return index, i, of the first element st. A[i]>x. If x>=A[n-1] returns n.
 * if x<A[0] then it returns 0.
 */
template <typename IdType>
__device__ IdType _UpperBound(const IdType *A, int64_t n, IdType x) {
  IdType l = 0, r = n, m = 0;
  while (l < r) {
    m = l + (r-l)/2;
    if (x >= A[m]) {
      l = m+1;
    } else {
      r = m;
    }
  }
  return l;
}

/*!
 * \brief Given a sorted array and a value this function returns the index
 * of the element who is equal to val. If not exist returns n+1
 *
 * This function assumes 0-based index
 * @param A: ascending sorted array
 * @param n: size of the A
 * @param x: value to search in A
 * @return index, i, st. A[i]==x. If such an index not exists returns 'n'.
 */
template <typename IdType>
__device__ IdType _BinarySearch(const IdType *A, int64_t n, IdType x) {
  IdType l = 0, r = n-1, m = 0;
  while (l <= r) {
    m = l + (r-l)/2;
    if (A[m] == x) {
      return m;
    }
    if (A[m] < x) {
      l = m+1;
    } else {
      r = m-1;
    }
  }
  return n;  // not found
}

}  // namespace cuda
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_UTILS_H_
