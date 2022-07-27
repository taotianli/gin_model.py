/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/rowwise_pick.h
 * \brief Template implementation for rowwise pick operators.
 */
#ifndef DGL_ARRAY_CPU_ROWWISE_PICK_H_
#define DGL_ARRAY_CPU_ROWWISE_PICK_H_

#include <dgl/array.h>
#include <dmlc/omp.h>
#include <dgl/runtime/parallel_for.h>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>

namespace dgl {
namespace aten {
namespace impl {

// User-defined function for picking elements from one row.
//
// The column indices of the given row are stored in
//   [col + off, col + off + len)
//
// Similarly, the data indices are stored in
//   [data + off, data + off + len)
// Data index pointer could be NULL, which means data[i] == i
//
// *ATTENTION*: This function will be invoked concurrently. Please make sure
// it is thread-safe.
//
// \param rowid The row to pick from.
// \param off Starting offset of this row.
// \param len NNZ of the row.
// \param col Pointer of the column indices.
// \param data Pointer of the data indices.
// \param out_idx Picked indices in [off, off + len).
template <typename IdxType>
using PickFn = std::function<void(
    IdxType rowid, IdxType off, IdxType len,
    const IdxType* col, const IdxType* data,
    IdxType* out_idx)>;

// User-defined function for picking elements from a range within a row.
//
// The column indices of each element is in
//   off + et_idx[et_offset+i]), where i is in [et_offset, et_offset+et_len)
//
// Similarly, the data indices are stored in
//   data[off+et_idx[et_offset+i])]
// Data index pointer could be NULL, which means data[i] == off+et_idx[et_offset+i])
//
// *ATTENTION*: This function will be invoked concurrently. Please make sure
// it is thread-safe.
//
// \param off Starting offset of this row.
// \param et_offset Starting offset of this range.
// \param cur_et The edge type.
// \param et_len Length of the range.
// \param et_idx A map from local idx to column id.
// \param data Pointer of the data indices.
// \param out_idx Picked indices in [et_offset, et_offset + et_len).
template <typename IdxType>
using RangePickFn = std::function<void(
    IdxType off, IdxType et_offset, IdxType cur_et, IdxType et_len,
    const std::vector<IdxType> &et_idx, const IdxType* data,
    IdxType* out_idx)>;

// Template for picking non-zero values row-wise. The implementation utilizes
// OpenMP parallelization on rows because each row performs computation independently.
template <typename IdxType>
COOMatrix CSRRowWisePick(CSRMatrix mat, IdArray rows,
                         int64_t num_picks, bool replace, PickFn<IdxType> pick_fn) {
  using namespace aten;
  const IdxType* indptr = static_cast<IdxType*>(mat.indptr->data);
  const IdxType* indices = static_cast<IdxType*>(mat.indices->data);
  const IdxType* data = CSRHasData(mat)? static_cast<IdxType*>(mat.data->data) : nullptr;
  const IdxType* rows_data = static_cast<IdxType*>(rows->data);
  const int64_t num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;

  // To leverage OMP parallelization, we create two arrays to store
  // picked src and dst indices. Each array is of length num_rows * num_picks.
  // For rows whose nnz < num_picks, the indices are padded with -1.
  //
  // We check whether all the given rows
  // have at least num_picks number of nnz when replace is false.
  //
  // If the check holds, remove -1 elements by remove_if operation, which simply
  // moves valid elements to the head of arrays and create a view of the original
  // array. The implementation consumes a little extra memory than the actual requirement.
  //
  // Otherwise, directly use the row and col arrays to construct the result COO matrix.
  //
  // [02/29/2020 update]: OMP is disabled for now since batch-wise parallelism is more
  //   significant. (minjie)
  IdArray picked_row = NDArray::Empty({num_rows * num_picks},
                                      DLDataType{kDLInt, 8*sizeof(IdxType), 1},
                                      ctx);
  IdArray picked_col = NDArray::Empty({num_rows * num_picks},
                                      DLDataType{kDLInt, 8*sizeof(IdxType), 1},
                                      ctx);
  IdArray picked_idx = NDArray::Empty({num_rows * num_picks},
                                      DLDataType{kDLInt, 8*sizeof(IdxType), 1},
                                      ctx);
  IdxType* picked_rdata = static_cast<IdxType*>(picked_row->data);
  IdxType* picked_cdata = static_cast<IdxType*>(picked_col->data);
  IdxType* picked_idata = static_cast<IdxType*>(picked_idx->data);

  const int num_threads = omp_get_max_threads();
  std::vector<int64_t> global_prefix(num_threads+1, 0);

#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();

    const int64_t start_i = thread_id * (num_rows/num_threads) +
        std::min(static_cast<int64_t>(thread_id), num_rows % num_threads);
    const int64_t end_i = (thread_id + 1) * (num_rows/num_threads) +
        std::min(static_cast<int64_t>(thread_id + 1), num_rows % num_threads);
    assert(thread_id + 1 < num_threads || end_i == num_rows);

    const int64_t num_local = end_i - start_i;

    // make sure we don't have to pay initialization cost
    std::unique_ptr<int64_t[]> local_prefix(new int64_t[num_local + 1]);
    local_prefix[0] = 0;
    for (int64_t i = start_i; i < end_i; ++i) {
      // build prefix-sum
      const int64_t local_i = i-start_i;
      const IdxType rid = rows_data[i];
      IdxType len;
      if (replace) {
        len = indptr[rid+1] == indptr[rid] ? 0 : num_picks;
      } else {
        len = std::min(
          static_cast<IdxType>(num_picks), indptr[rid + 1] - indptr[rid]);
      }
      local_prefix[local_i + 1] = local_prefix[local_i] + len;
    }
    global_prefix[thread_id + 1] = local_prefix[num_local];

    #pragma omp barrier
    #pragma omp master
    {
      for (int t = 0; t < num_threads; ++t) {
        global_prefix[t+1] += global_prefix[t];
      }
    }

    #pragma omp barrier
    const IdxType thread_offset = global_prefix[thread_id];

    for (int64_t i = start_i; i < end_i; ++i) {
      const IdxType rid = rows_data[i];

      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;
      if (len == 0)
        continue;

      const int64_t local_i = i - start_i;
      const int64_t row_offset = thread_offset + local_prefix[local_i];

      if (len <= num_picks && !replace) {
        // nnz <= num_picks and w/o replacement, take all nnz
        for (int64_t j = 0; j < len; ++j) {
          picked_rdata[row_offset + j] = rid;
          picked_cdata[row_offset + j] = indices[off + j];
          picked_idata[row_offset + j] = data? data[off + j] : off + j;
        }
      } else {
        pick_fn(rid, off, len,
                indices, data,
                picked_idata + row_offset);
        for (int64_t j = 0; j < num_picks; ++j) {
          const IdxType picked = picked_idata[row_offset + j];
          picked_rdata[row_offset + j] = rid;
          picked_cdata[row_offset + j] = indices[picked];
          picked_idata[row_offset + j] = data? data[picked] : picked;
        }
      }
    }
  }

  const int64_t new_len = global_prefix.back();
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(mat.num_rows, mat.num_cols,
                   picked_row, picked_col, picked_idx);
}

// Template for picking non-zero values row-wise. The implementation utilizes
// OpenMP parallelization on rows because each row performs computation independently.
template <typename IdxType>
COOMatrix CSRRowWisePerEtypePick(CSRMatrix mat, IdArray rows, IdArray etypes,
                                 const std::vector<int64_t>& num_picks, bool replace,
                                 bool etype_sorted, RangePickFn<IdxType> pick_fn) {
  using namespace aten;
  const IdxType* indptr = mat.indptr.Ptr<IdxType>();
  const IdxType* indices = mat.indices.Ptr<IdxType>();
  const IdxType* data = CSRHasData(mat)? mat.data.Ptr<IdxType>() : nullptr;
  const IdxType* rows_data = rows.Ptr<IdxType>();
  const int32_t* etype_data = etypes.Ptr<int32_t>();
  const int64_t num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;
  const int64_t num_etypes = num_picks.size();
  CHECK_EQ(etypes->dtype.bits / 8, sizeof(int32_t)) << "etypes must be int32";
  std::vector<IdArray> picked_rows(rows->shape[0]);
  std::vector<IdArray> picked_cols(rows->shape[0]);
  std::vector<IdArray> picked_idxs(rows->shape[0]);

  // Check if the number of picks have the same value.
  // If so, we can potentially speed up if we have a node with total number of neighbors
  // less than the given number of picks with replace=False.
  bool same_num_pick = true;
  int64_t num_pick_value = num_picks[0];
  for (int64_t num_pick : num_picks) {
    if (num_pick_value != num_pick) {
      same_num_pick = false;
      break;
    }
  }

  runtime::parallel_for(0, num_rows, [&](size_t b, size_t e) {
    for (int64_t i = b; i < e; ++i) {
      const IdxType rid = rows_data[i];
      CHECK_LT(rid, mat.num_rows);
      const IdxType off = indptr[rid];
      const IdxType len = indptr[rid + 1] - off;

      // do something here
      if (len == 0) {
        picked_rows[i] = NewIdArray(0, ctx, sizeof(IdxType) * 8);
        picked_cols[i] = NewIdArray(0, ctx, sizeof(IdxType) * 8);
        picked_idxs[i] = NewIdArray(0, ctx, sizeof(IdxType) * 8);
        continue;
      }

      // fast path
      if (same_num_pick && len <= num_pick_value && !replace) {
        IdArray rows = Full(rid, len, sizeof(IdxType) * 8, ctx);
        IdArray cols = Full(-1, len, sizeof(IdxType) * 8, ctx);
        IdArray idx = Full(-1, len, sizeof(IdxType) * 8, ctx);
        IdxType* cdata = cols.Ptr<IdxType>();
        IdxType* idata = idx.Ptr<IdxType>();
        for (int64_t j = 0; j < len; ++j) {
          cdata[j] = indices[off + j];
          idata[j] = data ? data[off + j] : off + j;
        }
        picked_rows[i] = rows;
        picked_cols[i] = cols;
        picked_idxs[i] = idx;
      } else {
        // need to do per edge type sample
        std::vector<IdxType> rows;
        std::vector<IdxType> cols;
        std::vector<IdxType> idx;

        std::vector<IdxType> et(len);
        std::vector<IdxType> et_idx(len);
        std::iota(et_idx.begin(), et_idx.end(), 0);
        for (int64_t j = 0; j < len; ++j) {
          et[j] = data ? etype_data[data[off+j]] : etype_data[off+j];
        }
        if (!etype_sorted)  // the edge type is sorted, not need to sort it
          std::sort(et_idx.begin(), et_idx.end(),
                    [&et](IdxType i1, IdxType i2) {return et[i1] < et[i2];});
        CHECK(et[et_idx[len - 1]] < num_etypes) <<
          "etype values exceed the number of fanouts";

        IdxType cur_et = et[et_idx[0]];
        int64_t et_offset = 0;
        int64_t et_len = 1;
        for (int64_t j = 0; j < len; ++j) {
          if ((j+1 == len) || cur_et != et[et_idx[j+1]]) {
            // 1 end of the current etype
            // 2 end of the row
            // random pick for current etype
            if (et_len <= num_picks[cur_et] && !replace) {
              // fast path, select all
              for (int64_t k = 0; k < et_len; ++k) {
                rows.push_back(rid);
                cols.push_back(indices[off+et_idx[et_offset+k]]);
                if (data)
                  idx.push_back(data[off+et_idx[et_offset+k]]);
                else
                  idx.push_back(off+et_idx[et_offset+k]);
              }
            } else {
              IdArray picked_idx = Full(-1, num_picks[cur_et], sizeof(IdxType) * 8, ctx);
              IdxType* picked_idata = static_cast<IdxType*>(picked_idx->data);

              // need call random pick
              pick_fn(off, et_offset, cur_et,
                      et_len, et_idx,
                      data, picked_idata);
              for (int64_t k = 0; k < num_picks[cur_et]; ++k) {
                const IdxType picked = picked_idata[k];
                rows.push_back(rid);
                cols.push_back(indices[off+et_idx[et_offset+picked]]);
                if (data)
                  idx.push_back(data[off+et_idx[et_offset+picked]]);
                else
                  idx.push_back(off+et_idx[et_offset+picked]);
              }
            }

            if (j+1 == len)
              break;
            // next etype
            cur_et = et[et_idx[j+1]];
            et_offset = j+1;
            et_len = 1;
          } else {
            et_len++;
          }
        }

        picked_rows[i] = VecToIdArray(rows, sizeof(IdxType) * 8, ctx);
        picked_cols[i] = VecToIdArray(cols, sizeof(IdxType) * 8, ctx);
        picked_idxs[i] = VecToIdArray(idx, sizeof(IdxType) * 8, ctx);
      }  // end processing one row

      CHECK_EQ(picked_rows[i]->shape[0], picked_cols[i]->shape[0]);
      CHECK_EQ(picked_rows[i]->shape[0], picked_idxs[i]->shape[0]);
    }  // end processing all rows
  });

  IdArray picked_row = Concat(picked_rows);
  IdArray picked_col = Concat(picked_cols);
  IdArray picked_idx = Concat(picked_idxs);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   picked_row, picked_col, picked_idx);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType>
COOMatrix COORowWisePick(COOMatrix mat, IdArray rows,
                         int64_t num_picks, bool replace, PickFn<IdxType> pick_fn) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto& picked = CSRRowWisePick<IdxType>(csr, new_rows, num_picks, replace, pick_fn);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType>
COOMatrix COORowWisePerEtypePick(COOMatrix mat, IdArray rows, IdArray etypes,
                                 const std::vector<int64_t>& num_picks, bool replace,
                                 bool etype_sorted, RangePickFn<IdxType> pick_fn) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto& picked = CSRRowWisePerEtypePick<IdxType>(
    csr, new_rows, etypes, num_picks, replace, etype_sorted, pick_fn);
  return COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_PICK_H_
