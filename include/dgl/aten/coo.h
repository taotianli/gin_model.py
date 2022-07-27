
/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/aten/coo.h
 * \brief Common COO operations required by DGL.
 */
#ifndef DGL_ATEN_COO_H_
#define DGL_ATEN_COO_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include <vector>
#include <utility>
#include <tuple>
#include <string>
#include "./types.h"
#include "./array_ops.h"
#include "./spmat.h"
#include "./macro.h"

namespace dgl {
namespace aten {

struct CSRMatrix;

/*!
 * \brief Plain COO structure
 *
 * The data array stores integer ids for reading edge features.
 * Note that we do allow duplicate non-zero entries -- multiple non-zero entries
 * that have the same row, col indices. It corresponds to multigraph in
 * graph terminology.
 */

constexpr uint64_t kDGLSerialize_AtenCooMatrixMagic = 0xDD61ffd305dff127;

// TODO(BarclayII): Graph queries on COO formats should support the case where
// data ordered by rows/columns instead of EID.
struct COOMatrix {
  /*! \brief the dense shape of the matrix */
  int64_t num_rows = 0, num_cols = 0;
  /*! \brief COO index arrays */
  IdArray row, col;
  /*! \brief data index array. When is null, assume it is from 0 to NNZ - 1. */
  IdArray data;
  /*! \brief whether the row indices are sorted */
  bool row_sorted = false;
  /*! \brief whether the column indices per row are sorted */
  bool col_sorted = false;
  /*! \brief whether the matrix is in pinned memory */
  bool is_pinned = false;
  /*! \brief default constructor */
  COOMatrix() = default;
  /*! \brief constructor */
  COOMatrix(int64_t nrows, int64_t ncols, IdArray rarr, IdArray carr,
            IdArray darr = NullArray(), bool rsorted = false,
            bool csorted = false)
      : num_rows(nrows),
        num_cols(ncols),
        row(rarr),
        col(carr),
        data(darr),
        row_sorted(rsorted),
        col_sorted(csorted) {
    CheckValidity();
  }

  /*! \brief constructor from SparseMatrix object */
  explicit COOMatrix(const SparseMatrix& spmat)
      : num_rows(spmat.num_rows),
        num_cols(spmat.num_cols),
        row(spmat.indices[0]),
        col(spmat.indices[1]),
        data(spmat.indices[2]),
        row_sorted(spmat.flags[0]),
        col_sorted(spmat.flags[1]) {
    CheckValidity();
  }

  // Convert to a SparseMatrix object that can return to python.
  SparseMatrix ToSparseMatrix() const {
    return SparseMatrix(static_cast<int32_t>(SparseFormat::kCOO), num_rows,
                        num_cols, {row, col, data}, {row_sorted, col_sorted});
  }

  bool Load(dmlc::Stream* fs) {
    uint64_t magicNum;
    CHECK(fs->Read(&magicNum)) << "Invalid Magic Number";
    CHECK_EQ(magicNum, kDGLSerialize_AtenCooMatrixMagic)
        << "Invalid COOMatrix Data";
    CHECK(fs->Read(&num_cols)) << "Invalid num_cols";
    CHECK(fs->Read(&num_rows)) << "Invalid num_rows";
    CHECK(fs->Read(&row)) << "Invalid row";
    CHECK(fs->Read(&col)) << "Invalid col";
    CHECK(fs->Read(&data)) << "Invalid data";
    CHECK(fs->Read(&row_sorted)) << "Invalid row_sorted";
    CHECK(fs->Read(&col_sorted)) << "Invalid col_sorted";
    CheckValidity();
    return true;
  }

  void Save(dmlc::Stream* fs) const {
    fs->Write(kDGLSerialize_AtenCooMatrixMagic);
    fs->Write(num_cols);
    fs->Write(num_rows);
    fs->Write(row);
    fs->Write(col);
    fs->Write(data);
    fs->Write(row_sorted);
    fs->Write(col_sorted);
  }

  inline void CheckValidity() const {
    CHECK_SAME_DTYPE(row, col);
    CHECK_SAME_CONTEXT(row, col);
    if (!aten::IsNullArray(data)) {
      CHECK_SAME_DTYPE(row, data);
      CHECK_SAME_CONTEXT(row, data);
    }
    CHECK_NO_OVERFLOW(row->dtype, num_rows);
    CHECK_NO_OVERFLOW(row->dtype, num_cols);
  }

  /*! \brief Return a copy of this matrix on the give device context. */
  inline COOMatrix CopyTo(const DLContext &ctx,
                          const DGLStreamHandle &stream = nullptr) const {
    if (ctx == row->ctx)
      return *this;
    return COOMatrix(num_rows, num_cols, row.CopyTo(ctx, stream),
                     col.CopyTo(ctx, stream),
                     aten::IsNullArray(data) ? data : data.CopyTo(ctx, stream),
                     row_sorted, col_sorted);
  }

  /*!
  * \brief Pin the row, col and data (if not Null) of the matrix.
  * \note This is an in-place method. Behavior depends on the current context,
  *       kDLCPU: will be pinned;
  *       IsPinned: directly return;
  *       kDLGPU: invalid, will throw an error.
  *       The context check is deferred to pinning the NDArray.
  */
  inline void PinMemory_() {
    if (is_pinned)
      return;
    row.PinMemory_();
    col.PinMemory_();
    if (!aten::IsNullArray(data)) {
      data.PinMemory_();
    }
    is_pinned = true;
  }

  /*!
  * \brief Unpin the row, col and data (if not Null) of the matrix.
  * \note This is an in-place method. Behavior depends on the current context,
  *       IsPinned: will be unpinned;
  *       others: directly return.
  *       The context check is deferred to unpinning the NDArray.
  */
  inline void UnpinMemory_() {
    if (!is_pinned)
      return;
    row.UnpinMemory_();
    col.UnpinMemory_();
    if (!aten::IsNullArray(data)) {
      data.UnpinMemory_();
    }
    is_pinned = false;
  }
};

///////////////////////// COO routines //////////////////////////

/*! \brief Return true if the value (row, col) is non-zero */
bool COOIsNonZero(COOMatrix , int64_t row, int64_t col);
/*!
 * \brief Batched implementation of COOIsNonZero.
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 */
runtime::NDArray COOIsNonZero(COOMatrix , runtime::NDArray row, runtime::NDArray col);

/*! \brief Return the nnz of the given row */
int64_t COOGetRowNNZ(COOMatrix , int64_t row);
runtime::NDArray COOGetRowNNZ(COOMatrix , runtime::NDArray row);

/*! \brief Return the data array of the given row */
std::pair<runtime::NDArray, runtime::NDArray>
COOGetRowDataAndIndices(COOMatrix , int64_t row);

/*! \brief Whether the COO matrix contains data */
inline bool COOHasData(COOMatrix csr) {
  return !IsNullArray(csr.data);
}

/*!
 * \brief Check whether the COO is sorted.
 *
 * It returns two flags: one for whether the row is sorted;
 * the other for whether the columns of each row is sorted
 * if the first flag is true.
 *
 * Complexity: O(NNZ)
 */
std::pair<bool, bool> COOIsSorted(COOMatrix coo);

/*!
 * \brief Get the data and the row,col indices for each returned entries.
 *
 * The operator supports matrix with duplicate entries and all the matched entries
 * will be returned. The operator assumes there is NO duplicate (row, col) pair
 * in the given input. Otherwise, the returned result is undefined.
 *
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 * \param mat Sparse matrix
 * \param rows Row index
 * \param cols Column index
 * \return Three arrays {rows, cols, data}
 */
std::vector<runtime::NDArray> COOGetDataAndIndices(
    COOMatrix mat, runtime::NDArray rows, runtime::NDArray cols);

/*! \brief Get data. The return type is an ndarray due to possible duplicate entries. */
inline runtime::NDArray COOGetAllData(COOMatrix mat, int64_t row, int64_t col) {
  IdArray rows = VecToIdArray<int64_t>({row}, mat.row->dtype.bits, mat.row->ctx);
  IdArray cols = VecToIdArray<int64_t>({col}, mat.row->dtype.bits, mat.row->ctx);
  const auto& rst = COOGetDataAndIndices(mat, rows, cols);
  return rst[2];
}

/*!
 * \brief Get the data for each (row, col) pair.
 *
 * The operator supports matrix with duplicate entries but only one matched entry
 * will be returned for each (row, col) pair. Support duplicate input (row, col)
 * pairs.
 *
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 *
 * \param mat Sparse matrix.
 * \param rows Row index.
 * \param cols Column index.
 * \return Data array. The i^th element is the data of (rows[i], cols[i])
 */
runtime::NDArray COOGetData(COOMatrix mat, runtime::NDArray rows, runtime::NDArray cols);

/*! \brief Return a transposed COO matrix */
COOMatrix COOTranspose(COOMatrix coo);

/*!
 * \brief Convert COO matrix to CSR matrix.
 *
 * If the input COO matrix does not have data array, the data array of
 * the result CSR matrix stores a shuffle index for how the entries
 * will be reordered in CSR. The i^th entry in the result CSR corresponds
 * to the CSR.data[i] th entry in the input COO.
 *
 * Conversion complexity: O(nnz)
 *
 * - The function first check whether the input COO matrix is sorted
 *   using a linear scan.
 * - If the COO matrix is row sorted, the conversion can be done very
 *   efficiently in a sequential scan. The result indices and data arrays
 *   are directly equal to the column and data arrays from the input.
 * - If the COO matrix is further column sorted, the result CSR is
 *   also column sorted.
 * - Otherwise, the conversion is more costly but still is O(nnz).
 *
 * \param coo Input COO matrix.
 * \return CSR matrix.
 */
CSRMatrix COOToCSR(COOMatrix coo);

/*!
 * \brief Slice rows of the given matrix and return.
 * \param coo COO matrix
 * \param start Start row id (inclusive)
 * \param end End row id (exclusive)
 */
COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end);
COOMatrix COOSliceRows(COOMatrix coo, runtime::NDArray rows);

/*!
 * \brief Get the submatrix specified by the row and col ids.
 *
 * In numpy notation, given matrix M, row index array I, col index array J
 * This function returns the submatrix M[I, J].
 *
 * \param coo The input coo matrix
 * \param rows The row index to select
 * \param cols The col index to select
 * \return submatrix
 */
COOMatrix COOSliceMatrix(COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);

/*! \return True if the matrix has duplicate entries */
bool COOHasDuplicate(COOMatrix coo);

/*!
 * \brief Deduplicate the entries of a sorted COO matrix, replacing the data with the
 * number of occurrences of the row-col coordinates.
 */
std::pair<COOMatrix, IdArray> COOCoalesce(COOMatrix coo);

/*!
 * \brief Sort the indices of a COO matrix in-place.
 *
 * The function sorts row indices in ascending order. If sort_column is true,
 * col indices are sorted in ascending order too. The data array of the returned COOMatrix
 * stores the shuffled index which could be used to fetch edge data.
 *
 * Complexity: O(N*log(N)) time and O(1) space, where N is the number of nonzeros.
 * TODO(minjie): The time complexity could be improved to O(N) by using a O(N) space.
 *
 * \param mat The coo matrix to sort.
 * \param sort_column True if column index should be sorted too.
 */
void COOSort_(COOMatrix* mat, bool sort_column = false);

/*!
 * \brief Sort the indices of a COO matrix.
 *
 * The function sorts row indices in ascending order. If sort_column is true,
 * col indices are sorted in ascending order too. The data array of the returned COOMatrix
 * stores the shuffled index which could be used to fetch edge data.
 *
 * Complexity: O(N*log(N)) time and O(1) space, where N is the number of nonzeros.
 * TODO(minjie): The time complexity could be improved to O(N) by using a O(N) space.
 *
 * \param mat The input coo matrix
 * \param sort_column True if column index should be sorted too.
 * \return COO matrix with index sorted.
 */
inline COOMatrix COOSort(COOMatrix mat, bool sort_column = false) {
  if ((mat.row_sorted && !sort_column) || mat.col_sorted)
    return mat;
  COOMatrix ret(mat.num_rows, mat.num_cols,
                mat.row.Clone(), mat.col.Clone(),
                COOHasData(mat)? mat.data.Clone() : mat.data,
                mat.row_sorted, mat.col_sorted);
  COOSort_(&ret, sort_column);
  return ret;
}

/*!
 * \brief Remove entries from COO matrix by entry indices (data indices)
 * \return A new COO matrix as well as a mapping from the new COO entries to the old COO
 *         entries.
 */
COOMatrix COORemove(COOMatrix coo, IdArray entries);

/*!
 * \brief Reorder the rows and colmns according to the new row and column order.
 * \param csr The input coo matrix.
 * \param new_row_ids the new row Ids (the index is the old row Id)
 * \param new_col_ids the new column Ids (the index is the old col Id).
 */
COOMatrix COOReorder(COOMatrix coo, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

/*!
 * \brief Randomly select a fixed number of non-zero entries along each given row independently.
 *
 * The function performs random choices along each row independently.
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than num_samples,
 * all the values are picked.
 *
 * Examples:
 *
 * // coo.num_rows = 4;
 * // coo.num_cols = 4;
 * // coo.rows = [0, 0, 1, 3, 3]
 * // coo.cols = [0, 1, 1, 2, 3]
 * // coo.data = [2, 3, 0, 1, 4]
 * COOMatrix coo = ...;
 * IdArray rows = ... ; // [1, 3]
 * COOMatrix sampled = COORowWiseSampling(coo, rows, 2, FloatArray(), false);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [1, 3, 3]
 * // sampled.cols = [1, 2, 3]
 * // sampled.data = [3, 0, 4]
 *
 * \param mat Input coo matrix.
 * \param rows Rows to sample from.
 * \param num_samples Number of samples
 * \param prob Unnormalized probability array. Should be of the same length as the data array.
 *             If an empty array is provided, assume uniform.
 * \param replace True if sample with replacement
 * \return A COOMatrix storing the picked row and col indices. Its data field stores the
 *         the index of the picked elements in the value array.
 */
COOMatrix COORowWiseSampling(
    COOMatrix mat,
    IdArray rows,
    int64_t num_samples,
    FloatArray prob = FloatArray(),
    bool replace = true);

/*!
 * \brief Randomly select a fixed number of non-zero entries for each edge type
 *        along each given row independently.
 *
 * The function performs random choices along each row independently.
 * In each row, num_samples samples is picked for each edge type. (The edge
 * type is stored in etypes)
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than num_samples,
 * all the values are picked.
 *
 * Examples:
 *
 * // coo.num_rows = 4;
 * // coo.num_cols = 4;
 * // coo.rows = [0, 0, 0, 0, 3]
 * // coo.cols = [0, 1, 3, 2, 3]
 * // coo.data = [2, 3, 0, 1, 4]
 * // etype = [0, 0, 0, 2, 1]
 * COOMatrix coo = ...;
 * IdArray rows = ... ; // [0, 3]
 * std::vector<int64_t> num_samples = {2, 2, 2};
 * COOMatrix sampled = COORowWisePerEtypeSampling(coo, rows, etype, num_samples,
 *                                                FloatArray(), false);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [0, 0, 0, 3]
 * // sampled.cols = [0, 3, 2, 3]
 * // sampled.data = [2, 0, 1, 4]
 *
 * \param mat Input coo matrix.
 * \param rows Rows to sample from.
 * \param etypes Edge types of each edge.
 * \param num_samples Number of samples
 * \param prob Unnormalized probability array. Should be of the same length as the data array.
 *             If an empty array is provided, assume uniform.
 * \param replace True if sample with replacement
 * \param etype_sorted True if the edge types are already sorted
 * \return A COOMatrix storing the picked row and col indices. Its data field stores the
 *         the index of the picked elements in the value array.
 */
COOMatrix COORowWisePerEtypeSampling(
    COOMatrix mat,
    IdArray rows,
    IdArray etypes,
    const std::vector<int64_t>& num_samples,
    FloatArray prob = FloatArray(),
    bool replace = true,
    bool etype_sorted = false);

/*!
 * \brief Select K non-zero entries with the largest weights along each given row.
 *
 * The function performs top-k selection along each row independently.
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than k,
 * all the values are picked.
 *
 * Examples:
 *
 * // coo.num_rows = 4;
 * // coo.num_cols = 4;
 * // coo.rows = [0, 0, 1, 3, 3]
 * // coo.cols = [0, 1, 1, 2, 3]
 * // coo.data = [2, 3, 0, 1, 4]
 * COOMatrix coo = ...;
 * IdArray rows = ... ;  // [0, 1, 3]
 * FloatArray weight = ... ;  // [1., 0., -1., 10., 20.]
 * COOMatrix sampled = COORowWiseTopk(coo, rows, 1, weight);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [0, 1, 3]
 * // sampled.cols = [1, 1, 2]
 * // sampled.data = [3, 0, 1]
 *
 * \param mat Input COO matrix.
 * \param rows Rows to sample from.
 * \param k The K value.
 * \param weight Weight associated with each entry. Should be of the same length as the
 *               data array. If an empty array is provided, assume uniform.
 * \param ascending If true, elements are sorted by ascending order, equivalent to find
 *                  the K smallest values. Otherwise, find K largest values.
 * \return A COOMatrix storing the picked row and col indices. Its data field stores the
 *         the index of the picked elements in the value array.
 */
COOMatrix COORowWiseTopk(
    COOMatrix mat,
    IdArray rows,
    int64_t k,
    NDArray weight,
    bool ascending = false);

/*!
 * \brief Union two COOMatrix into one COOMatrix.
 *
 * Two Matrix must have the same shape.
 *
 * Example:
 *
 * A = [[0, 0, 1, 0],
 *      [1, 0, 1, 1],
 *      [0, 1, 0, 0]]
 *
 * B = [[0, 1, 1, 0],
 *      [0, 0, 0, 1],
 *      [0, 0, 1, 0]]
 *
 * COOMatrix_A.num_rows : 3
 * COOMatrix_A.num_cols : 4
 * COOMatrix_B.num_rows : 3
 * COOMatrix_B.num_cols : 4
 *
 * C = UnionCoo({A, B});
 *
 * C = [[0, 1, 2, 0],
 *      [1, 0, 1, 2],
 *      [0, 1, 1, 0]]
 *
 * COOMatrix_C.num_rows : 3
 * COOMatrix_C.num_cols : 4
 */
COOMatrix UnionCoo(
  const std::vector<COOMatrix>& coos);

/*!
 * \brief DisjointUnion a list COOMatrix into one COOMatrix.
 *
 * Examples:
 *
 * A = [[0, 0, 1],
 *      [1, 0, 1],
 *      [0, 1, 0]]
 *
 * B = [[0, 0],
 *      [1, 0]]
 *
 * COOMatrix_A.num_rows : 3
 * COOMatrix_A.num_cols : 3
 * COOMatrix_B.num_rows : 2
 * COOMatrix_B.num_cols : 2
 *
 * C = DisjointUnionCoo({A, B});
 *
 * C = [[0, 0, 1, 0, 0],
 *      [1, 0, 1, 0, 0],
 *      [0, 1, 0, 0, 0],
 *      [0, 0, 0, 0, 0],
 *      [0, 0, 0, 1, 0]]
 * COOMatrix_C.num_rows : 5
 * COOMatrix_C.num_cols : 5
 *
 * \param coos The input list of coo matrix.
 * \param src_offset A list of integers recording src vertix id offset of each Matrix in coos
 * \param src_offset A list of integers recording dst vertix id offset of each Matrix in coos
 * \return The combined COOMatrix.
 */
COOMatrix DisjointUnionCoo(
  const std::vector<COOMatrix>& coos);

/*!
 * \brief COOMatrix toSimple.
 *
 * A = [[0, 0, 0],
 *      [3, 0, 2],
 *      [1, 1, 0],
 *      [0, 0, 4]]
 *
 * B, cnt, edge_map = COOToSimple(A)
 *
 * B = [[0, 0, 0],
 *      [1, 0, 1],
 *      [1, 1, 0],
 *      [0, 0, 1]]
 * cnt = [3, 2, 1, 1, 4]
 * edge_map = [0, 0, 0, 1, 1, 2, 3, 4, 4, 4, 4]
 *
 * \return The simplified COOMatrix
 *         The count recording the number of duplicated edges from the original graph.
 *         The edge mapping from the edge IDs of original graph to those of the
 *         returned graph.
 */
std::tuple<COOMatrix, IdArray, IdArray> COOToSimple(const COOMatrix& coo);

/*!
 * \brief Split a COOMatrix into multiple disjoin components.
 *
 * Examples:
 *
 * C = [[0, 0, 1, 0, 0],
 *      [1, 0, 1, 0, 0],
 *      [0, 1, 0, 0, 0],
 *      [0, 0, 0, 0, 0],
 *      [0, 0, 0, 1, 0],
 *      [0, 0, 0, 0, 1]]
 * COOMatrix_C.num_rows : 6
 * COOMatrix_C.num_cols : 5
 *
 * batch_size : 2
 * edge_cumsum : [0, 4, 6]
 * src_vertex_cumsum : [0, 3, 6]
 * dst_vertex_cumsum : [0, 3, 5]
 *
 * ret = DisjointPartitionCooBySizes(C,
 *                                   batch_size,
 *                                   edge_cumsum,
 *                                   src_vertex_cumsum,
 *                                   dst_vertex_cumsum)
 *
 * A = [[0, 0, 1],
 *      [1, 0, 1],
 *      [0, 1, 0]]
 * COOMatrix_A.num_rows : 3
 * COOMatrix_A.num_cols : 3
 *
 * B = [[0, 0],
 *      [1, 0],
 *      [0, 1]]
 * COOMatrix_B.num_rows : 3
 * COOMatrix_B.num_cols : 2
 *
 * \param coo COOMatrix to split.
 * \param batch_size Number of disjoin components (Sub COOMatrix)
 * \param edge_cumsum Number of edges of each components
 * \param src_vertex_cumsum Number of src vertices of each component.
 * \param dst_vertex_cumsum Number of dst vertices of each component.
 * \return A list of COOMatrixes representing each disjoint components.
 */
std::vector<COOMatrix> DisjointPartitionCooBySizes(
  const COOMatrix &coo,
  const uint64_t batch_size,
  const std::vector<uint64_t> &edge_cumsum,
  const std::vector<uint64_t> &src_vertex_cumsum,
  const std::vector<uint64_t> &dst_vertex_cumsum);

/*!
 * \brief Slice a contiguous chunk from a COOMatrix
 *
 * Examples:
 *
 * C = [[0, 0, 1, 0, 0],
 *      [1, 0, 1, 0, 0],
 *      [0, 1, 0, 0, 0],
 *      [0, 0, 0, 0, 0],
 *      [0, 0, 0, 1, 0],
 *      [0, 0, 0, 0, 1]]
 * COOMatrix_C.num_rows : 6
 * COOMatrix_C.num_cols : 5
 *
 * edge_range : [4, 6]
 * src_vertex_range : [3, 6]
 * dst_vertex_range : [3, 5]
 *
 * ret = COOSliceContiguousChunk(C,
 *                               edge_range,
 *                               src_vertex_range,
 *                               dst_vertex_range)
 *
 * ret = [[0, 0],
 *        [1, 0],
 *        [0, 1]]
 * COOMatrix_ret.num_rows : 3
 * COOMatrix_ret.num_cols : 2
 *
 * \param coo COOMatrix to slice.
 * \param edge_range ID range of the edges in the chunk
 * \param src_vertex_range ID range of the src vertices in the chunk.
 * \param dst_vertex_range ID range of the dst vertices in the chunk.
 * \return COOMatrix representing the chunk.
 */
COOMatrix COOSliceContiguousChunk(
  const COOMatrix &coo,
  const std::vector<uint64_t> &edge_range,
  const std::vector<uint64_t> &src_vertex_range,
  const std::vector<uint64_t> &dst_vertex_range);

/*!
 * \brief Create a LineGraph of input coo
 *
 * A = [[0, 0, 1],
 *      [1, 0, 1],
 *      [1, 1, 0]]
 * A.row = [0, 1, 1, 2, 2]
 * A.col = [2, 0, 2, 0, 1]
 * A.eid = [0, 1, 2, 3, 4]
 *
 * B = COOLineGraph(A, backtracking=False)
 *
 * B = [[0, 0, 0, 0, 1],
 *      [1, 0, 0, 0, 0],
 *      [0, 0, 0, 1, 0],
 *      [0, 0, 0, 0, 0],
 *      [0, 1, 0, 0, 0]]
 *
 * C = COOLineGraph(A, backtracking=True)
 *
 * C = [[0, 0, 0, 1, 1],
 *      [1, 0, 0, 0, 0],
 *      [0, 0, 0, 1, 1],
 *      [1, 0, 0, 0, 0],
 *      [0, 1, 1, 0, 0]]
 *
 * \param coo COOMatrix to create the LineGraph
 * \param backtracking whether the pair of (v, u) (u, v) edges are treated as linked
 * \return LineGraph in COO format
 */
COOMatrix COOLineGraph(const COOMatrix &coo, bool backtracking);

}  // namespace aten
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::aten::COOMatrix, true);
}  // namespace dmlc

#endif  // DGL_ATEN_COO_H_
