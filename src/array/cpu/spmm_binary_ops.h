/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/spmm_binary_ops.h
 * \brief SPMM CPU Binary ops.
 */
#ifndef DGL_ARRAY_CPU_SPMM_BINARY_OPS_H_
#define DGL_ARRAY_CPU_SPMM_BINARY_OPS_H_
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <limits>
namespace dgl {
namespace aten {
namespace cpu {
namespace op {

//////////////////////////////// binary operators on CPU
///////////////////////////////////
template <typename DType>
struct Add {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off + *rhs_off;
  }
};
template <typename DType>
constexpr bool Add<DType>::use_lhs;
template <typename DType>
constexpr bool Add<DType>::use_rhs;

template <typename DType>
struct Sub {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off - *rhs_off;
  }
};
template <typename DType>
constexpr bool Sub<DType>::use_lhs;
template <typename DType>
constexpr bool Sub<DType>::use_rhs;

template <typename DType>
struct Mul {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off * *rhs_off;
  }
};
template <typename DType>
constexpr bool Mul<DType>::use_lhs;
template <typename DType>
constexpr bool Mul<DType>::use_rhs;

template <typename DType>
struct Div {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off / *rhs_off;
  }
};
template <typename DType>
constexpr bool Div<DType>::use_lhs;
template <typename DType>
constexpr bool Div<DType>::use_rhs;

template <typename DType>
struct CopyLhs {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(const DType* lhs_off, const DType*) {
    return *lhs_off;
  }
};
template <typename DType>
constexpr bool CopyLhs<DType>::use_lhs;
template <typename DType>
constexpr bool CopyLhs<DType>::use_rhs;

template <typename DType>
struct CopyRhs {
  typedef DType type;
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType*, const DType* rhs_off) {
    return *rhs_off;
  }
};
template <typename DType>
constexpr bool CopyRhs<DType>::use_lhs;
template <typename DType>
constexpr bool CopyRhs<DType>::use_rhs;

//////////////////////////////// Reduce operators on CPU
///////////////////////////////////
template <typename DType>
struct Max {
  typedef DType type;
  static constexpr DType zero = -std::numeric_limits<DType>::infinity();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) { return accum < val; }
};
template <typename DType>
constexpr DType Max<DType>::zero;

template <typename DType>
struct Min {
  typedef DType type;
  static constexpr DType zero = std::numeric_limits<DType>::infinity();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) { return accum > val; }
};
template <typename DType>
constexpr DType Min<DType>::zero;

#define SWITCH_OP(op, Op, ...)                                  \
  do {                                                          \
    if ((op) == "add") {                                        \
      typedef dgl::aten::cpu::op::Add<DType> Op;                \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "sub") {                                 \
      typedef dgl::aten::cpu::op::Sub<DType> Op;                \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "mul") {                                 \
      typedef dgl::aten::cpu::op::Mul<DType> Op;                \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "div") {                                 \
      typedef dgl::aten::cpu::op::Div<DType> Op;                \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_lhs") {                            \
      typedef dgl::aten::cpu::op::CopyLhs<DType> Op;            \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_rhs") {                            \
      typedef dgl::aten::cpu::op::CopyRhs<DType> Op;            \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op; \
    }                                                           \
  } while (0)

#define SWITCH_BITS(bits, DType, ...)                           \
  do {                                                          \
    if ((bits) == 16) {                                         \
      LOG(FATAL) << "FP16 not supported on CPU";                \
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

}  // namespace op

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SPMM_BINARY_OPS_H_
