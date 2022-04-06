#pragma once

// Attempt to keep as many providers as possible in the compiled library.
// Tries to follow a flat #ifdef instead of heavy nesting in this file.
// If a library can exist without conflicting, they're included. There are
// definitions switch which are propogated from corresponding CMake variables.
//
// Templates are used to keep multiple implementations ODR compatible. Routing
// happens once all the implementations are defined, based on precedence.
//
// For example, Eigen is a fallback. An x86-64 processor will have Intel MKL.
// Both of these can co-exist. Addition deletion can be done at compile time by
// controlling the respective CMake variable.
//
// 0. The simplest implementation is an ABORT, as it existed before.
// 1. We incrementally add implementations the standard GEMM API using
//    different providers.
// 2. Given a functional GEMM, BatchedGEMM can be realized by explicitly looping.
// 3. Some providers allow faster variants of BatchedGEMM by reducing
//    allocations/grouping. In this case, we explicitly specialize the templates
//    to the faster implementation.
//
// Client calls a GemmBatched, through a translation layer from marian::Tensor
// to GEMM API arguments.
//
// Units are added or removed as a whole, without interspersing ifdefs in an
// attempt to DRY. This leads to an increased verbosity, much the units are
// much more pliable.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#ifdef MARIAN_USE_RUY_SGEMM
#include <ruy/ruy.h>
#include <ruy/system_aligned_alloc.h>
#endif  // MARIAN_USE_RUY_SGEMM

#ifdef MARIAN_USE_MKL
#include <mkl.h>
#endif  // MARIAN_USE_MKL

#ifdef MARIAN_USE_EIGEN_SGEMM
#include "Eigen/Core"
#include "Eigen/Dense"
#endif  // MARIAN_USE_EIGEN_SGEMM

#ifdef MARIAN_USE_BLAS
#include <cblas.h>
#endif  // MARIAN_USE_BLAS

#include "tensor.h"

namespace marian {
namespace gemm {

// The following is about to be used further down below in templating multiple
// implementations, allowing them to exist in an ODR compatible way.
enum class Provider {
  kNone,
  kEigen,  // Eigen Library; Portable fallback. Works on most platforms. Used by
           // WASM
  kMKL,    //
  kBLAS,   //
  kRuy,    // Ruy, targetting ARM. X86 etc available, but not best.
  kARMPL   // ARM Performance Library
};

// A marian connected GEMM function. Arguments are in the order of the
// expression being evaluated:
//
//    C = alpha * op(A) * op(B) + beta*C
//
// transA, transB are boolean flags deciding whether to transpose the matrices
// A or B.
//
// op(A) is an M x K matrix, op(B) is a K x N matrix. Supply M, K, N
// accordingly.
//

template <enum Provider>
inline void Gemm(bool transA,
                 bool transB,
                 int M,
                 int N,
                 int K,
                 float alpha,
                 float *A,
                 int lda,
                 float *B,
                 int ldb,
                 float beta,
                 float *C,
                 int ldc) {
  ABORT("No available GEMM Implementation;");
}

template <enum Provider>
inline void GemmBatched(bool transA,
                        bool transB,
                        int batchSize,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        float *A,
                        int lda,
                        float *B,
                        int ldb,
                        float beta,
                        float *C,
                        int ldc) {
  ABORT("No available GEMM (Batched) Implementation;");
}

#ifdef MARIAN_USE_EIGEN_SGEMM

// Minimum definitions required for the PyTorch import to work. Taken from:
// https://github.com/pytorch/pytorch/blob/936e7eabcabc97fbc40f488e67a94c4733c33dd6/caffe2/utils/eigen_utils.h
using EigenOuterStride = Eigen::OuterStride<Eigen::Dynamic>;
template <typename T>
using EigenOuterStridedMatrixMap
    = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenOuterStride>;
template <typename T>
using ConstEigenOuterStridedMatrixMap
    = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenOuterStride>;

template <>
inline void Gemm<Provider::kEigen>(bool transA,
                                   bool transB,
                                   int M,
                                   int N,
                                   int K,
                                   float alpha,
                                   float *A,
                                   int lda,
                                   float *B,
                                   int ldb,
                                   float beta,
                                   float *C,
                                   int ldc) {
  CBLAS_TRANSPOSE trans_A = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE trans_B = transB ? CblasTrans : CblasNoTrans;

  // Taken from https://github.com/pytorch/pytorch/blob/0d7aad822e77b9f5ca9649114b6c2cbdf54564e3/caffe2/utils/math_cpu.cc#L155
  // PyTorch. BSD License.
  EigenOuterStridedMatrixMap<float> C_mat(C, N, M, EigenOuterStride(ldc));
  if(beta == 0) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }

  switch(trans_A) {
    case CblasNoTrans: {
      switch(trans_B) {
        case CblasNoTrans:
          C_mat.noalias()
              += alpha
                 * (ConstEigenOuterStridedMatrixMap<float>(B, N, K, EigenOuterStride(ldb))
                    * ConstEigenOuterStridedMatrixMap<float>(A, K, M, EigenOuterStride(lda)));
          return;
        case CblasTrans:
          C_mat.noalias()
              += alpha
                 * (ConstEigenOuterStridedMatrixMap<float>(B, K, N, EigenOuterStride(ldb))
                        .transpose()
                    * ConstEigenOuterStridedMatrixMap<float>(A, K, M, EigenOuterStride(lda)));
          return;
        default:
          ABORT("Unexpected CBLAS_TRANSPOSE for trans_B");
          return;  // The line above calls `abort()`. Should never reach here.
      }
    }
    case CblasTrans: {
      switch(trans_B) {
        case CblasNoTrans:
          C_mat.noalias()
              += alpha
                 * (ConstEigenOuterStridedMatrixMap<float>(B, N, K, EigenOuterStride(ldb))
                    * ConstEigenOuterStridedMatrixMap<float>(A, M, K, EigenOuterStride(lda))
                          .transpose());
          return;
        case CblasTrans:
          C_mat.noalias()
              += alpha
                 * (ConstEigenOuterStridedMatrixMap<float>(B, K, N, EigenOuterStride(ldb))
                        .transpose()
                    * ConstEigenOuterStridedMatrixMap<float>(A, M, K, EigenOuterStride(lda))
                          .transpose());
          return;
        default:
          ABORT("Unexpected CBLAS_TRANSPOSE for trans_B");
          return;  // The line above calls `abort()`. Should never reach here.
      }
    }
    default: ABORT("Unexpected CBLAS_TRANSPOSE for trans_A");
  }
}
#endif  // MARIAN_USE_EIGEN_SGEMM

#ifdef MARIAN_USE_BLAS
template <>
inline void Gemm<Provider::kBLAS>(bool transA,
                                  bool transB,
                                  int M,
                                  int N,
                                  int K,
                                  float alpha,
                                  float *A,
                                  int lda,
                                  float *B,
                                  int ldb,
                                  float beta,
                                  float *C,
                                  int ldc) {
  // Converting booleans to CBLAS_TRANSPOSE (char).
  CBLAS_TRANSPOSE cTransA = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE cTransB = transB ? CblasTrans : CblasNoTrans;
  cblas_sgemm(CblasRowMajor, cTransA, cTransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif  // MARIAN_USE_BLAS

// Translates marian::Tensor to GEMM API args.
inline void inferGemmParamsFromTensor(marian::Tensor C,
                                      marian::Tensor A,
                                      marian::Tensor B,
                                      bool transA,
                                      bool transB,
                                      size_t &M,
                                      size_t &N,
                                      size_t &K,
                                      size_t &batchSize) {
  // Incoming matrices are row-major storage format.
  // N1 x N2 x .. N_k x rows x cols
  //                     -2  x - 1
  M = A->shape()[-2];
  K = A->shape()[-1];

  if(transA) {
    std::swap(M, K);
  }

  size_t L;
  L = B->shape()[-2];
  N = B->shape()[-1];

  if(transB) {
    std::swap(L, N);
  }

  // To be compliant for matrix multiplication.
  auto computeBatchSize = [](const marian::Tensor &T, int rows, int cols) {
    return T->shape().size() / (rows * cols);
  };

  assert(L == K);
  assert(computeBatchSize(A, M, K) == computeBatchSize(B, K, N));
  assert(computeBatchSize(A, M, K) == computeBatchSize(C, M, N));

  batchSize = A->shape().size() / (M * K);
}

#ifdef MARIAN_USE_RUY_SGEMM
namespace {

template <class T>
class AlignedVector {
public:
  AlignedVector(size_t num_elem)
      : size_(num_elem),
        storage_(reinterpret_cast<T *>(ruy::detail::SystemAlignedAlloc(sizeof(T) * num_elem))) {}

  T *begin() { return storage_; }
  T *data() { return storage_; }
  size_t size() const { return size_; }
  size_t memSize() const { return sizeof(T) * size_; }

  // Forbid copy
  AlignedVector(const AlignedVector &) = delete;
  AlignedVector &operator=(const AlignedVector &) = delete;

  ~AlignedVector() { ruy::detail::SystemAlignedFree(reinterpret_cast<void *>(storage_)); }

private:
  T *storage_;
  size_t size_;
};

}  // namespace

template <>
inline void Gemm<Provider::kRuy>(bool transA,
                                 bool transB,
                                 int M,
                                 int N,
                                 int K,
                                 float alpha,
                                 float *A,
                                 int lda,
                                 float *B,
                                 int ldb,
                                 float beta,
                                 float *C,
                                 int ldc) {
  ruy::Context context;

  // If we need to transpose, we can swap dimensions in layout claim the matrix
  // is just column-major. Set ordering so transpose.
  const auto orderA = (transA ? ruy::Order::kColMajor : ruy::Order::kRowMajor);
  const auto orderB = (transB ? ruy::Order::kColMajor : ruy::Order::kRowMajor);

  AlignedVector<float> intermediate(M * N);

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(M, K, orderA, lhs.mutable_layout());
  lhs.set_data(A);

  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(K, N, orderB, rhs.mutable_layout());
  rhs.set_data(B);

  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(M, N, ruy::Order::kRowMajor, dst.mutable_layout());
  dst.set_data(intermediate.data());

  ruy::MulParams<float, float> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  // Write out C as C = alpha * [op(A) * op(B)] + beta * C
  // Can we expect the compiler to autovectorize this?
  // TODO: Come back and explicitly use SIMD.
  const size_t size    = M * N;
  const float *opA_opB = intermediate.data();
  for(size_t i = 0; i < size; i++) {
    C[i] = alpha * opA_opB[i] + beta * C[i];
  }
}

// See documentation for Gemm above. Adds a batchSize parameter, which is used
// if the available libraries provide one. Else, we resort to using an explicit
// batching.
#define __UNROLL(provider)                         \
  template <>                                      \
  inline void GemmBatched<provider>(bool transA,   \
                                    bool transB,   \
                                    int batchSize, \
                                    int M,         \
                                    int N,         \
                                    int K,         \
                                    float alpha,   \
                                    float *A,      \
                                    int lda,       \
                                    float *B,      \
                                    int ldb,       \
                                    float beta,    \
                                    float *C,      \
                                    int ldc) {     \
    size_t strideA = M * K;                        \
    size_t strideB = K * N;                        \
    size_t strideC = M * N;                        \
                                                   \
    for(size_t i = 0; i < batchSize; ++i) {        \
      Gemm<provider>(transA,                       \
                     transB,                       \
                     (int)M,                       \
                     (int)N,                       \
                     (int)K,                       \
                     alpha,                        \
                     A + i * strideA,              \
                     (int)lda,                     \
                     B + i * strideB,              \
                     (int)ldb,                     \
                     beta,                         \
                     C + i * strideC,              \
                     (int)ldc);                    \
    }                                              \
  }

__UNROLL(Provider::kBLAS);
__UNROLL(Provider::kEigen);

#undef __UNROLL

template <>
void GemmBatched<Provider::kRuy>(bool transA,
                                 bool transB,
                                 int batchSize,
                                 int M,
                                 int N,
                                 int K,
                                 float alpha,
                                 float *A,
                                 int lda,
                                 float *B,
                                 int ldb,
                                 float beta,
                                 float *C,
                                 int ldc) {
  ruy::Context context;

  // If we need to transpose, we can swap dimensions in layout claim the matrix
  // is just column-major. Set ordering so transpose.
  const auto orderA = (transA ? ruy::Order::kColMajor : ruy::Order::kRowMajor);
  const auto orderB = (transB ? ruy::Order::kColMajor : ruy::Order::kRowMajor);

  size_t strideA = M * K;
  size_t strideB = K * N;
  size_t strideC = M * N;

  // Compute AB (op(A)*op(B), given we have configured transpose)
  // Ruy allows some form of bias
  AlignedVector<float> intermediate(batchSize * M * N);

  for(size_t batchId = 0; batchId < batchSize; batchId++) {
    ruy::Matrix<float> lhs;
    ruy::MakeSimpleLayout(M, K, orderA, lhs.mutable_layout());
    lhs.set_data(A + batchId * strideA);

    ruy::Matrix<float> rhs;
    ruy::MakeSimpleLayout(K, N, orderB, rhs.mutable_layout());
    rhs.set_data(B + batchId * strideB);

    ruy::Matrix<float> dst;
    ruy::MakeSimpleLayout(M, N, ruy::Order::kRowMajor, dst.mutable_layout());
    dst.set_data(intermediate.data() + batchId * strideC);

    ruy::MulParams<float, float> mul_params;
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);
  }

  // Write out C as C = alpha * [op(A) * op(B)] + beta * C
  // Can we expect the compiler to autovectorize this?
  // TODO: Come back and explicitly use SIMD.
  const size_t cSize   = batchSize * M * N;
  const float *opA_opB = intermediate.data();
  for(size_t i = 0; i < cSize; i++) {
    C[i] = alpha * opA_opB[i] + beta * C[i];
  }
}

void gemmRuy(marian::Tensor C,
             marian::Tensor A,
             marian::Tensor B,
             bool transA,
             bool transB,
             float beta,
             float alpha) {
  size_t M, K, N, batchSize;
  inferGemmParamsFromTensor(C, A, B, transA, transB, M, N, K, batchSize);

  size_t lda = A->shape()[-1];
  size_t ldb = A->shape()[-1];
  size_t ldc = N;

  GemmBatched<Provider::kRuy>(transA,
                              transB,
                              batchSize,
                              M,
                              N,
                              K,
                              alpha,
                              A->data(),
                              lda,
                              B->data(),
                              ldb,
                              beta,
                              C->data(),
                              ldc);
}

#endif

#ifdef MARIAN_USE_MKL

template <>
inline void GemmBatched<Provider::kMKL>(bool transA,
                                        bool transB,
                                        int batchSize,
                                        int M,
                                        int N,
                                        int K,
                                        float alpha,
                                        float *A,
                                        int lda,
                                        float *B,
                                        int ldb,
                                        float beta,
                                        float *C,
                                        int ldc) {
  CBLAS_TRANSPOSE trans_A = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE trans_B = transB ? CblasTrans : CblasNoTrans;

  /* cblas_sgemm_batch allows us to group all the small GEMMs that are done in a
   * for loop with sgemm and compute them in only one MKL call. 
   * The API supports dependencies, where you can specify one "group" of GEMMs
   * to be computed after another. (This controlled by the group_count
   * parameter). In our case, the operations are not dependent on one another so
   * we hardcode one group. The rest of the arguments (with the exception of
   * group_size) are the same as the ones that cblas_sgemm expects, with the
   * difference that we are supposed to provide an array pointer (One element
   * per group).
   */

  const MKL_INT mM(M), mN(N), mK(K);
  const MKL_INT mlda(lda), mldb(ldb), mldc(ldc);
  const MKL_INT strideB(mN * mK), strideA(mM * mK), strideC(mN * mM);
  const MKL_INT mBatchSize(batchSize);

  std::vector<const float *> A_array(batchSize, nullptr);
  std::vector<const float *> B_array(batchSize, nullptr);

  std::vector<float *> C_array(batchSize, nullptr);

  for(int i = 0; i < batchSize; ++i) {
    A_array[i] = A + i * strideA;
    B_array[i] = B + i * strideB;
    C_array[i] = C + i * strideC;
  }

  cblas_sgemm_batch(CblasRowMajor,
                    &trans_A,
                    &trans_B,
                    &mM,
                    &mN,
                    &mK,
                    &alpha,
                    A_array.data(),
                    &mlda,
                    B_array.data(),
                    &mldb,
                    &beta,
                    C_array.data(),
                    &mldc,
                    /*group_count=*/1,
                    &mBatchSize);
}
#endif  // MARIAN_USE_MKL

void ProdBatchedOld(marian::Tensor C,
                    const marian::Tensor A,
                    const marian::Tensor B,
                    bool transA,
                    bool transB,
                    float beta,
                    float alpha) {
  size_t M, K, N, batchSize;
  inferGemmParamsFromTensor(C, A, B, transA, transB, M, N, K, batchSize);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = N;

#ifdef MARIAN_USE_MKL
  GemmBatched<Provider::kMKL>(transA,
                              transB,
                              batchSize,
                              M,
                              N,
                              K,
                              alpha,
                              A->data(),
                              lda,
                              B->data(),
                              ldb,
                              beta,
                              C->data(),
                              ldc);
#else   // MARIAN_USE_MKL
  GemmBatched<Provider::kBLAS>(transA,
                               transB,
                               batchSize,
                               M,
                               N,
                               K,
                               alpha,
                               A->data(),
                               lda,
                               B->data(),
                               ldb,
                               beta,
                               C->data(),
                               ldc);
#endif  // MARIAN_USE_MKL
}

#ifndef SGEMM_IMPL_
#define SGEMM_IMPL_
#include "gemm-impl.cpp"
#endif

void dispatch(std::string provider,
              marian::Tensor C,
              const marian::Tensor A,
              const marian::Tensor B,
              bool transA,
              bool transB,
              float beta,
              float alpha) {
  size_t M, N, K, batchSize;
  inferGemmParamsFromTensor(C, A, B, transA, transB, M, N, K, batchSize);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = N;

  void (*gemmFn)(bool transA,
                 bool transB,
                 int batchSize,
                 int M,
                 int N,
                 int K,
                 float alpha,
                 float *A,
                 int lda,
                 float *B,
                 int ldb,
                 float beta,
                 float *C,
                 int ldc)
      = nullptr;

  if(provider == "ruy") {
    gemmFn = &GemmBatched<Provider::kRuy>;
  } else if(provider == "mkl") {
    gemmFn = &GemmBatched<Provider::kMKL>;
  } else if(provider == "blas") {
    gemmFn = &GemmBatched<Provider::kBLAS>;
  } else if(provider == "eigen") {
    gemmFn = &GemmBatched<Provider::kEigen>;
  } else {
    ABORT("Unknown Gemm Provider {}", provider);
  }

  // Make call
  gemmFn(transA,
         transB,
         batchSize,
         M,
         N,
         K,
         alpha,
         A->data(),
         lda,
         B->data(),
         ldb,
         beta,
         C->data(),
         ldc);
}

}  // namespace gemm
}  // namespace marian
