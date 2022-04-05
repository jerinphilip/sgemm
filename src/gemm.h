#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#ifdef MARIAN_USE_RUY_SGEMM
#include <ruy/ruy.h>
#endif  // MARIAN_USE_RUY_SGEMM
#include "tensor.h"

#ifdef MARIAN_USE_MKL
#include <mkl.h>
#endif  // MARIAN_USE_MKL

#ifdef WASM_COMPATIBLE_BLAS
#include "3rd_party/onnxjs/src/wasm-ops/gemm.h"
#endif  // WASM_COMPATIBLE_BLAS

#ifdef MARIAN_USE_BLAS
#include <cblas.h>
#endif  // MARIAN_USE_BLAS

namespace marian {
namespace gemm {

// The following is about to be used further down below in templating multiple
// implementations, allowing them to exist in an ODR compatible way.
enum class Provider {
  kEigen,  // Eigen Library; Portable fallback. Works on most platforms. Used by WASM
  kMKL,    //
  kBLAS,   //
  kRuy     // Ruy, targetting ARM. X86 etc available, but not best.
};

// A marian connected GEMM function. Arguments are in the order of the
// expression being evaluated:
//
//    C = alpha * op(A) * op(B) + beta*C
//
// transA, transB are boolean flags deciding whether to transpose the matrices
// A or B.
//
// op(A) is an M x K matrix, op(B) is a K x N matrix. Supply M, K, N accordingly.
//
// We do not bother about lda, ldb, ldc, as all calls that reach here
// come though `ProdBatched` and sub-matrices / views are not expected.
// In this case, we can infer what LDA is from M, N, K, transA, transB.

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

// See documentation for Gemm above. Adds a batchSize parameter, which is used
// if the available libraries provide one. Else, we resort to using an explicit
// batching.
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
  ABORT("No available GEMM Implementation;");
}

#ifdef WASM_COMPATIBLE_BLAS
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
  // TODO: Use lda, ldb, ldc skipping ONNXJS and adding Eigen
  gemm_f32_imp(transA, transB, M, K, N, alpha, A, B, beta, C);
}
#endif  //WASM_COMPATIBLE_BLAS

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
                                 int ldc) {}

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
  assert(L == K);

  batchSize = A->shape().size() / (M * K);
}

void gemmRuy(marian::Tensor C,
             marian::Tensor A,
             marian::Tensor B,
             bool transA,
             bool transB,
             float beta,
             float alpha) {
  ruy::Context context;

  size_t M, K, N, batchSize;
  inferGemmParamsFromTensor(C, A, B, transA, transB, M, N, K, batchSize);

  // If we need to transpose, we can swap dimensions in layout claim the matrix is just
  // column-major. Set ordering so transpose.
  const auto orderA = (transA ? ruy::Order::kColMajor : ruy::Order::kRowMajor);
  const auto orderB = (transB ? ruy::Order::kColMajor : ruy::Order::kRowMajor);

  size_t strideA = M * K;
  size_t strideB = K * N;
  size_t strideC = M * N;

  // Compute AB (op(A)*op(B), given we have configured transpose)
  // Ruy allows some form of bias
  marian::Tensor AB = marian::make_tensor({batchSize, M, N});

  for(size_t batchId = 0; batchId < batchSize; batchId++) {
    ruy::Matrix<float> lhs;
    ruy::MakeSimpleLayout(M, K, orderA, lhs.mutable_layout());
    lhs.set_data(A->data() + batchId * strideA);

    ruy::Matrix<float> rhs;
    ruy::MakeSimpleLayout(K, N, orderB, rhs.mutable_layout());
    rhs.set_data(B->data() + batchId * strideB);

    ruy::Matrix<float> dst;
    ruy::MakeSimpleLayout(M, N, ruy::Order::kRowMajor, dst.mutable_layout());
    dst.set_data(AB->data() + batchId * strideC);

    ruy::MulParams<float, float> mul_params;
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);
  }

  // Write out C as C = alpha * [op(A) * op(B)] + beta * C
  // Can we expect the compiler to autovectorize this?
  // TODO: Come back and explicitly use SIMD.
  float *cData        = C->data();
  const size_t cSize  = C->shape().size();
  const float *ABData = AB->data();
  for(size_t i = 0; i < cSize; i++) {
    cData[i] = alpha * ABData[i] + beta * cData[i];
  }
}

void ProdBatchedOld(marian::Tensor C,
                    const marian::Tensor A,
                    const marian::Tensor B,
                    bool transA,
                    bool transB,
                    float beta,
                    float alpha) {
  /// The map to the notations below:
  /// m x k matrix is being multiplied with l x n

  size_t batchA = A->shape().size() / (A->shape()[-1] * A->shape()[-2]);
  size_t batchB = B->shape().size() / (B->shape()[-1] * B->shape()[-2]);

  size_t m = A->shape()[-2];
  size_t k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  size_t l = B->shape()[-2];
  size_t n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  auto strideB = batchB == 1 ? 0 : n * k;
  auto strideA = batchA == 1 ? 0 : m * k;
  auto strideC = n * m;

  auto batchC = std::max(batchA, batchB);

#if MARIAN_USE_MKL
  CBLAS_TRANSPOSE transA_forarr = CblasNoTrans;
  CBLAS_TRANSPOSE transB_forarr = CblasNoTrans;

  if(transA)
    transA_forarr = CblasTrans;

  if(transB)
    transB_forarr = CblasTrans;

  /* cblas_sgemm_batch allows us to group all the small GEMMs that are done in a
   * for loop with sgemm and compute them in only one MKL call. For the API
   * documentation refer to
   * https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-like-extensions/cblas-gemm-batch.html
   * The API supports dependencies, where you can specify one "group" of GEMMs
   * to be computed after another. (This controlled by the group_count
   * parameter). In our case, the operations are not dependent on one another so
   * we hardcode one group. The rest of the arguments (with the exception of
   * group_size) are the same as the ones that cblas_sgemm expects, with the
   * difference that we are supposed to provide an array pointer (One element
   * per group). Weirdly enough, we are required to to provide all of the
   * integer arguments as the MKL_INT datatype
   */

  static const constexpr size_t group_count = 1;  // We have one group
  const std::vector<CBLAS_TRANSPOSE> transa_arr(group_count, transA_forarr);
  const std::vector<CBLAS_TRANSPOSE> transb_arr(group_count, transB_forarr);
  const std::vector<MKL_INT> m_arr(group_count, (MKL_INT)m);
  const std::vector<MKL_INT> n_arr(group_count, (MKL_INT)n);
  const std::vector<MKL_INT> k_arr(group_count, (MKL_INT)k);
  const std::vector<float> alpha_arr(group_count, alpha);
  const std::vector<float> beta_arr(group_count, beta);
  const std::vector<MKL_INT> lda_arr(group_count, (MKL_INT)lda);
  const std::vector<MKL_INT> ldb_arr(group_count, (MKL_INT)ldb);
  const std::vector<MKL_INT> ldc_arr(group_count, (MKL_INT)ldc);
  const std::vector<MKL_INT> group_size(group_count,
                                        (MKL_INT)batchC);  // Group size specifies number of GEMM
                                                           // operations per group (Which is batchC)

  std::vector<const float *> a_array(batchC, nullptr);
  std::vector<const float *> b_array(batchC, nullptr);
  std::vector<float *> c_array(batchC, nullptr);

  // This loop initializes the array pointers in the same way as the for loop
  // in the normal sgemm version a few lines below
  for(size_t i = 0; i < batchC; ++i) {
    a_array[i] = A->data() + (i % batchA) * strideA;
    b_array[i] = B->data() + (i % batchB) * strideB;
    c_array[i] = C->data() + i * strideC;
  }
  cblas_sgemm_batch(CblasRowMajor,
                    &transa_arr[0],
                    &transb_arr[0],
                    &m_arr[0],
                    &n_arr[0],
                    &k_arr[0],
                    &alpha_arr[0],
                    &a_array[0],
                    &lda_arr[0],
                    &b_array[0],
                    &ldb_arr[0],
                    &beta_arr[0],
                    &c_array[0],
                    &ldc_arr[0],
                    group_count,
                    &group_size[0]);
#else   // MARIAN_USE_MKL
  for(size_t i = 0; i < batchC; ++i) {
    Gemm<Provider::kBLAS>(transA,
                          transB,
                          (int)m,
                          (int)n,
                          (int)k,
                          alpha,
                          A->data() + (i % batchA) * strideA,
                          (int)lda,
                          B->data() + (i % batchB) * strideB,
                          (int)ldb,
                          beta,
                          C->data() + i * strideC,
                          (int)ldc);
  }
#endif  // MARIAN_USE_MKL
}

#ifndef SGEMM_IMPL_
#define SGEMM_IMPL_
#include "gemm-impl.cpp"
#endif
}  // namespace gemm
}  // namespace marian
