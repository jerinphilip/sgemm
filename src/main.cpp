#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include <ruy/ruy.h>
#include "stub_tensor.h"

#if MKL_FOUND
#include <mkl.h>
#elif BLAS_FOUND
#if WASM_COMPATIBLE_BLAS
#include "3rd_party/onnxjs/src/wasm-ops/gemm.h"
#else
#include <cblas.h>
#endif  // WASM_COMPATIBLE_BLAS
#endif

#define ABORT(s)                                \
  do {                                          \
    std::cerr << "Aborted: " << s << std::endl; \
    std::abort();                               \
  } while(0)

inline void sgemm(bool transA,
                  bool transB,
                  int rows_a,
                  int rows_b,
                  int width,
                  float alpha,
                  float *a,
                  int lda,
                  float *b,
                  int ldb,
                  float beta,
                  float *c,
                  int ldc) {
#if BLAS_FOUND
#if WASM_COMPATIBLE_BLAS
  gemm_f32_imp(transA, transB, rows_a, rows_b, width, alpha, a, b, beta, c);
#else   // WASM_COMPATIBLE_BLAS
  cblas_sgemm(CblasRowMajor,
              transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans,
              rows_a,
              rows_b,
              width,
              alpha,
              a,
              lda,
              b,
              ldb,
              beta,
              c,
              ldc);
#endif  // WASM_COMPATIBLE_BLAS
#else   // BLAS_FOUND
  transA;
  transB;
  rows_a;
  rows_b;
  width;
  alpha;
  a;
  lda;
  b;
  ldb;
  beta;
  c;
  ldc;  // make compiler happy
  ABORT("Marian must be compiled with a BLAS library");
#endif  // BLAS_FOUND
}

void MulFloat(marian::Tensor C, marian::Tensor A, marian::Tensor B) {
  ruy::Context context;
  size_t m, k, n;

  m = A->shape()[-2];
  k = A->shape()[-1];
  // l = B->shape()[-2];
  n = B->shape()[-1];

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(m, k, ruy::Order::kRowMajor, lhs.mutable_layout());
  lhs.set_data(A->data());
  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(k, n, ruy::Order::kRowMajor, rhs.mutable_layout());
  rhs.set_data(B->data());
  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(m, n, ruy::Order::kRowMajor, dst.mutable_layout());
  dst.set_data(C->data());

  ruy::MulParams<float, float> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);
}

void ProdBatchedOld(marian::Tensor C,
                    const marian::Tensor A,
                    const marian::Tensor B,
                    bool transA,
                    bool transB,
                    float beta,
                    float scalar) {
#if BLAS_FOUND
  /// The map to the notations below:
  /// m x k matrix is being multiplied with l x n
  float alpha = scalar;

  size_t batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  size_t batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);

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

#if MKL_FOUND
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
#else
  for(size_t i = 0; i < batchC; ++i) {
    sgemm(transA,
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
#endif
#else
  C;
  A;
  B;
  transA;
  transB;
  beta;
  scalar;
  ABORT("You need to compile with MKL in order to use the CPU version");
#endif
}

int main() {
  const size_t m = 20, k = 10, n = 30;
  auto A = marian::make_tensor(m, k);
  auto B = marian::make_tensor(k, n);

  auto C_old = marian::make_tensor(m, n);

  // With normal path
  ProdBatchedOld(C_old,
                 A,
                 B,
                 /*transA=*/false,
                 /*transB=*/false,
                 /*beta=*/0,
                 /*scalar or alpha=*/1);
  std::cout << "Old\n" << C_old;

  // With Ruy
  auto C_ruy = marian::make_tensor(m, n);
  MulFloat(C_ruy, A, B);
  std::cout << "Ruy:\n" << C_ruy;

  return 0;
}
