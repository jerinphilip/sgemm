#include "gemm.h"

namespace marian {
namespace gemm {

Provider EnvironmentProvider() {
#if defined(_MSC_VER)
  char env_override[11];
  size_t len = 0;
  if(getenv_s(&len, env_override, sizeof(env_override), "MARIAN_SGEMM_PROVIDER"))
    return kHighestProvider;
  if(!len)
    return kHighestProvider;
#else
  const char *env_override = getenv("MARIAN_SGEMM_PROVIDER");
  if(!env_override)
    return kHighestProvider; /* This will be capped to actual ID */
#endif
  if(!strcmp(env_override, "EIGEN"))
    return Provider::kEigen;
  if(!strcmp(env_override, "RUY"))
    return Provider::kRuy;
  if(!strcmp(env_override, "BLAS"))
    return Provider::kBLAS;
  if(!strcmp(env_override, "MKL"))
    return Provider::kMKL;
  fprintf(stderr, "Ignoring unrecognized MARIAN_SGEMM_PROVIDER %s\n", env_override);
  return kHighestProvider;
}

void ProdBatchedOld(marian::Tensor C,
                    const marian::Tensor A,
                    const marian::Tensor B,
                    bool transA,
                    bool transB,
                    float beta,
                    float alpha) {
  static const Provider kEnvironmentProvider = EnvironmentProvider();

  Provider kChosenProvider = std::min(kHighestProvider, kEnvironmentProvider);

  GemmBatchedDispatchByProvider(kChosenProvider, C, A, B, transA, transB, beta, alpha);
}

void GemmBatchedDispatchByProvider(Provider provider,
                                   marian::Tensor C,
                                   const marian::Tensor A,
                                   const marian::Tensor B,
                                   bool transA,
                                   bool transB,
                                   float beta,
                                   float alpha) {
  // Infer GEMM parameters from marian::Tensor and transpose information

  size_t M, N, K, batchSize, lda, ldb, ldc;
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

  lda = A->shape()[-1];
  ldb = B->shape()[-1];

  batchSize = A->shape().size() / (M * K);

  ldc = N;

  // Dispatch to the relevant GEMM function.
  void (*gemmFn)(const bool transA,
                 const bool transB,
                 const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float *A,
                 const int lda,
                 const float *B,
                 const int ldb,
                 const float beta,
                 float *C,
                 const int ldc,
                 int batchSize)
      = nullptr;

  switch(provider) {
    case Provider::kRuy: gemmFn = &GemmBatched<Provider::kRuy>; break;
    case Provider::kMKL: gemmFn = &GemmBatched<Provider::kMKL>; break;
    case Provider::kBLAS: gemmFn = &GemmBatched<Provider::kBLAS>; break;
    case Provider::kEigen: gemmFn = &GemmBatched<Provider::kEigen>; break;
    default: ABORT("Unknown Gemm Provider {}", (int)provider); break;
  }

  // Make call
  gemmFn(transA,
         transB,
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
         ldc,
         batchSize);
}

}  // namespace gemm
}  // namespace marian
