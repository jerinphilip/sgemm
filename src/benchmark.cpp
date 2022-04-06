
#include <iostream>
#include "3rd-party/CLI11.hpp"
#include "gemm.h"
#include "tensor.h"

void dispatch(std::string provider,
              marian::Tensor C,
              const marian::Tensor A,
              const marian::Tensor B,
              bool transA,
              bool transB,
              float beta,
              float alpha) {
  using namespace marian::gemm;
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

int main(int argc, char *argv[]) {
  size_t batchSize = 1;
  size_t M = 2000, K = 1000, N = 3000;

  char transA = 'N';
  char transB = 'N';

  float alpha = 1.0;
  float beta  = 0.0;

  std::string provider("ruy");

  CLI::App app{"App description"};

  // clang-format off
  app.add_option("--batchSize" , batchSize , "Batches to pack together");
  app.add_option("--rowsA"     , M         , "Rows of A");
  app.add_option("--colsA"     , K         , "Cols of A = rows B");
  app.add_option("--colsB"     , N         , "Cols of B");
  app.add_option("--alpha"     , alpha     , "alpha value");
  app.add_option("--beta"      , beta      , "beta value");
  app.add_option("--transA"    , transA    , "Transpose A?");
  app.add_option("--transB"    , transB    , "Transpose B?");
  app.add_option("--provider"  , provider  , "SGGEMM provider");
  // clang-format on

  CLI11_PARSE(app, argc, argv);

  marian::Tensor A, B;

  switch(transA) {
    case 'T': A = marian::make_tensor({batchSize, K, M}); break;
    case 'N': A = marian::make_tensor({batchSize, M, K}); break;
    default: ABORT("Unknown transA argument {}.", transA); break;
  }

  switch(transB) {
    case 'T': B = marian::make_tensor({batchSize, N, K}); break;
    case 'N': B = marian::make_tensor({batchSize, K, N}); break;
    default: ABORT("Unknown transB argument {}.", transA); break;
  }

  auto C = marian::make_tensor({batchSize, M, N});

  dispatch(provider, C, A, B, (transA == 'T'), (transB == 'T'), beta, alpha);

  // std::cout << C << std::endl;

  return 0;
}
