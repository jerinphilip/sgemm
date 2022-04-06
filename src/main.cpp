#include <iostream>
#include "3rd-party/CLI11.hpp"
#include "gemm.h"
#include "tensor.h"

using namespace marian::gemm;

int main(int argc, char *argv[]) {
  size_t batchSize = 1;
  size_t M = 20, K = 10, N = 30;

  char transA = 'N';
  char transB = 'N';

  float alpha = 1.0;
  float beta  = 0.0;

  std::string provider = "ruy";

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
  app.add_option("--provider"  , provider  , "Backend to use. ruy | eigen | mkl | blas");
  // clang-format on

  CLI11_PARSE(app, argc, argv);

  marian::Tensor A, B, C;

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

  C = marian::make_tensor({batchSize, M, N});
  dispatch(provider, C, A, B, (transA == 'T'), (transB == 'T'), alpha, beta);
  std::cout << C << std::endl;

  return 0;
}
