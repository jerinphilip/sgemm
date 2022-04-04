#include <iostream>
#include "3rd-party/CLI11.hpp"
#include "gemm.h"
#include "tensor.h"

int main(int argc, char *argv[]) {
  size_t batchSize = 1;
  size_t M = 20, K = 10, N = 30;

  char transA = 'N';
  char transB = 'N';

  float alpha = 1.0;
  float beta  = 0.0;

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
  // clang-format on

  CLI11_PARSE(app, argc, argv);

  auto A = marian::make_tensor({batchSize, M, K});
  auto B = marian::make_tensor({batchSize, K, N});

  auto C_old = marian::make_tensor({batchSize, M, N});

  // With normal path
  ProdBatchedOld(C_old, A, B, (transA == 'T'), (transB == 'T'), beta, alpha);
  std::cout << "Old\n" << C_old;

  // With Ruy
  auto C_ruy = marian::make_tensor({batchSize, M, N});
  MulFloat(C_ruy, A, B);
  std::cout << "Ruy:\n" << C_ruy;

  return 0;
}
