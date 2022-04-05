
#include <cassert>
#include <iostream>
#include "gtest/gtest.h"

#include "gemm.h"
#include "tensor.h"

TEST(Tensor, creation) {
  using namespace marian;
  Tensor A = make_tensor({3, 4});
  Tensor B = make_tensor({1, 3, 4});
  std::cout << A << std::endl;
  ASSERT_EQ(true, true);
}

void Batched(size_t batchSize) {
  const size_t m = 20, k = 10, n = 30;
  auto A = marian::make_tensor({batchSize, m, k});
  auto B = marian::make_tensor({batchSize, k, n});

  auto C_old = marian::make_tensor({batchSize, m, n});

  // With normal path
  ProdBatchedOld(C_old,
                 A,
                 B,
                 /*transA=*/false,
                 /*transB=*/false,
                 /*beta=*/0,
                 /*scalar or alpha=*/1);
  // std::cout << "Old\n" << C_old;

  // With Ruy
  auto C_ruy = marian::make_tensor({batchSize, m, n});
  gemmRuy(C_ruy,
          A,
          B,
          /*transA=*/false,
          /*transB=*/false,
          /*beta=*/0,
          /*scalar or alpha=*/1);
  // std::cout << "Ruy:\n" << C_ruy;

  using marian::is_close;
  ASSERT_EQ(is_close(C_ruy, C_old), true);
}

TEST(RuyVsBLAS, SingleSamplePseudoBatch) {
  Batched(1);
}

TEST(RuyVsBLAS, Batched) {
  Batched(100);
}

TEST(RuyVsBLAS, Combinations) {
  const size_t m = 20, k = 10, n = 30, batchSize = 2;
  const float alpha = 2.0, beta = 3.0;

  std::vector<bool> transU = {true, false};
  for(const bool transA : transU) {
    for(const bool transB : transU) {
      marian::Tensor A, B;

      auto generate = [batchSize](size_t rows, size_t cols) {
        return marian::make_tensor({batchSize, rows, cols});
      };

      A = transA ? generate(k, m) : generate(m, k);
      B = transB ? generate(n, k) : generate(k, n);

      auto C_blas = marian::make_tensor({batchSize, m, n});

      // With normal path
      ProdBatchedOld(C_blas, A, B, transA, transB, beta, alpha);
      // std::cerr << "BLAS: " << C_blas << std::endl;

      // With Ruy
      auto C_ruy = marian::make_tensor({batchSize, m, n});
      gemmRuy(C_ruy, A, B, transA, transB, beta, alpha);
      // std::cerr << "Ruy: " << C_ruy << std::endl;

      bool close = marian::is_close(C_ruy, C_blas);
      ASSERT_EQ(close, true);
    }
  }
}
