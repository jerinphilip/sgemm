
#include <cassert>
#include <cstdlib>
#include <iostream>
#include "gtest/gtest.h"

#include "gemm.h"
#include "tensor.h"
#include "utils.h"

using namespace marian::gemm;

TEST(Tensor, creation) {
  using namespace marian;
  Tensor A = make_tensor({3, 4});
  Tensor B = make_tensor({1, 3, 4});
  SGEMM_DEBUG(A);
  SGEMM_DEBUG(B);
  ASSERT_EQ(true, true);
}

void compare_random(Provider Provider1, Provider Provider2) {
  const size_t m = 20, k = 10, n = 30, batchSize = 2;

  // There is a beta = 0 fast special case for ruy. It is important that this
  // codepath be tested.
  const std::vector<float> betaU = {0.0, 3.0};

  // There is an alpha = 1.0 special case as well, where the product flops can
  // be ignored because identity.
  const std::vector<float> alphaU = {1.0, 2.0};

  std::vector<bool> transU = {true, false};

  for(const bool transA : transU) {
    for(const bool transB : transU) {
      marian::Tensor A, B;

      auto generate = [batchSize](size_t rows, size_t cols) {
        return marian::make_tensor({batchSize, rows, cols});
      };

      A = transA ? generate(k, m) : generate(m, k);
      B = transB ? generate(n, k) : generate(k, n);

      for(const float &beta : betaU) {
        for(const float &alpha : alphaU) {
          auto C1 = marian::make_tensor({batchSize, m, n});
          GemmBatchedDispatchByProvider(Provider1, C1, A, B, transA, transB, beta, alpha);
          SGEMM_DEBUG(C1);

          auto C2 = marian::make_tensor({batchSize, m, n});
          GemmBatchedDispatchByProvider(Provider2, C2, A, B, transA, transB, beta, alpha);
          SGEMM_DEBUG(C2);

          bool close = marian::is_close(C1, C2);
          ASSERT_EQ(close, true);
        }
      }
    }
  }
}

#if defined(MARIAN_WITH_RUY_SGEMM) && defined(MARIAN_WITH_MKL)
TEST(RuyVsMKL, Combinations) {
  compare_random(Provider::kRuy, Provider::kMKL);
}
#endif  // defined(MARIAN_WITH_RUY_SGEMM) && defined(MARIAN_WITH_MKL)

#if defined(MARIAN_WITH_RUY_SGEMM) && defined(MARIAN_WITH_EIGEN_SGEMM)
TEST(RuyVsEigen, Combinations) {
  compare_random(Provider::kRuy, Provider::kEigen);
}
#endif  // defined(MARIAN_WITH_RUY_SGEMM) && defined(MARIAN_WITH_EIGEN_SGEMM)

#if defined(MARIAN_WITH_MKL) && defined(MARIAN_WITH_EIGEN_SGEMM)
TEST(MKLVsEigen, Combinations) {
  compare_random(Provider::kMKL, Provider::kEigen);
}
#endif  //defined(MARIAN_WITH_MKL) && defined(MARIAN_WITH_EIGEN_SGEMM)

#if defined(MARIAN_WITH_BLAS) && defined(MARIAN_WITH_EIGEN_SGEMM)
TEST(BLASVsEigen, Combinations) {
  compare_random(Provider::kBLAS, Provider::kEigen);
}
#endif  //defined(MARIAN_WITH_BLAS) && defined(MARIAN_WITH_EIGEN_SGEMM)
