#include "gemm.h"
#include "tensor.h"

#include <iostream>

int main() {
  const size_t m = 20, k = 10, n = 30;
  auto A = marian::make_tensor({1, m, k});
  auto B = marian::make_tensor({1, k, n});

  auto C_old = marian::make_tensor({1, m, n});

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
  auto C_ruy = marian::make_tensor({1, m, n});
  MulFloat(C_ruy, A, B);
  std::cout << "Ruy:\n" << C_ruy;

  return 0;
}
