
#include <iostream>
#include <random>
#include "3rd-party/CLI11.hpp"
#include "gemm.h"
#include "tensor.h"
#include "utils.h"

int main(int argc, char *argv[]) {
  size_t batchSize = 1;
  std::mt19937_64 gen64;

  size_t maxM = 128, maxK = 128, maxN = 128;
  size_t minM = 64, minK = 64, minN = 64;
  int seed = 42;

  float alpha       = 3.0;
  float beta        = 1.0;
  size_t iterations = 100;

  std::string provider("ruy");

  CLI::App app{"App description"};

  // clang-format off
  app.add_option("--batchSize"  , batchSize  , "Batches to pack together");
  app.add_option("--rowsA"      , maxM       , "Rows of A, must be greater than 64");
  app.add_option("--colsA"      , maxK       , "Cols of A = rows B, must be greater than 64");
  app.add_option("--colsB"      , maxN       , "Cols of B, must be greater than 64");
  app.add_option("--provider"   , provider   , "SGGEMM provider");
  app.add_option("--iterations" , iterations , "Number of iterations");
  app.add_option("--seed"       , seed       , "Seed for random init");
  // clang-format on

  CLI11_PARSE(app, argc, argv);

  gen64.seed(seed);

  marian::Tensor A, B;
  for(size_t i = 0; i < iterations; i++) {
    std::vector<bool> transU = {true, false};
    for(const bool transA : transU) {
      for(const bool transB : transU) {
        marian::Tensor A, B;

        size_t M, N, K;
        // std::cout << M << "x" << K << ", " << K << "x" << N << "\n";
        M = minM + gen64() % (maxM - minM + 1);
        N = minN + gen64() % (maxN - minM + 1);
        K = minN + gen64() % (maxK - minM + 1);

        auto generate = [batchSize](size_t rows, size_t cols) {
          return marian::make_tensor({batchSize, rows, cols});
        };

        A = transA ? generate(K, M) : generate(M, K);
        B = transB ? generate(N, K) : generate(K, N);

        auto C = marian::make_tensor({batchSize, M, N});
        marian::gemm::dispatch(provider, C, A, B, transA, transB, beta, alpha);
        SGEMM_DEBUG(C);
      }
    }
  }

  return 0;
}
