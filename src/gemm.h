#pragma once

// Attempt to keep as many providers as possible in the compiled library.
// Tries to follow a flat #ifdef instead of heavy nesting in this file.
// If a library can exist without conflicting, they're included. There are
// definitions switch which are propogated from corresponding CMake variables.
//
// Templates are used to keep multiple implementations ODR compatible. Routing
// happens once all the implementations are defined, based on precedence.
//
// For example, Eigen is a fallback. An x86-64 processor will have Intel MKL.
// Both of these can co-exist. Addition deletion can be done at compile time by
// controlling the respective CMake variable.
//
// 0. The simplest implementation is an ABORT, as it existed before.
// 1. We incrementally add implementations the standard GEMM API using
//    different providers.
// 2. Given a functional GEMM, BatchedGEMM can be realized by explicitly looping.
// 3. Some providers allow faster variants of BatchedGEMM by reducing
//    allocations/grouping. In this case, we explicitly specialize the templates
//    to the faster implementation.
//
// Client calls a GemmBatched, through a translation layer from marian::Tensor
// to GEMM API arguments.
//
// Units are added or removed as a whole, without interspersing ifdefs in an
// attempt to DRY. This leads to an increased verbosity, but the units are
// much more pliable.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#ifdef MARIAN_WITH_RUY_SGEMM
#include <ruy/ruy.h>
#include <ruy/system_aligned_alloc.h>
#endif  // MARIAN_WITH_RUY_SGEMM

#ifdef MARIAN_WITH_MKL
#include <mkl.h>
#endif  // MARIAN_WITH_MKL

#ifdef MARIAN_WITH_EIGEN_SGEMM
#include "Eigen/Core"
#include "Eigen/Dense"
#endif  // MARIAN_WITH_EIGEN_SGEMM

#ifdef MARIAN_WITH_BLAS
#include <cblas.h>
#endif  // MARIAN_WITH_BLAS

#include "tensor.h"

namespace marian {
namespace gemm {

// The following is about to be used further down below in templating multiple
// implementations, allowing them to exist in an ODR compatible way.
enum class Provider {
  kNone  = 0,  // Default: ABORT at runtime.
  kEigen = 1,  // Eigen Library; Portable fallback. Works on most platforms.
               // Used by  WASM
  kRuy   = 2,  // Ruy, targetting ARM. X86 etc available, but not best.
  kBLAS  = 3,  // OpenBLAS, Netlib (c)BLAS etc. Some provider which implements the BLAS API.
  kMKL   = 4,  // Intel provides MKL library which is tuned for performance.
  kARMPL = 5   // ARM Performance Library
};

// This is a temporary hack to improve readability, edit easiness.
// TODO: Fix by means of sed replacement.
#define MARIAN_GEMM_ARGS                                                                          \
  const bool transA, const bool transB, const int M, const int N, const int K, const float alpha, \
      const float *A, const int lda, const float *B, const int ldb, const float beta, float *C,   \
      const int ldc

template <enum Provider>
inline void Gemm(MARIAN_GEMM_ARGS);

template <enum Provider>
inline void GemmBatched(MARIAN_GEMM_ARGS, int batchSize);

// A marian function which dispatches to the relevant GEMM function which is
// one of the specializations of the above declaration.
//
//    C = alpha * op(A) * op(B) + beta*C
//
// transA, transB are boolean flags deciding whether to transpose the matrices
// A or B.
//
// op(A) is an M x K matrix, op(B) is a K x N matrix. Supply M, K, N
// accordingly.
//
// Intention is to replace the contents of thus function in
// browsermt/marian-dev, while working organizing providers behind these.

void ProdBatchedOld(marian::Tensor C,
                    const marian::Tensor A,
                    const marian::Tensor B,
                    bool transA,
                    bool transB,
                    float beta,
                    float alpha);

// Convenience function to dispatch GEMM by means of string provider.
void dispatch(std::string provider,
              marian::Tensor C,
              const marian::Tensor A,
              const marian::Tensor B,
              bool transA,
              bool transB,
              float beta,
              float alpha);

// TODO: We have flattened declarations/definitions. Many
// GemmBatched<provider>, Gemm<provider> exist. A sane default can be choosen
// at compile-time by adjusting precedence in the sequence below.

// clang-format off
static const Provider kChosenProvider = std::max({
       Provider::kNone
#ifdef MARIAN_WITH_RUY_SGEMM
     , Provider::kRuy
#endif  // MARIAN_WITH_RUY_SGEMM
#ifdef MARIAN_WITH_MKL
    , Provider::kMKL
#endif  // MARIAN_WITH_MKL
#ifdef MARIAN_WITH_EIGEN_SGEMM
    , Provider::kEigen
#endif  // MARIAN_WITH_EIGEN_SGEMM
#ifdef MARIAN_WITH_BLAS
    , Provider::kBLAS
#endif  // MARIAN_WITH_BLAS
});
// clang-format on

#ifndef SGEMM_IMPL_
#define SGEMM_IMPL_
#include "gemm-impl.cpp"
#endif

#undef MARIAN_GEMM_ARGS

}  // namespace gemm
}  // namespace marian
