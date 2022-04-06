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
// attempt to DRY. This leads to an increased verbosity, much the units are
// much more pliable.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#ifdef MARIAN_USE_RUY_SGEMM
#include <ruy/ruy.h>
#include <ruy/system_aligned_alloc.h>
#endif  // MARIAN_USE_RUY_SGEMM

#ifdef MARIAN_USE_MKL
#include <mkl.h>
#endif  // MARIAN_USE_MKL

#ifdef MARIAN_USE_EIGEN_SGEMM
#include "Eigen/Core"
#include "Eigen/Dense"
#endif  // MARIAN_USE_EIGEN_SGEMM

#ifdef MARIAN_USE_BLAS
#include <cblas.h>
#endif  // MARIAN_USE_BLAS

#include "tensor.h"

namespace marian {
namespace gemm {

// The following is about to be used further down below in templating multiple
// implementations, allowing them to exist in an ODR compatible way.
enum class Provider {
  kNone,
  kEigen,  // Eigen Library; Portable fallback. Works on most platforms. Used by
           // WASM
  kMKL,    //
  kBLAS,   //
  kRuy,    // Ruy, targetting ARM. X86 etc available, but not best.
  kARMPL   // ARM Performance Library
};

// A marian connected GEMM function. Arguments are in the order of the
// expression being evaluated:
//
//    C = alpha * op(A) * op(B) + beta*C
//
// transA, transB are boolean flags deciding whether to transpose the matrices
// A or B.
//
// op(A) is an M x K matrix, op(B) is a K x N matrix. Supply M, K, N
// accordingly.
//

// This is a temporary hack to improve readability.
// TODO: Fix by means of sed replacement.
#define MARIAN_GEMM_ARGS                                                                          \
  const bool transA, const bool transB, const int M, const int N, const int K, const float alpha, \
      const float *A, const int lda, const float *B, const int ldb, const float beta, float *C,   \
      const int ldc

// The following two are configured to ABORT (Provider = kNone).
template <enum Provider>
inline void Gemm(MARIAN_GEMM_ARGS);

template <enum Provider>
inline void GemmBatched(MARIAN_GEMM_ARGS, int batchSize);

void ProdBatchedOld(marian::Tensor C,
                    const marian::Tensor A,
                    const marian::Tensor B,
                    bool transA,
                    bool transB,
                    float beta,
                    float alpha);

void dispatch(std::string provider,
              marian::Tensor C,
              const marian::Tensor A,
              const marian::Tensor B,
              bool transA,
              bool transB,
              float beta,
              float alpha);

#ifndef SGEMM_IMPL_
#define SGEMM_IMPL_
#include "gemm-impl.cpp"
#endif

#undef MARIAN_GEMM_ARGS

}  // namespace gemm
}  // namespace marian
