#ifndef SGEMM_IMPL_
#error "This is an implementation file and should not be included directly."
#endif

// Base case: ABORT; Converts compile time errors into runtime errors.
// This is also a future hook to replace with a standard-CPP implementation if
// necessary.
template <enum Provider = Provider::kNone>
inline void Gemm(const bool transA,
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
                 const int ldc) {
  ABORT("No available GEMM Implementation;");
}

template <enum Provider = Provider::kNone>
inline void GemmBatched(const bool transA,
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
                        int batchSize) {
  ABORT("No available GEMM (Batched) Implementation;");
}

// EIGEN Specializations
#ifdef MARIAN_WITH_EIGEN_SGEMM

// Minimum definitions required for the PyTorch import to work. Taken from:
// https://github.com/pytorch/pytorch/blob/936e7eabcabc97fbc40f488e67a94c4733c33dd6/caffe2/utils/eigen_utils.h
using EigenOuterStride = Eigen::OuterStride<Eigen::Dynamic>;
template <typename T>
using EigenOuterStridedMatrixMap
    = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenOuterStride>;
template <typename T>
using ConstEigenOuterStridedMatrixMap
    = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenOuterStride>;

template <>
inline void Gemm<Provider::kEigen>(const bool transA,
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
                                   const int ldc) {
  // Taken from https://github.com/pytorch/pytorch/blob/0d7aad822e77b9f5ca9649114b6c2cbdf54564e3/caffe2/utils/math_cpu.cc#L155
  // PyTorch. BSD License.
  EigenOuterStridedMatrixMap<float> C_mat(C, N, M, EigenOuterStride(ldc));
  if(beta == 0) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }

  if(transA) {
    if(transB) {
      C_mat.noalias()
          += alpha
             * (ConstEigenOuterStridedMatrixMap<float>(B, K, N, EigenOuterStride(ldb)).transpose()
                * ConstEigenOuterStridedMatrixMap<float>(A, M, K, EigenOuterStride(lda))
                      .transpose());
    } else {
      C_mat.noalias() += alpha
                         * (ConstEigenOuterStridedMatrixMap<float>(B, N, K, EigenOuterStride(ldb))
                            * ConstEigenOuterStridedMatrixMap<float>(A, M, K, EigenOuterStride(lda))
                                  .transpose());
    }
  } else {
    if(transB) {
      C_mat.noalias()
          += alpha
             * (ConstEigenOuterStridedMatrixMap<float>(B, K, N, EigenOuterStride(ldb)).transpose()
                * ConstEigenOuterStridedMatrixMap<float>(A, K, M, EigenOuterStride(lda)));
    } else {
      C_mat.noalias()
          += alpha
             * (ConstEigenOuterStridedMatrixMap<float>(B, N, K, EigenOuterStride(ldb))
                * ConstEigenOuterStridedMatrixMap<float>(A, K, M, EigenOuterStride(lda)));
    }
  }
}
#endif  // MARIAN_WITH_EIGEN_SGEMM

#ifdef MARIAN_WITH_BLAS
template <>
inline void Gemm<Provider::kBLAS>(const bool transA,
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
                                  const int ldc) {
  // Converting booleans to CBLAS_TRANSPOSE (char).
  CBLAS_TRANSPOSE cTransA = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE cTransB = transB ? CblasTrans : CblasNoTrans;
  cblas_sgemm(CblasRowMajor, cTransA, cTransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif  // MARIAN_WITH_BLAS

#ifdef MARIAN_WITH_RUY_SGEMM
namespace {

// AlignedVector allocates aligned memory and cleans up after itself. RAII
// wrapper similar to intgemm's AlignedVector.
template <class T>
class AlignedVector {
public:
  AlignedVector(size_t num_elem)
      : size_(num_elem),
        storage_(reinterpret_cast<T *>(ruy::detail::SystemAlignedAlloc(sizeof(T) * num_elem))) {}

  T *begin() { return storage_; }
  T *data() { return storage_; }
  size_t size() const { return size_; }
  size_t memSize() const { return sizeof(T) * size_; }

  // Forbid copy
  AlignedVector(const AlignedVector &) = delete;
  AlignedVector &operator=(const AlignedVector &) = delete;

  ~AlignedVector() { ruy::detail::SystemAlignedFree(reinterpret_cast<void *>(storage_)); }

private:
  T *storage_;
  size_t size_;
};

}  // namespace

template <>
inline void Gemm<Provider::kRuy>(const bool transA,
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
                                 const int ldc) {
  ruy::Context context;

  // If we need to transpose, we can swap dimensions in layout claim the matrix
  // is just column-major. Set ordering so transpose.
  const auto orderA = (transA ? ruy::Order::kColMajor : ruy::Order::kRowMajor);
  const auto orderB = (transB ? ruy::Order::kColMajor : ruy::Order::kRowMajor);

  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(M, K, orderA, lhs.mutable_layout());
  lhs.set_data(A);

  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(K, N, orderB, rhs.mutable_layout());
  rhs.set_data(B);

  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(M, N, ruy::Order::kRowMajor, dst.mutable_layout());

  if(beta == 0) {
    // For beta = 0, we want to avoid the additional allocation. This is a
    // large amount of our inference use-cases. sgemm is called with `beta` for
    // accumulating gradients in backpropogation, which is 0.0 during
    // inference.

    dst.set_data(C);
    ruy::MulParams<float, float> mul_params;
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);

    if(alpha != 1.0) {
      // Write out C as C = alpha * [op(A) * op(B)] + beta * C
      // Can we expect the compiler to autovectorize this?
      // TODO: Come back and explicitly use SIMD.
      const size_t size    = M * N;
      const float *opA_opB = C;  // Alias.
#pragma clang loop vectorize(enable) interleave(enable)
      for(size_t i = 0; i < size; i++) {
        C[i] = alpha * opA_opB[i];
      }
    }
  } else {
    // @jerinphilip has not yet been able to find a ruy primitive that does in
    // place addition to obtain full gemm.
    //
    // Safe bet is to make an additional allocation to store the result of
    // multiply  and use the existing values in C.
    //
    // See also: https://github.com/google/ruy/issues/307

    AlignedVector<float> intermediate(M * N);
    dst.set_data(intermediate.data());
    ruy::MulParams<float, float> mul_params;
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);

    // Write out C as C = alpha * [op(A) * op(B)] + beta * C
    // Can we expect the compiler to autovectorize this?
    // TODO: Come back and explicitly use SIMD.
    const size_t size    = M * N;
    const float *opA_opB = intermediate.data();
#pragma clang loop vectorize(enable) interleave(enable)
    for(size_t i = 0; i < size; i++) {
      C[i] = alpha * opA_opB[i] + beta * C[i];
    }
  }
}

#endif

#ifdef MARIAN_WITH_MKL

// MKL provides cblas_sgemm_batch optimized for performance on intel hardware.
template <>
inline void GemmBatched<Provider::kMKL>(const bool transA,
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
                                        int batchSize) {
  CBLAS_TRANSPOSE trans_A = transA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE trans_B = transB ? CblasTrans : CblasNoTrans;

  /* cblas_sgemm_batch allows us to group all the small GEMMs that are done in a
   * for loop with sgemm and compute them in only one MKL call. 
   * The API supports dependencies, where you can specify one "group" of GEMMs
   * to be computed after another. (This controlled by the group_count
   * parameter). In our case, the operations are not dependent on one another so
   * we hardcode one group. The rest of the arguments (with the exception of
   * group_size) are the same as the ones that cblas_sgemm expects, with the
   * difference that we are supposed to provide an array pointer (One element
   * per group).
   */

  const MKL_INT mM(M), mN(N), mK(K);
  const MKL_INT mlda(lda), mldb(ldb), mldc(ldc);
  const MKL_INT strideB(mN * mK), strideA(mM * mK), strideC(mN * mM);
  const MKL_INT mBatchSize(batchSize);

  std::vector<const float *> A_array(batchSize, nullptr);
  std::vector<const float *> B_array(batchSize, nullptr);

  std::vector<float *> C_array(batchSize, nullptr);

  for(int i = 0; i < batchSize; ++i) {
    A_array[i] = A + i * strideA;
    B_array[i] = B + i * strideB;
    C_array[i] = C + i * strideC;
  }

  cblas_sgemm_batch(CblasRowMajor,
                    &trans_A,
                    &trans_B,
                    &mM,
                    &mN,
                    &mK,
                    &alpha,
                    A_array.data(),
                    &mlda,
                    B_array.data(),
                    &mldb,
                    &beta,
                    C_array.data(),
                    &mldc,
                    /*group_count=*/1,
                    &mBatchSize);
}
#endif  // MARIAN_WITH_MKL \

// GemmBatched<> by means of looping Gemm<>, applies to kEigen and kBLAS which
// doesn't have a more optimal batched-variant.
#define __UNROLL(provider)                             \
  template <>                                          \
  inline void GemmBatched<provider>(const bool transA, \
                                    const bool transB, \
                                    const int M,       \
                                    const int N,       \
                                    const int K,       \
                                    const float alpha, \
                                    const float *A,    \
                                    const int lda,     \
                                    const float *B,    \
                                    const int ldb,     \
                                    const float beta,  \
                                    float *C,          \
                                    const int ldc,     \
                                    int batchSize) {   \
    size_t strideA = M * K;                            \
    size_t strideB = K * N;                            \
    size_t strideC = M * N;                            \
                                                       \
    for(size_t i = 0; i < batchSize; ++i) {            \
      Gemm<provider>(transA,                           \
                     transB,                           \
                     M,                                \
                     N,                                \
                     K,                                \
                     alpha,                            \
                     A + i * strideA,                  \
                     lda,                              \
                     B + i * strideB,                  \
                     ldb,                              \
                     beta,                             \
                     C + i * strideC,                  \
                     ldc);                             \
    }                                                  \
  }

// Applied to kBLAS
#ifdef MARIAN_WITH_BLAS
__UNROLL(Provider::kBLAS);
#endif  // MARIAN_WITH_BLAS

// Applied to kEigen
#ifdef MARIAN_WITH_EIGEN_SGEMM
__UNROLL(Provider::kEigen);
#endif  // MARIAN_WITH_EIGEN_SGEMM

// Applied to kRuy
#ifdef MARIAN_WITH_RUY_SGEMM
__UNROLL(Provider::kRuy);
#endif  // MARIAN_WITH_RUY_SGEMM

#undef __UNROLL
