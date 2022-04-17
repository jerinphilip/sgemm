#include "tensor.h"
#include <iomanip>
#include <iostream>

void pprint(std::ostream &out, float *M, size_t rows, size_t cols) {
  // Two-space indent, truncate after 4-items for readability.
  const std::string indent    = " ";
  const size_t truncate       = 4;
  const size_t precision      = 2;
  const std::string separator = " ";
  const std::string ellipsis  = "   ..   ";  // Pad-spaces hardcoded.

  auto printRow = [&](float *M, size_t i) {
    // Print the cols.
    for(size_t j = 0; j <= std::min<size_t>(cols - 1, truncate); j++) {
      // Add space only for the second element onwards.
      if(j != 0) {
        out << separator;
      }
      out << std::setprecision(precision) << std::scientific << (double)M[i * cols + j];
    }

    if(cols > truncate) {
      out << ellipsis;
      for(size_t j = std::max(cols - truncate, truncate); j < cols; j++) {
        out << separator;
        out << std::setprecision(precision) << std::scientific << (double)M[i * cols + j];
      }
    }
  };

  for(size_t i = 0; i <= std::min<size_t>(rows - 1, truncate); i++) {
    out << indent;
    printRow(M, i);
    out << "\n";
  }

  if(rows > truncate) {
    for(size_t j = 0; j < 2 * truncate + 1; j++) {
      if(j != 0)
        out << separator;
      out << ellipsis;
    }
    out << "\n";

    for(size_t i = std::max(rows - truncate, truncate); i < rows; i++) {
      out << indent;
      printRow(M, i);
      out << "\n";
    }
  }
}

std::ostream &operator<<(std::ostream &out, marian::Tensor &t) {
  float *A = t->data();

  // The input Tensor is N1 x N2 ... N_k x rows x cols
  // Treat [N1 x N2 ... N_k] as batches.
  size_t rows = t->shape()[-2];
  size_t cols = t->shape()[-1];

  // Obtain batch-count by checking how many rows*cols add up to number of elements.
  size_t batchSize = t->shape().size() / (rows * cols);

  out << "Batch: " << t->shape() << "\n";

  for(size_t b = 0; b < batchSize; b++) {
    out << "M" << b << ": "
        << "\n";

    // Offset by these many matrices.
    size_t offset = b * (rows * cols);
    float *M      = &(A[offset]);

    pprint(out, M, rows, cols);
  }

  return out;
}

namespace marian {

bool is_close(const Tensor &A, const Tensor &B) {
  if(!(A->shape() == B->shape())) {
    return false;
  }

  const float *pA = A->cdata(), *pB = B->cdata();
  const size_t N  = A->shape().size();
  const float EPS = 1e-3;

  for(size_t i = 0; i < N; i++) {
    if(std::abs(pA[i] - pB[i]) >= EPS)
      return false;
  }
  return true;
}
}  // namespace marian
