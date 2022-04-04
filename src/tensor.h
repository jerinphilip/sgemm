#pragma once
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include "utils.h"

namespace marian {

// A replacement for marian::Shape. Mostly copied over.
struct Shape {
  Shape(const std::initializer_list<size_t> &ls) : shape_(ls) {}

  size_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
  }

  size_t operator[](int i) const {
    if(i >= 0) {
      ABORT_IF(i >= (int)shape_.size(),
               "Index {} is out of bounds, shape {} has {} dimension",
               i,
               std::string(*this),
               shape_.size());
      return shape_[i];
    } else {
      ABORT_IF((int)shape_.size() + i < 0,
               "Negative index {} is out of bounds, shape {} has {} dimension",
               i,
               std::string(*this),
               shape_.size());
      return shape_[shape_.size() + i];
    }
  }

  std::string toString() const {
    std::stringstream strm;
    strm << "shape=" << shape_[0];
    const size_t sz = size();
    for(int i = 1; i < sz; ++i) {
      strm << "x" << shape_[i];
    }
    strm << " size=" << sz;
    return strm.str();
  }

  friend std::ostream &operator<<(std::ostream &strm, const Shape &shape) {
    strm << shape.toString();
    return strm;
  }

  operator std::string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  bool operator==(const Shape &other) const {
    if(shape_.size() != other.shape_.size()) {
      return false;
    }

    for(size_t i = 0; i < shape_.size(); i++) {
      if(shape_[i] != other.shape_[i]) {
        return false;
      }
    }
    return true;
  }

private:
  // Dimensions of the matrix. Something like {Batch, Rows, Columns}, or {Rows, Columns}
  std::vector<size_t> shape_;
};

// Minimum stub
struct _Tensor {
public:
  _Tensor(const std::initializer_list<size_t> &ls) : shape_(ls), data_(shape_.size()) {
    std::fill(data_.begin(), data_.end(), 1);
  }
  const Shape &shape() const { return shape_; }
  const float *cdata() const { return data_.data(); }
  float *data() { return data_.data(); }

private:
  Shape shape_;
  std::vector<float> data_;
};

typedef std::shared_ptr<_Tensor> Tensor;

Tensor make_tensor(const std::initializer_list<size_t> &ls) {
  Tensor t = std::make_shared<_Tensor>(ls);
  return t;
}

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
