#pragma once
#include <iostream>
#include <numeric>
#include <sstream>
#include "utils.h"

namespace marian {

struct Shape {
  Shape(const std::initializer_list<size_t> &ls) : shape_(ls) {}

  size_t size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
  }

  size_t operator[](int i) const {
    if(i >= 0) {
      ABORT_IF(i >= (int)size(),
               "Index {} is out of bounds, shape {} has {} dimension",
               i,
               std::string(*this),
               size());
      return shape_[i];
    } else {
      ABORT_IF((int)size() + i < 0,
               "Negative index {} is out of bounds, shape {} has {} dimension",
               i,
               std::string(*this),
               size());
      return shape_[size() + i];
    }
  }

  std::string toString() const {
    std::stringstream strm;
    strm << "shape=" << (*this)[0];
    for(int i = 1; i < size(); ++i)
      strm << "x" << (*this)[i];
    strm << " size=" << size();
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

private:
  // Dimensions of the matrix. Something like {Batch, Rows, Columns}, or {Rows, Columns}
  std::vector<size_t> shape_;
};

// Minimum stub
struct _Tensor {
public:
  _Tensor(const std::initializer_list<size_t> &ls) : shape_(ls), data_(ls.size()) {
    std::fill(data_.begin(), data_.end(), 1);
  }
  const Shape &shape() { return shape_; }
  float *data() { return data_.data(); }

private:
  Shape shape_;
  std::vector<float> data_;
};

typedef std::shared_ptr<_Tensor> Tensor;

Tensor make_tensor(const std::initializer_list<size_t> &ls) {
  return std::make_shared<_Tensor>(ls);
}

}  // namespace marian

std::ostream &operator<<(std::ostream &out, marian::Tensor &t) {
  float *A    = t->data();
  size_t rows = t->shape()[0];
  size_t cols = t->shape()[1];
  for(size_t i = 0; i < rows; i++) {
    for(size_t j = 0; j < cols; j++) {
      if(j != 0) {
        out << " ";
      }
      out << A[i * cols + j];
    }
    out << "\n";
  }
  return out;
}
