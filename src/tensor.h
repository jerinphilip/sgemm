#pragma once
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
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

inline Tensor make_tensor(const std::initializer_list<size_t> &ls) {
  Tensor t = std::make_shared<_Tensor>(ls);
  return t;
}

bool is_close(const Tensor &A, const Tensor &B);

}  // namespace marian

void pprint(std::ostream &out, float *M, size_t rows, size_t cols);
std::ostream &operator<<(std::ostream &out, marian::Tensor &t);
