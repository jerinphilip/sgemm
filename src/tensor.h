#pragma once
#include <cstdlib>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include "utils.h"

#ifdef _WIN32
#include <malloc.h>
#endif

namespace marian {

const size_t ALIGNMENT = 256;

#ifdef _WIN32
inline void *genericMalloc(size_t alignment, size_t size) {
  void *ret = _aligned_malloc(size, alignment);
  ABORT_IF(!ret, "Failed to allocate memory on CPU");
  return ret;
}
inline void genericFree(void *ptr) {
  _aligned_free(ptr);
}
#else
// Linux and OS X.  There is no fallback to malloc because we need it to be aligned.
inline void *genericMalloc(size_t alignment, size_t size) {
  // On macos, aligned_alloc is available only on c++17
  // Furthermore, it requires that the memory requested is an exact multiple of the alignment, otherwise it fails.
  // posix_memalign is available both Mac (Since 2016) and Linux and in both gcc and clang
  void *result;
  // Error could be detected by return value or just remaining nullptr.
  ABORT_IF(posix_memalign(&result, alignment, size), "Failed to allocate memory on CPU");
  return result;
}

inline void genericFree(void *ptr) {
  free(ptr);
}
#endif

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
    for(int i = 1; i < shape_.size(); ++i) {
      strm << "x" << shape_[i];
    }
    strm << " size=" << shape_.size();
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
  _Tensor(const std::initializer_list<size_t> &ls) : shape_(ls) {
    data_ = genericMalloc(ALIGNMENT, sizeof(float) * shape_.size());
    std::fill(data(), data() + shape_.size(), 1);
  }
  const Shape &shape() const { return shape_; }
  const float *cdata() const { return reinterpret_cast<const float *>(data_); }
  float *data() { return reinterpret_cast<float *>(data_); }

  _Tensor(const _Tensor &) = delete;
  _Tensor &operator=(const _Tensor &) = delete;

  ~_Tensor() { genericFree(data_); }

private:
  Shape shape_;
  void *data_;
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
