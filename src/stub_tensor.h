#include <iostream>

namespace marian {

struct Shape {
  Shape(size_t rows, size_t cols) : rows_(rows), cols_(cols) {}
  size_t elements() const { return rows_ * cols_; }
  size_t operator[](int dimension) const {
    if (dimension == 0)
      return rows_;
    if (dimension == 1)
      return cols_;
    // TODO(jerin): Fix this
    if (dimension == -1)
      return rows_;
    if (dimension == -2)
      return cols_;

    return -1;
  }

private:
  size_t rows_;
  size_t cols_;
};

// Minimum stub
struct _Tensor {
public:
  _Tensor(size_t rows, size_t cols) : shape_(rows, cols), data_(rows * cols) {
    std::fill(data_.begin(), data_.end(), 1);
  }
  const Shape &shape() { return shape_; }
  float *data() { return data_.data(); }

private:
  Shape shape_;
  std::vector<float> data_;
};

typedef std::shared_ptr<_Tensor> Tensor;

Tensor make_tensor(size_t rows, size_t cols) {
  return std::make_shared<_Tensor>(rows, cols);
}

} // namespace marian

std::ostream &operator<<(std::ostream &out, marian::Tensor &t) {
  float *A = t->data();
  size_t rows = t->shape()[0];
  size_t cols = t->shape()[1];
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (j != 0) {
        out << " ";
      }
      out << A[i * cols + j];
    }
    out << "\n";
  }
  return out;
}
