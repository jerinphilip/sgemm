
#include <cassert>
#include <iostream>
#include "gtest/gtest.h"

#include "tensor.h"

TEST(Basic, Basic) {
  using namespace marian;
  Tensor A = make_tensor({3, 4});
  Tensor B = make_tensor({1, 3, 4});
  std::cout << A << std::endl;
  ASSERT_EQ(true, true);
}
