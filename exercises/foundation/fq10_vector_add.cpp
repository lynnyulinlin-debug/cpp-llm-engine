#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

std::vector<float> vector_add(const std::vector<float>& left,
                              const std::vector<float>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector sizes must match");
  }

  std::vector<float> result(left.size());
  for (std::size_t i = 0; i < left.size(); ++i) result[i] = left[i] + right[i];
  return result;
}

int main() {
  const std::vector<float> left{1.0f, 2.0f, 3.0f};
  const std::vector<float> right{4.0f, 5.0f, 6.0f};
  const auto result = vector_add(left, right);
  assert(result == std::vector<float>({5.0f, 7.0f, 9.0f}));

  bool rejected = false;
  try {
    vector_add({1.0f}, {1.0f, 2.0f});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
