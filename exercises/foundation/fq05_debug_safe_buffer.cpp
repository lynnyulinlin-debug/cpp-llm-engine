#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

float read_checked(const std::vector<float>& values, std::size_t index) {
  if (index >= values.size()) throw std::out_of_range("buffer index out of range");
  return values[index];
}

int main() {
  const std::vector<float> values{1.0F, 2.0F, 3.0F};
  assert(read_checked(values, 1) == 2.0F);
  bool rejected = false;
  try { (void)read_checked(values, values.size()); } catch (const std::out_of_range&) { rejected = true; }
  assert(rejected);
}
