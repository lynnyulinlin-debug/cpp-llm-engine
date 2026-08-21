#include <cassert>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

std::size_t numel(const std::vector<std::size_t>& shape) {
  if (shape.empty()) throw std::invalid_argument("shape must not be empty");
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, [](std::size_t total, std::size_t dim) {
    if (dim == 0) throw std::invalid_argument("dimension must be positive");
    return total * dim;
  });
}

int main() {
  assert(numel({2, 3, 4}) == 24);
  assert(numel({1}) == 1);
  bool rejected = false;
  try { (void)numel({2, 0}); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
