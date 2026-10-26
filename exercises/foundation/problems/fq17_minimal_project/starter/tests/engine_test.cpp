#include "engine.h"

#include <cassert>
#include <utility>

int main() {
  EngineStep step(1);
  assert(step.run(41) == 42);

  Workspace first(2);
  Workspace second = std::move(first);
  assert(first.size() == 0);
  assert(second.size() == 2);
  second.at(1) = 7;
  assert(second.at(1) == 7);
  return 0;
}
