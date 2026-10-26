#include "engine.h"

#include <iostream>

int main() {
  EngineStep step(1);
  std::cout << "result: " << step.run(41) << '\n';
  return 0;
}
