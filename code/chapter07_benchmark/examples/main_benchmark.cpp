#include "chapter07_benchmark/benchmark.h"
#include "chapter07_benchmark/test_helpers.h"

#include <iostream>

int main() {
    const auto result = chapter07_benchmark::make_result("demo", 100, 25.0);
    const auto samples = chapter07_benchmark::sample_generation_path();

    std::cout << "[chapter07] benchmark demo\n";
    std::cout << "unit checks: " << std::boolalpha << chapter07_benchmark::run_unit_checks() << '\n';
    std::cout << "e2e checks: " << std::boolalpha << chapter07_benchmark::run_e2e_checks() << '\n';
    std::cout << "result name: " << result.name << '\n';
    std::cout << "result items/s: " << result.items_per_second << '\n';
    std::cout << "samples: " << samples.size() << '\n';
    std::cout << chapter07_benchmark::render_summary();
    return 0;
}
