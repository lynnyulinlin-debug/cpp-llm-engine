#include "chapter07_benchmark/benchmark.h"

#include <numeric>

namespace chapter07_benchmark {

BenchmarkResult make_result(const std::string& name, std::size_t iterations, double elapsed_ms) {
    BenchmarkResult result;
    result.name = name;
    result.iterations = iterations;
    result.elapsed_ms = elapsed_ms;
    result.items_per_second = iterations > 0 && elapsed_ms > 0.0
                                  ? (static_cast<double>(iterations) * 1000.0 / elapsed_ms)
                                  : 0.0;
    return result;
}

double tokens_per_second(std::size_t tokens, double elapsed_ms) {
    if (tokens == 0 || elapsed_ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(tokens) * 1000.0 / elapsed_ms;
}

std::vector<BenchmarkSample> sample_generation_path() {
    return {
        {"prefill", 5, 0.40},
        {"decode", 2, 0.20},
        {"benchmark", 7, 0.60},
    };
}

}  // namespace chapter07_benchmark
