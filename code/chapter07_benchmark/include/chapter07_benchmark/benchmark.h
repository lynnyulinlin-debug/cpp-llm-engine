#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace chapter07_benchmark {

struct BenchmarkResult {
    std::string name;
    std::size_t iterations = 0;
    double elapsed_ms = 0.0;
    double items_per_second = 0.0;
};

struct BenchmarkSample {
    std::string phase;
    std::size_t tokens = 0;
    double elapsed_ms = 0.0;
};

BenchmarkResult make_result(const std::string& name, std::size_t iterations, double elapsed_ms);
double tokens_per_second(std::size_t tokens, double elapsed_ms);
std::vector<BenchmarkSample> sample_generation_path();

}  // namespace chapter07_benchmark
