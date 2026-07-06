#include "chapter07_benchmark/test_helpers.h"

#include "chapter07_benchmark/benchmark.h"

#include <sstream>

namespace chapter07_benchmark {

bool run_unit_checks() {
    const BenchmarkResult result = make_result("unit", 10, 2.0);
    return result.name == "unit" && result.iterations == 10 && result.items_per_second > 0.0;
}

bool run_e2e_checks() {
    const auto samples = sample_generation_path();
    return samples.size() == 3 && samples.front().phase == "prefill" && samples.back().phase == "benchmark";
}

std::string render_summary() {
    const auto samples = sample_generation_path();
    std::ostringstream oss;
    oss << "chapter07 benchmark summary\n";
    for (const auto& sample : samples) {
        oss << sample.phase << ": tokens=" << sample.tokens << ", elapsed_ms=" << sample.elapsed_ms << '\n';
    }
    return oss.str();
}

}  // namespace chapter07_benchmark
