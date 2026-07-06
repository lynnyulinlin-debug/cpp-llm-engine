#include "chapter07_benchmark/benchmark.h"
#include "chapter07_benchmark/test_framework.h"
#include "chapter07_benchmark/test_helpers.h"

#include <cmath>
#include <string>

namespace {

using chapter07_benchmark::BenchmarkResult;

bool test_make_result() {
    const BenchmarkResult result = chapter07_benchmark::make_result("demo", 100, 25.0);
    return result.name == "demo" &&
           result.iterations == 100 &&
           std::abs(result.items_per_second - 4000.0) < 1e-5;
}

bool test_tokens_per_second() {
    return std::abs(chapter07_benchmark::tokens_per_second(10, 2.0) - 5000.0) < 1e-5;
}

bool test_unit_checks() {
    return chapter07_benchmark::run_unit_checks();
}

bool test_e2e_checks() {
    return chapter07_benchmark::run_e2e_checks();
}

bool test_summary_text() {
    return chapter07_benchmark::render_summary().find("prefill") != std::string::npos;
}

}  // namespace

int main() {
    const auto summary = chapter07_benchmark::run_test_suite({
        {"make_result", &test_make_result},
        {"tokens_per_second", &test_tokens_per_second},
        {"unit_checks", &test_unit_checks},
        {"e2e_checks", &test_e2e_checks},
        {"summary_text", &test_summary_text},
    });

    return summary.failed == 0 ? 0 : 1;
}
