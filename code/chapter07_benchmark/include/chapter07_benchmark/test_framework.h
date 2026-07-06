#pragma once

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string_view>

namespace chapter07_benchmark {

struct TestCase {
    std::string_view name;
    bool (*fn)();
};

struct TestSummary {
    std::size_t passed = 0;
    std::size_t failed = 0;
};

inline TestSummary run_test_suite(std::initializer_list<TestCase> tests) {
    TestSummary summary;
    for (const auto& test : tests) {
        const bool ok = test.fn();
        std::cout << "[test] " << test.name << ": " << (ok ? "PASS" : "FAIL") << '\n';
        if (ok) {
            ++summary.passed;
        } else {
            ++summary.failed;
        }
    }
    std::cout << "[test] summary: passed=" << summary.passed << ", failed=" << summary.failed << '\n';
    return summary;
}

}  // namespace chapter07_benchmark
