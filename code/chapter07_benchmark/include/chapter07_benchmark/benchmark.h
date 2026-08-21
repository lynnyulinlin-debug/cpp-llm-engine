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

struct TraceEvent {
    std::string phase;
    std::string event;
    double start_ms = 0.0;
    double duration_ms = 0.0;
    std::string note;
};

struct HotspotEntry {
    std::string phase;
    std::string symbol;
    double self_ms = 0.0;
    double total_ms = 0.0;
    std::size_t calls = 0;
    std::string reason;
};

BenchmarkResult make_result(const std::string& name, std::size_t iterations, double elapsed_ms);
double tokens_per_second(std::size_t tokens, double elapsed_ms);
double phase_share(double elapsed_ms, double total_ms);
std::vector<BenchmarkSample> sample_generation_path();
std::vector<TraceEvent> sample_trace_events();
std::vector<HotspotEntry> sample_hotspots();

}  // namespace chapter07_benchmark
