// Chapter 07：运行可重复的 CPU 多请求 benchmark，并输出 JSON 结果。
// 职责：复用 Chapter 05 Scheduler/Runtime，比较 1、2、4 个请求的批量行为。
// 验证：每个场景检查请求完成、生成数量、KV Cache 隔离和结果字段完整。
#include "chapter05_generation/embedding.h"
#include "chapter05_generation/position_encoding.h"
#include "chapter05_generation/request.h"
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/sampler.h"
#include "chapter05_generation/scheduler.h"
#include "chapter05_generation/transformer.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct ScenarioResult {
    std::size_t request_count = 0;
    std::size_t prompt_tokens = 0;
    std::size_t max_new_tokens = 0;
    std::size_t warmup = 0;
    std::size_t iterations = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    double ttft_ms = 0.0;
    double throughput_tokens_per_second = 0.0;
    std::size_t generated_tokens = 0;
    std::size_t completed_requests = 0;
    bool correctness = false;
};

struct RunMeasurement {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double first_token_decode_ms = 0.0;
    std::size_t generated_tokens = 0;
    std::size_t completed_requests = 0;
    bool correctness = false;
};

RunMeasurement run_once(
    std::size_t request_count,
    const chapter05_generation::GenerationConfig& config,
    const chapter05_generation::Transformer& transformer) {
    using namespace chapter05_generation;

    Runtime runtime(config, transformer);
    GreedySampler sampler;
    std::vector<Request> requests;
    std::vector<ScheduledRequest> scheduled;
    requests.reserve(request_count);
    scheduled.reserve(request_count);

    for (std::size_t i = 0; i < request_count; ++i) {
        const TokenId base = static_cast<TokenId>(1 + i);
        requests.emplace_back(TokenSequence{base, base + 1, base + 2, base + 3});
    }
    for (std::size_t i = 0; i < request_count; ++i) {
        scheduled.push_back({i, &requests[i], &sampler});
    }

    const auto prefill_start = Clock::now();
    std::size_t prefetched = 0;
    for (Request& request : requests) {
        prefetched += runtime.prefill(request) > 0 ? 1 : 0;
    }
    const auto prefill_end = Clock::now();

    const auto decode_start = Clock::now();
    double first_token_decode_ms = 0.0;
    const SchedulerStats stats = RoundRobinScheduler{}.run_with_stats(
        runtime,
        scheduled,
        [&](std::size_t /*request_id*/, TokenId /*token*/, std::size_t /*step*/) {
            if (first_token_decode_ms == 0.0) {
                first_token_decode_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - decode_start).count();
            }
        });
    const auto decode_end = Clock::now();

    bool correct = prefetched == request_count && stats.completed_requests == request_count;
    for (const Request& request : requests) {
        correct = correct && request.finished();
        correct = correct && request.generated_size() == config.max_new_tokens;
        correct = correct && request.cache().size() == request.prompt_size() + request.generated_size();
    }

    return {
        std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count(),
        std::chrono::duration<double, std::milli>(decode_end - decode_start).count(),
        first_token_decode_ms,
        stats.generated_tokens,
        stats.completed_requests,
        correct,
    };
}

ScenarioResult benchmark_scenario(
    std::size_t request_count,
    std::size_t warmup,
    std::size_t iterations,
    const chapter05_generation::GenerationConfig& config,
    const chapter05_generation::Transformer& transformer) {
    for (std::size_t i = 0; i < warmup; ++i) {
        (void)run_once(request_count, config, transformer);
    }

    ScenarioResult result;
    result.request_count = request_count;
    result.prompt_tokens = 4;
    result.max_new_tokens = config.max_new_tokens;
    result.warmup = warmup;
    result.iterations = iterations;
    result.correctness = true;

    for (std::size_t i = 0; i < iterations; ++i) {
        const RunMeasurement measurement = run_once(request_count, config, transformer);
        result.prefill_ms += measurement.prefill_ms;
        result.decode_ms += measurement.decode_ms;
        result.generated_tokens += measurement.generated_tokens;
        result.completed_requests += measurement.completed_requests;
        result.correctness = result.correctness && measurement.correctness;
        result.ttft_ms += measurement.prefill_ms + measurement.first_token_decode_ms;
    }

    if (iterations > 0) {
        result.prefill_ms /= static_cast<double>(iterations);
        result.decode_ms /= static_cast<double>(iterations);
        result.ttft_ms /= static_cast<double>(iterations);
        result.generated_tokens /= iterations;
        result.completed_requests /= iterations;
    }
    result.total_ms = result.prefill_ms + result.decode_ms;
    result.throughput_tokens_per_second = result.total_ms > 0.0
        ? static_cast<double>(result.generated_tokens) * 1000.0 / result.total_ms
        : 0.0;
    return result;
}

std::string json_number(double value) {
    std::ostringstream output;
    output << value;
    return output.str();
}

void write_json(
    const std::string& output_path,
    std::size_t warmup,
    std::size_t iterations,
    const std::vector<ScenarioResult>& results) {
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("cannot open benchmark result: " + output_path);
    }

    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"benchmark\": \"chapter07_cpu_multi_request\",\n"
           << "  \"device\": \"cpu\",\n"
           << "  \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n"
           << "  \"warmup\": " << warmup << ",\n"
           << "  \"iterations\": " << iterations << ",\n"
           << "  \"scenarios\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const ScenarioResult& result = results[i];
        output << "    {\n"
               << "      \"request_count\": " << result.request_count << ",\n"
               << "      \"prompt_tokens\": " << result.prompt_tokens << ",\n"
               << "      \"max_new_tokens\": " << result.max_new_tokens << ",\n"
               << "      \"prefill_ms\": " << json_number(result.prefill_ms) << ",\n"
               << "      \"decode_ms\": " << json_number(result.decode_ms) << ",\n"
               << "      \"total_ms\": " << json_number(result.total_ms) << ",\n"
               << "      \"ttft_ms\": " << json_number(result.ttft_ms) << ",\n"
               << "      \"throughput_tokens_per_second\": "
               << json_number(result.throughput_tokens_per_second) << ",\n"
               << "      \"generated_tokens\": " << result.generated_tokens << ",\n"
               << "      \"completed_requests\": " << result.completed_requests << ",\n"
               << "      \"correctness\": " << (result.correctness ? "true" : "false") << "\n"
               << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string output_path = "chapter07_cpu_multi_request.json";
        std::size_t warmup = 1;
        std::size_t iterations = 3;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (argument == "--warmup" && i + 1 < argc) {
                warmup = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (argument == "--iterations" && i + 1 < argc) {
                iterations = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else {
                std::cerr << "usage: multi_request_benchmark [--output PATH] [--warmup N] [--iterations N]\n";
                return 2;
            }
        }
        if (iterations == 0) {
            std::cerr << "iterations must be greater than zero\n";
            return 2;
        }

        chapter05_generation::GenerationConfig config;
        config.max_new_tokens = 4;
        config.max_context_tokens = 32;
        chapter05_generation::Embedding embedding(64);
        chapter05_generation::PositionEncoding position_encoding(32);
        chapter05_generation::Transformer transformer(2, embedding, position_encoding);

        std::vector<ScenarioResult> results;
        for (const std::size_t request_count : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
            results.push_back(benchmark_scenario(request_count, warmup, iterations, config, transformer));
        }
        write_json(output_path, warmup, iterations, results);

        bool correct = true;
        for (const ScenarioResult& result : results) {
            correct = correct && result.correctness;
            std::cout << "requests=" << result.request_count
                      << " correctness=" << (result.correctness ? "PASS" : "FAIL")
                      << " prefill_ms=" << result.prefill_ms
                      << " decode_ms=" << result.decode_ms
                      << " throughput_tokens_per_second=" << result.throughput_tokens_per_second
                      << '\n';
        }
        std::cout << "result_file=" << output_path << '\n';
        return correct ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_error=" << error.what() << '\n';
        return 1;
    }
}
