// 多模态 Runtime：请求阶段与资源指标契约
// 职责：聚合图片输入到 LLM prefill 的阶段耗时、调用次数和峰值存活字节。
// 契约：同一时刻只能记录一个阶段；视觉编码与 Projector 可按 visual input 重复出现。
// 验证：multimodal_request_metrics_test 使用注入时钟覆盖顺序、聚合和错误输入。
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <functional>
#include <string>

namespace multimodal {

enum class MultimodalRequestStage : std::size_t {
    ImageDecode = 0,
    DynamicTiling,
    PromptTokenization,
    VisionEncoder,
    Projector,
    EmbeddingComposition,
    LanguagePrefill,
    Count,
};

struct MultimodalStageMetrics {
    double elapsed_seconds = 0.0;
    std::size_t invocation_count = 0;
    std::size_t peak_live_bytes = 0;
};

struct MultimodalRequestMetrics {
    std::array<MultimodalStageMetrics,
               static_cast<std::size_t>(MultimodalRequestStage::Count)> stages{};
    std::size_t visual_input_count = 0;
    std::size_t visual_row_count = 0;
    std::size_t prompt_position_count = 0;
    std::size_t peak_live_bytes = 0;

    const MultimodalStageMetrics& stage(MultimodalRequestStage value) const;
    double input_pipeline_seconds() const;
};

using MultimodalMetricsClock = std::chrono::steady_clock;
using MultimodalMetricsNow = std::function<MultimodalMetricsClock::time_point()>;

class MultimodalStageRecorder final {
public:
    explicit MultimodalStageRecorder(MultimodalMetricsNow now = {});

    bool begin(MultimodalRequestStage stage,
               std::size_t live_bytes,
               std::string* error = nullptr);
    bool finish(MultimodalRequestStage stage,
                std::size_t live_bytes,
                std::string* error = nullptr);
    bool set_sequence_shape(std::size_t visual_inputs,
                            std::size_t visual_rows,
                            std::size_t prompt_positions,
                            std::string* error = nullptr);

    bool active() const;
    const MultimodalRequestMetrics& metrics() const;

private:
    MultimodalMetricsNow now_;
    MultimodalRequestMetrics metrics_;
    bool active_ = false;
    MultimodalRequestStage active_stage_ = MultimodalRequestStage::ImageDecode;
    MultimodalMetricsClock::time_point started_at_{};
};

const char* multimodal_request_stage_name(MultimodalRequestStage stage);

}  // namespace multimodal
