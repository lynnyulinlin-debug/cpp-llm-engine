// 多模态 Runtime：请求阶段与资源指标实现
// 职责：用单活动阶段状态机生成可聚合、可测试的输入 pipeline 指标。
// 契约：耗时来自注入的 steady clock；live bytes 由执行路径估算或采样后传入。
// 验证：multimodal_request_metrics_test。
#include "multimodal/request_metrics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace multimodal {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

bool valid_stage(MultimodalRequestStage stage) {
    return static_cast<std::size_t>(stage) <
           static_cast<std::size_t>(MultimodalRequestStage::Count);
}

std::size_t stage_index(MultimodalRequestStage stage) {
    return static_cast<std::size_t>(stage);
}

}  // namespace

const MultimodalStageMetrics& MultimodalRequestMetrics::stage(
    MultimodalRequestStage value) const {
    if (!valid_stage(value)) throw std::out_of_range("multimodal request stage is invalid");
    return stages[stage_index(value)];
}

double MultimodalRequestMetrics::input_pipeline_seconds() const {
    double total = 0.0;
    for (const auto& value : stages) total += value.elapsed_seconds;
    return total;
}

MultimodalStageRecorder::MultimodalStageRecorder(MultimodalMetricsNow now)
    : now_(std::move(now)) {
    if (!now_) now_ = []() { return MultimodalMetricsClock::now(); };
}

bool MultimodalStageRecorder::begin(
    MultimodalRequestStage stage,
    std::size_t live_bytes,
    std::string* error) {
    if (!valid_stage(stage)) return fail(error, "multimodal request stage is invalid");
    if (active_) return fail(error, "a multimodal request stage is already active");
    active_ = true;
    active_stage_ = stage;
    started_at_ = now_();
    auto& metrics = metrics_.stages[stage_index(stage)];
    metrics.peak_live_bytes = std::max(metrics.peak_live_bytes, live_bytes);
    metrics_.peak_live_bytes = std::max(metrics_.peak_live_bytes, live_bytes);
    if (error != nullptr) error->clear();
    return true;
}

bool MultimodalStageRecorder::finish(
    MultimodalRequestStage stage,
    std::size_t live_bytes,
    std::string* error) {
    if (!active_) return fail(error, "no multimodal request stage is active");
    if (stage != active_stage_) return fail(error, "multimodal request stage does not match");
    const auto finished_at = now_();
    if (finished_at < started_at_) return fail(error, "multimodal metrics clock moved backwards");
    const double elapsed = std::chrono::duration<double>(finished_at - started_at_).count();
    if (!std::isfinite(elapsed) || elapsed < 0.0) {
        return fail(error, "multimodal request stage elapsed time is invalid");
    }
    auto& metrics = metrics_.stages[stage_index(stage)];
    metrics.elapsed_seconds += elapsed;
    ++metrics.invocation_count;
    metrics.peak_live_bytes = std::max(metrics.peak_live_bytes, live_bytes);
    metrics_.peak_live_bytes = std::max(metrics_.peak_live_bytes, live_bytes);
    active_ = false;
    if (error != nullptr) error->clear();
    return true;
}

bool MultimodalStageRecorder::set_sequence_shape(
    std::size_t visual_inputs,
    std::size_t visual_rows,
    std::size_t prompt_positions,
    std::string* error) {
    if (visual_inputs == 0 || visual_rows == 0 || prompt_positions == 0 ||
        visual_rows < visual_inputs || prompt_positions < visual_rows) {
        return fail(error, "multimodal request sequence shape is invalid");
    }
    metrics_.visual_input_count = visual_inputs;
    metrics_.visual_row_count = visual_rows;
    metrics_.prompt_position_count = prompt_positions;
    if (error != nullptr) error->clear();
    return true;
}

bool MultimodalStageRecorder::active() const {
    return active_;
}

const MultimodalRequestMetrics& MultimodalStageRecorder::metrics() const {
    return metrics_;
}

const char* multimodal_request_stage_name(MultimodalRequestStage stage) {
    switch (stage) {
        case MultimodalRequestStage::ImageDecode: return "image_decode";
        case MultimodalRequestStage::DynamicTiling: return "dynamic_tiling";
        case MultimodalRequestStage::PromptTokenization: return "prompt_tokenization";
        case MultimodalRequestStage::VisionEncoder: return "vision_encoder";
        case MultimodalRequestStage::Projector: return "projector";
        case MultimodalRequestStage::EmbeddingComposition: return "embedding_composition";
        case MultimodalRequestStage::LanguagePrefill: return "language_prefill";
        case MultimodalRequestStage::Count: return "invalid";
    }
    return "invalid";
}

}  // namespace multimodal
