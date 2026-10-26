// 多模态 Runtime 测试：验证请求阶段耗时、重复视觉调用和峰值存活字节。
// 覆盖：注入时钟、阶段匹配、shape、聚合、非法枚举和时钟回退。
// 依赖：不需要真实模型、Python、网络、CUDA 或 BLAS。
// 运行：由 CTest 执行 multimodal_request_metrics_test。
#include "multimodal/request_metrics.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

bool near(double lhs, double rhs) {
    return std::abs(lhs - rhs) < 1.0e-12;
}

class ManualClock final {
public:
    multimodal::MultimodalMetricsClock::time_point now() const { return now_; }
    void advance(std::chrono::milliseconds duration) { now_ += duration; }
    void rewind(std::chrono::milliseconds duration) { now_ -= duration; }

private:
    multimodal::MultimodalMetricsClock::time_point now_{};
};

}  // namespace

int main() {
    using multimodal::MultimodalRequestStage;

    ManualClock clock;
    multimodal::MultimodalStageRecorder recorder([&clock]() { return clock.now(); });
    std::string error;
    assert(recorder.begin(MultimodalRequestStage::ImageDecode, 100, &error));
    assert(recorder.active());
    assert(!recorder.begin(MultimodalRequestStage::DynamicTiling, 0, &error));
    assert(error.find("already active") != std::string::npos);
    assert(!recorder.finish(MultimodalRequestStage::DynamicTiling, 0, &error));
    assert(error.find("does not match") != std::string::npos);
    clock.advance(std::chrono::milliseconds(10));
    assert(recorder.finish(MultimodalRequestStage::ImageDecode, 120, &error));
    assert(!recorder.active());

    for (int visual = 0; visual < 2; ++visual) {
        assert(recorder.begin(MultimodalRequestStage::VisionEncoder, 200 + visual * 50, &error));
        clock.advance(std::chrono::milliseconds(20));
        assert(recorder.finish(MultimodalRequestStage::VisionEncoder, 300 + visual * 50, &error));
        assert(recorder.begin(MultimodalRequestStage::Projector, 180, &error));
        clock.advance(std::chrono::milliseconds(5));
        assert(recorder.finish(MultimodalRequestStage::Projector, 190, &error));
    }
    assert(recorder.set_sequence_shape(2, 128, 166, &error));

    const auto& metrics = recorder.metrics();
    assert(metrics.stage(MultimodalRequestStage::ImageDecode).invocation_count == 1);
    assert(metrics.stage(MultimodalRequestStage::VisionEncoder).invocation_count == 2);
    assert(near(metrics.stage(MultimodalRequestStage::VisionEncoder).elapsed_seconds, 0.04));
    assert(metrics.stage(MultimodalRequestStage::Projector).invocation_count == 2);
    assert(near(metrics.stage(MultimodalRequestStage::Projector).elapsed_seconds, 0.01));
    assert(metrics.visual_input_count == 2);
    assert(metrics.visual_row_count == 128);
    assert(metrics.prompt_position_count == 166);
    assert(metrics.peak_live_bytes == 350);
    assert(near(metrics.input_pipeline_seconds(), 0.06));
    assert(std::string(multimodal::multimodal_request_stage_name(
               MultimodalRequestStage::EmbeddingComposition)) == "embedding_composition");

    bool invalid_stage_threw = false;
    try {
        (void)metrics.stage(MultimodalRequestStage::Count);
    } catch (const std::out_of_range&) {
        invalid_stage_threw = true;
    }
    assert(invalid_stage_threw);
    assert(!recorder.set_sequence_shape(2, 1, 3, &error));
    assert(!recorder.finish(MultimodalRequestStage::ImageDecode, 0, &error));

    assert(recorder.begin(MultimodalRequestStage::LanguagePrefill, 400, &error));
    clock.rewind(std::chrono::milliseconds(1));
    assert(!recorder.finish(MultimodalRequestStage::LanguagePrefill, 400, &error));
    assert(recorder.active());
    return 0;
}
