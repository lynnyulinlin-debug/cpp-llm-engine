// Chapter 09：实现 C++ Service Core 的 admission、队列推进和资源释放。
// 契约：Service Core 只编排 Request；Runtime、Scheduler 和 Request 仍由 Chapter 05 提供。
// 验证：测试检查请求隔离、容量拒绝、终态处理和 cache clear。
#include "chapter09_serving/serving_core.h"

#include "chapter05_generation/request.h"
#include "chapter05_generation/runtime.h"
#include "chapter05_generation/serving_preview.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace chapter09_serving {

namespace {

double elapsed_seconds(
    ServiceClock::time_point start,
    ServiceClock::time_point end) {
    return std::max(0.0, std::chrono::duration<double>(end - start).count());
}

}  // namespace

ServingCore::ServingCore(
    std::size_t max_inflight,
    std::size_t max_batch,
    ServiceNow now,
    std::shared_ptr<chapter05_generation::KvBlockPool> block_pool)
    : max_inflight_(max_inflight),
      max_batch_(max_batch),
      now_(std::move(now)),
      block_pool_(std::move(block_pool)) {
    if (max_inflight_ == 0 || max_batch_ == 0 || max_batch_ > max_inflight_) {
        throw std::invalid_argument("service capacity and batch size must be positive");
    }
    if (!now_) now_ = []() { return ServiceClock::now(); };
}

bool ServingCore::contains(std::size_t request_id) const {
    const auto matches = [request_id](const Entry& entry) {
        return entry.request_id == request_id;
    };
    return std::any_of(waiting_.begin(), waiting_.end(), matches) ||
           std::any_of(running_.begin(), running_.end(), matches);
}

AdmissionResult ServingCore::admit(
    std::size_t request_id,
    chapter05_generation::Request& request,
    const chapter05_generation::Sampler& sampler,
    std::shared_ptr<chapter05_generation::RequestModelExecutor> executor) {
    if (contains(request_id)) {
        ++metrics_.duplicate_total;
        return {AdmissionStatus::DuplicateId};
    }
    if (request.terminal()) {
        ++metrics_.invalid_total;
        return {AdmissionStatus::InvalidRequest};
    }
    if (inflight_size() >= max_inflight_) {
        ++metrics_.queue_full_total;
        return {AdmissionStatus::QueueFull};
    }
    waiting_.push_back({
        request_id, &request, &sampler, std::move(executor), now_(), false, false});
    ++metrics_.accepted_total;
    metrics_.peak_inflight = std::max(metrics_.peak_inflight, inflight_size());
    return {AdmissionStatus::Accepted};
}

bool ServingCore::mark_terminal(std::size_t request_id, bool timed_out) {
    const auto waiting = std::find_if(
        waiting_.begin(), waiting_.end(),
        [request_id](const Entry& entry) { return entry.request_id == request_id; });
    if (waiting != waiting_.end()) {
        if (timed_out) {
            waiting->request->mark_timed_out();
        } else {
            waiting->request->cancel();
        }
        observe_completion(*waiting, now_());
        waiting_.erase(waiting);
        if (timed_out) {
            ++metrics_.timed_out_total;
        } else {
            ++metrics_.cancelled_total;
        }
        return true;
    }

    const auto running = std::find_if(
        running_.begin(), running_.end(),
        [request_id](const Entry& entry) { return entry.request_id == request_id; });
    if (running == running_.end()) {
        return false;
    }
    if (timed_out) {
        running->request->mark_timed_out();
    } else {
        running->request->cancel();
    }
    observe_completion(*running, now_());
    if (timed_out) {
        ++metrics_.timed_out_total;
    } else {
        ++metrics_.cancelled_total;
    }
    return true;
}

bool ServingCore::cancel(std::size_t request_id) {
    return mark_terminal(request_id, false);
}

bool ServingCore::timeout(std::size_t request_id) {
    return mark_terminal(request_id, true);
}

ServiceStats ServingCore::dispatch(
    chapter05_generation::Runtime& runtime,
    const chapter05_generation::TokenStreamCallback& on_token) {
    if (block_pool_ != nullptr) {
        throw std::logic_error("block-aware ServiceCore requires dispatch_step");
    }
    ServiceStats stats;
    while (!waiting_.empty() && running_.size() < max_batch_) {
        Entry entry = std::move(waiting_.front());
        waiting_.pop_front();
        metrics_.queue_wait_seconds_sum += elapsed_seconds(entry.admitted_at, now_());
        ++metrics_.queue_wait_count;
        running_.push_back(std::move(entry));
    }
    stats.admitted = running_.size();

    std::vector<chapter05_generation::ScheduledRequest> scheduled;
    scheduled.reserve(running_.size());
    for (const Entry& entry : running_) {
        scheduled.push_back({entry.request_id, entry.request, entry.sampler, entry.executor});
    }
    const chapter05_generation::SchedulerStats scheduler_stats =
        chapter05_generation::RoundRobinScheduler{}.run_with_stats(
            runtime,
            std::move(scheduled),
            [this, &on_token](
                std::size_t request_id,
                chapter05_generation::TokenId token,
                std::size_t step) {
                Entry* entry = find_running(request_id);
                if (entry != nullptr) observe_token(*entry, now_());
                if (on_token) on_token(request_id, token, step);
            });
    stats.completed = scheduler_stats.completed_requests;
    stats.generated_tokens = scheduler_stats.generated_tokens;
    metrics_.generated_tokens_total += stats.generated_tokens;
    for (Entry& entry : running_) {
        if (entry.request != nullptr && entry.request->terminal()) {
            observe_completion(entry, now_());
        }
    }
    return stats;
}

ServiceStats ServingCore::dispatch_step(
    chapter05_generation::Runtime& runtime,
    std::size_t max_batch_tokens,
    const chapter05_generation::TokenStreamCallback& on_token) {
    ServiceStats stats;
    while (!waiting_.empty() && running_.size() < max_batch_) {
        Entry entry = std::move(waiting_.front());
        waiting_.pop_front();
        metrics_.queue_wait_seconds_sum += elapsed_seconds(entry.admitted_at, now_());
        ++metrics_.queue_wait_count;
        running_.push_back(std::move(entry));
    }
    stats.admitted = running_.size();

    std::vector<chapter05_generation::ScheduledRequest> scheduled;
    scheduled.reserve(running_.size());
    for (const Entry& entry : running_) {
        scheduled.push_back({entry.request_id, entry.request, entry.sampler, entry.executor});
    }
    const auto step_stats = chapter05_generation::ContinuousBatchScheduler{}.run_step(
        runtime,
        scheduled,
        max_batch_tokens,
        max_batch_,
        [this, &on_token](
            std::size_t request_id,
            chapter05_generation::TokenId token,
            std::size_t step) {
            Entry* entry = find_running(request_id);
            if (entry != nullptr) observe_token(*entry, now_());
            if (on_token) on_token(request_id, token, step);
        },
        block_pool_.get());
    stats.completed = static_cast<std::size_t>(std::count_if(
        running_.begin(), running_.end(),
        [](const Entry& entry) {
            return entry.request != nullptr && entry.request->finished();
        }));
    stats.generated_tokens = step_stats.generated_tokens;
    stats.scheduled_requests = step_stats.scheduled_requests;
    stats.scheduled_tokens = step_stats.scheduled_tokens;
    stats.deferred_requests = step_stats.deferred_requests;
    metrics_.generated_tokens_total += stats.generated_tokens;
    for (Entry& entry : running_) {
        if (entry.request != nullptr && entry.request->terminal()) {
            observe_completion(entry, now_());
        }
    }
    return stats;
}

std::size_t ServingCore::release_completed() {
    const auto completed = [](const Entry& entry) {
        return entry.request != nullptr && entry.request->terminal();
    };
    std::size_t released = 0;
    for (Entry& entry : running_) {
        if (completed(entry)) {
            observe_completion(entry, now_());
            entry.request->cache().clear();
            if (block_pool_ != nullptr) block_pool_->release(entry.request_id);
            ++released;
            if (entry.request->finished()) {
                ++metrics_.completed_total;
            } else if (entry.request->failed()) {
                ++metrics_.failed_total;
            }
        }
    }
    running_.erase(
        std::remove_if(running_.begin(), running_.end(), completed), running_.end());
    metrics_.released_total += released;
    return released;
}

std::size_t ServingCore::waiting_size() const {
    return waiting_.size();
}

std::size_t ServingCore::running_size() const {
    return running_.size();
}

std::size_t ServingCore::inflight_size() const {
    return waiting_.size() + running_.size();
}

const ServiceMetrics& ServingCore::metrics() const {
    return metrics_;
}

const std::shared_ptr<chapter05_generation::KvBlockPool>&
ServingCore::block_pool() const {
    return block_pool_;
}

ServingCore::Entry* ServingCore::find_running(std::size_t request_id) {
    const auto entry = std::find_if(
        running_.begin(), running_.end(),
        [request_id](const Entry& candidate) {
            return candidate.request_id == request_id;
        });
    return entry == running_.end() ? nullptr : &*entry;
}

void ServingCore::observe_token(
    Entry& entry,
    ServiceClock::time_point observed_at) {
    if (!entry.first_token_observed) {
        metrics_.ttft_seconds_sum += elapsed_seconds(entry.admitted_at, observed_at);
        ++metrics_.ttft_count;
        entry.first_token_observed = true;
    }
    if (entry.request != nullptr && entry.request->terminal()) {
        observe_completion(entry, observed_at);
    }
}

void ServingCore::observe_completion(
    Entry& entry,
    ServiceClock::time_point observed_at) {
    if (entry.completion_observed) return;
    metrics_.completion_seconds_sum += elapsed_seconds(entry.admitted_at, observed_at);
    ++metrics_.completion_count;
    entry.completion_observed = true;
}

}  // namespace chapter09_serving
