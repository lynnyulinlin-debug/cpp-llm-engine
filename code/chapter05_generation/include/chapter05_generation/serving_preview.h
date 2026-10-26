// Chapter 05：声明 token-level step planner、CPU KV block pool 和 block table。
// 职责：按 token/block budget 选择请求，并把 logical token 映射到物理 block/slot。
// 边界：管理 CPU block ownership，不保存分层 K/V 数值或实现 PagedAttention kernel。
#pragma once

#include "chapter05_generation/scheduler.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace chapter05_generation {

class KvBlockPool;

struct BatchCandidate {
    std::size_t scheduled_index = 0;
    std::size_t request_id = 0;
    Request* request = nullptr;
    bool prefill = false;
    std::size_t token_count = 0;
};

struct StepBatchPlan {
    std::vector<BatchCandidate> candidates;
    std::size_t token_budget = 0;
    std::size_t scheduled_tokens = 0;
    std::size_t prefill_requests = 0;
    std::size_t decode_requests = 0;
    std::size_t deferred_requests = 0;
};

class StepBatchPlanner final {
public:
    StepBatchPlan build(
        const std::vector<ScheduledRequest>& requests,
        std::size_t max_batch_tokens,
        std::size_t max_requests = 0,
        std::size_t max_prefill_chunk_tokens = 0) const;
};

struct StepBatchStats {
    std::size_t scheduled_requests = 0;
    std::size_t scheduled_tokens = 0;
    std::size_t prefill_requests = 0;
    std::size_t decode_requests = 0;
    std::size_t generated_tokens = 0;
    std::size_t terminal_requests = 0;
    std::size_t deferred_requests = 0;
};

class ContinuousBatchScheduler final {
public:
    StepBatchStats run_step(
        Runtime& runtime,
        const std::vector<ScheduledRequest>& requests,
        std::size_t max_batch_tokens,
        std::size_t max_requests = 0,
        const TokenStreamCallback& on_token = {},
        KvBlockPool* block_pool = nullptr,
        std::size_t max_prefill_chunk_tokens = 0) const;
};

struct BlockLocation {
    std::size_t block = 0;
    std::size_t slot = 0;
};

class BlockTable final {
public:
    explicit BlockTable(std::size_t block_size);

    std::size_t block_size() const;
    std::size_t token_count() const;
    std::size_t block_count() const;
    std::size_t capacity_tokens() const;
    const std::vector<std::size_t>& physical_blocks() const;
    BlockLocation locate(std::size_t token_index) const;

private:
    friend class KvBlockPool;

    void append_block(std::size_t physical_block);
    std::size_t remove_last_block();
    void set_token_count(std::size_t token_count);

    std::size_t block_size_;
    std::size_t token_count_ = 0;
    std::vector<std::size_t> physical_blocks_;
};

class KvBlockPool final {
public:
    KvBlockPool(std::size_t total_blocks, std::size_t block_size);

    std::size_t total_blocks() const;
    std::size_t free_blocks() const;
    std::size_t used_blocks() const;
    std::size_t block_size() const;
    bool contains(std::size_t request_id) const;
    std::size_t block_count(std::size_t request_id) const;
    std::size_t token_count(std::size_t request_id) const;

    bool reserve(std::size_t request_id, std::size_t total_tokens);
    bool commit_tokens(std::size_t request_id, std::size_t total_tokens);
    bool trim_to_tokens(std::size_t request_id, std::size_t total_tokens);
    bool release(std::size_t request_id);
    BlockLocation locate(std::size_t request_id, std::size_t token_index) const;

private:
    std::size_t required_blocks(std::size_t token_count) const;

    std::size_t block_size_;
    std::vector<bool> free_;
    std::unordered_map<std::size_t, BlockTable> tables_;
};

}  // namespace chapter05_generation
