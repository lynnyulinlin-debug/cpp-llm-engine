# 05.10 Continuous Batching 入门

朴素生成通常按“一个请求跑完再跑下一个”理解，但真实 serving runtime 更关心吞吐和等待时间。Continuous Batching 的核心是：每个 decode step 都重新组织当前可运行的请求，让新请求可以插入，让已完成请求及时离开。

**阅读入口**

- 前置阅读：[`05.5 Scheduler`](./05_scheduler.md)、[`05.9 Request 生命周期`](./08_request_lifecycle.md)
- 源码直达：<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>、<a href="../_code/chapter05_generation/src/serving_preview.cpp"><code>serving_preview.cpp</code></a>、<a href="../_code/chapter05_generation/tests/serving_preview_test.cpp"><code>serving_preview_test.cpp</code></a>
- 相关阅读：[`05.11 Block KV Cache 阅读入口`](./10_block_kv_cache.md)、[`05.8 Runtime 框架对照`](./07_vllm_comparison.md)、[`Mini-SGLang 项目卡`](../reference/ecosystem/mini_sglang.md)

## 它解决什么问题

如果 batch 在请求开始时固定，短请求会被长请求拖住，新请求也要等整批结束后才能进入。Continuous Batching 把 batch 的边界从“请求集合”改成“每个 step 的可执行集合”。

这样 runtime 每一步都可以做三件事：

- 把新请求放进 prefill
- 让正在 decode 的请求继续生成一个 token
- 把结束、取消或资源不足的请求移出运行集合

## 最小调度循环

一个便于入门的循环可以写成下面这个形状：

```text
while has_work:
  admit_waiting_requests()
  run_prefill_for_new_requests()
  run_one_decode_step_for_active_requests()
  sample_next_tokens()
  update_request_status()
  release_finished_requests()
```

真实 vLLM 会把这件事做得复杂得多：它需要考虑 token budget、block budget、prefill 和 decode 的混排、抢占、streaming output，以及 GPU batch 形状。但入门时先抓住一个事实：scheduler 调度的不是完整请求的一次执行，而是每个 step 里哪些 token 应该被推进。

最小 admission 规则应保持明确：先检查 token/cache budget，再把 waiting 请求加入当前 step；预算不足时回压、延迟或拒绝，而不是让 batch 无限增长。取消和超时请求必须在下一次 step 前移出运行集合，已完成请求先发出最终结果，再进入 cache 和 workspace 回收路径。

## 代码如何形成一个 step

`StepBatchPlanner::build()` 每次重新扫描请求状态，先选择已经完成 prefill 的 decode 请求，每个 decode 消耗 1 个 token budget；再用剩余预算选择 prefill 请求。支持 chunk 的 executor 可以只消费 prompt 的一部分，`Request::prefill_offset()` 记录已经提交的 token 数。旧 executor 保持整段 prefill，不会被调度器强行切分。

中间 chunk 只向 Context 追加 token 和分层 K/V，不请求 logits，也不采样 token。最后一个 chunk 才请求 logits、把 Request 标记为 prefilled，并采样首个输出 token。这样可避免每段重置 Context，也不会把中间位置误当成生成起点。

| 输入状态 | 本 step 动作 | 预算计数 |
| --- | --- | --- |
| 已 prefill、未终止 | 执行一次 decode | 1 token |
| 未 prefill、executor 支持 chunk | 从 `prefill_offset` 追加一段；仅 final chunk 采样首 token | 本段 prompt token 数 |
| 未 prefill、executor 不支持 chunk | 仅在完整 prompt 可放入预算时执行 | 剩余 prompt token 数 |
| 取消、失败或完成 | 跳过 | 0 |
| 超出剩余预算 | 留到下一 step | 0 |

## 公平性与提交规则

decode 仍然优先，prefill 请求按已提交的 `prefill_offset` 从小到大稳定排序。长 prompt 消费一个 chunk 后，进度更少的新请求会在下一 step 排到前面；两个请求进度相同则保持 admission 顺序。这个最少进度优先规则提供可复现的基础防饥饿行为，但还不是优先级、deadline 或抢占调度。

Scheduler 在执行前预留 candidate 所需的物理 block。只有 Request 游标或生成 token 实际前进时才提交 token count；executor 失败时撤销本 step 的预留。Chapter09 的 `ServingCore::dispatch_step()` 负责从 waiting 补充 running，取消部分 prefill 请求后由 `release_completed()` 释放已经提交的 block。

| 场景 | Request 进度 | logits / 采样 | Block Pool |
| --- | --- | --- | --- |
| 中间 chunk 成功 | `prefill_offset += chunk_size` | 不产生 | 提交本段 token count |
| final chunk 成功 | 游标等于 prompt 长度 | 产生 logits 并采样首 token | 提交最后一段 prompt token |
| executor 失败 | 游标不变，进入 `Failed` | 不采样 | 回滚本 step 预留 |
| 部分 prefill 后取消 | 保留终态供回收检查 | 不再执行 | release 时归还全部 block |

## 验证证据

验证同时保留原子 prefill 兼容路径，并增加两条 5-token prompt 在 2-token budget 下的轮换推进。真实 TinyLlama 服务 smoke 使用 3-token budget，使 `Hello TinyLlama` 必须分段 prefill，再核对原有 golden token 和 22 层 Context 隔离。

| 验证项 | 固定证据 |
| --- | --- |
| 中间 chunk | 不生成 token，`prefill_offset` 与 block token count 同步增加 |
| 防饥饿 | 请求 A 推进后，请求 B 在下一 step 先获得预算 |
| final chunk | 两个剩余 1-token chunk 可在同一步完成并各采样一次 |
| 失败回滚 | Request 游标为 0，pool 恢复全部 free block |
| 真实模型 | `chunked_prefill_match: true`，两组 golden token 不变 |
