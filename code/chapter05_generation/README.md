# Chapter 05: Generation

这个目录放的是 `chapter05_generation` 的最小运行时实现。

这一章的代码目标很明确：

- 先把 `forward` 变成 `prefill / decode` 分离的生成 runtime
- 再把 `Request`、`KVCache`、`Sampler` 和 `Scheduler` 显式连起来
- 最后用测试和示例确认这条链路可以重复运行

## 本目录包含什么

- `include/`：章节对外接口
- `src/`：最小 runtime、scheduler、sampler、cache 和 transformer 实现
- `tests/`：章节内最小测试和运行时收口测试
- `examples/`：生成 demo 和 streaming demo
- `scripts/run.sh`：一键构建、测试和运行示例

## 怎么看这章的代码

建议按下面顺序看：

1. `request` 和 `generation_types`
   - 先理解请求状态怎么表示
2. `kv_cache`
   - 再看历史 K/V 怎么跟请求绑定
3. `transformer`
   - 然后看 `prefill / decode` 如何从结构层进入运行时
4. `runtime`
   - 最后看 runtime 如何把状态、cache 和 sampler 串起来
5. `scheduler`
   - 再看多请求场景如何推进

## 当前实现的重点

- `RequestPhase` 已经显式区分 `Initial`、`Prefilled`、`Finished`
- `prefill` 和 `decode` 已经分开
- `KVCache` 以最小形态出现，但已经能观察缓存增长
- `RoundRobinScheduler` 负责把活跃请求推进下去
- `SchedulerStats` 负责把运行过程显式化
- `streaming callback` 已经作为最小观察入口接入

## 怎么运行

最简单的方式是：

```bash
bash scripts/run.sh
```

这个脚本会依次完成：

1. CMake 配置
2. 编译
3. 运行主 demo
4. 运行测试
5. 运行 streaming demo

## 测试关注点

- `runtime_smoke_test`
  - 验证请求状态机是否能正常初始化和推进
- `kv_cache_test`
  - 验证 cache 的增长和清理行为
- `sampler_test`
  - 验证采样器能把 logits 转成 token
- `chapter05_scheduler_test`
  - 验证多请求 round-robin 调度
- `chapter05_generation_test`
  - 验证完整 generation runtime 闭环

## 当前状态

- 这一章已经具备最小但完整的 generation runtime
- 现在的重点不是再加复杂服务能力，而是把这条链路写得更清楚、更可验证
- 后续最自然的增量是：
  - continuous batching
  - streaming control
  - backpressure
  - request prioritization

## 和教程主线的关系

- `chapter04_attention` 提供 decoder 结构
- `chapter05_generation` 把 decoder 结构变成运行时
- `chapter06_quantization` 和 `chapter07_benchmark` 会继续复用这条 runtime

