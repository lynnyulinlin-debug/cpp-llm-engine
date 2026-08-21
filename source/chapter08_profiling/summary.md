# 08.8 Chapter Wrap-up 本章解决了什么

这章回答的是：benchmark 之后，如何解释性能变化，并把优化变成可复查的工程闭环。

## 本章解决了什么

- benchmark 和 profiling 的职责已经分开
- prefill、decode、sampling、output 的阶段边界已经明确
- 内存、带宽、kernel launch 和调度开销已经有了归因入口
- A/B 实验和优化保留标准已经显式化
- vLLM / nano-vLLM 的性能优化可以映射回本教程主线

## 为什么重要

`chapter07` 让结果可比较，本章让原因可解释。没有这一步，优化很容易变成“看到哪里慢就改哪里”，最后缺少可复查的收益和边界。

## 下一步

如果继续扩展教程，下一步应该把 profiling 记录、A/B 报告和 benchmark 输出统一成一种长期可维护格式，并把它接回 `chapter07` 的证据链。当前章已经补入可运行的 `chapter07_benchmark_demo` 基线输出、阶段计时、最小事件级 trace、hotspot 汇总、结构化 JSON 样例、简化时间轴视图、最小 `tid / sync point` 语义，以及一个多请求时间轴样例；继续扩展时应非常克制，避免越过教程边界进入完整 profiler 工具实现。

## 本章产物与验证

本章结束时，应能产出一份包含以下内容的优化记录：

- 一个可复现的 benchmark 场景
- 一张 prefill / decode / sampling / output 拆解表
- 一条 profiling 证据
- 一组 baseline / experiment A/B 数据
- 一个明确的保留或回滚决策

## 复盘路径

- 先回看 [`chapter08 导读`](./intro.md)
- 再看 [`08.5 A/B Experiment`](./05_ab_experiment.md)
- 最后看 [`08.7 vLLM / nano-vLLM Profiling 阅读桥`](./07_vllm_nano_vllm_profiling_bridge.md)
