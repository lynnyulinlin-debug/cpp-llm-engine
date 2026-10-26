# 08.5 A/B Experiment A/B 对比实验

优化结论必须能复查。A/B 实验的作用是把“我觉得快了”变成“在同一条件下，这个改动带来了可解释变化”。

**阅读入口**

- 前置阅读：[`07.6 Result Interpretation`](../chapter07_benchmark/06_result_interpretation.md)、[`08.2 Latency Breakdown`](./02_latency_breakdown.md)
- 源码直达：<a href="../_code/chapter07_benchmark/examples/main_benchmark.cpp"><code>main_benchmark.cpp</code></a>、<a href="../_code/chapter07_benchmark/scripts/run.sh"><code>run.sh</code></a>、<a href="../_code/chapter07_benchmark/tests/chapter07_benchmark_test.cpp"><code>chapter07_benchmark_test.cpp</code></a>
- 相关阅读：[`回归测试`](../reference/regression_testing.md)、[`08.6 Optimization Loop`](./06_optimization_loop.md)

## 实验模板

- 只改变一个变量
- 固定模型、输入、batch、硬件和运行参数
- 同时记录正确性和性能
- 记录足够的环境信息
- 预先写清楚保留或回滚标准

如果一次实验同时改 scheduler、cache 和 kernel，就算结果变快，也很难判断到底是哪一项生效。

### 结果模板

| 项目 | baseline | experiment | 结论 |
| --- | --- | --- | --- |
| correctness | pass / fail | pass / fail | 是否可继续比较 |
| prefill latency | 数值 | 数值 | 是否改善 |
| decode latency | 数值 | 数值 | 是否改善 |
| throughput | 数值 | 数值 | 是否改善 |
| memory | 数值 | 数值 | 是否可接受 |
| complexity | 原实现 | 新实现 | 是否值得维护 |

这个表的价值不是漂亮，而是逼迫优化结论同时回答收益和代价。

### 一个可直接套用的实验记录

```markdown
## A/B Report

### 条件
- 硬件：
- 模型 / dtype：
- prompt length：
- output length：
- batch size：
- commit：

### 改动
- baseline：
- experiment：
- 只改变的变量：

### 结果
| 指标 | baseline | experiment | delta |
| --- | --- | --- | --- |
| correctness | pass | pass | - |
| prefill latency |  |  |  |
| decode latency |  |  |  |
| throughput |  |  |  |
| peak memory |  |  |  |

### 决策
- 保留 / 回滚 / 继续拆分：
- 理由：
```

### 用当前基线先写一份 baseline 记录

第一次补实验记录时，不要等“有优化之后”再写。先把 baseline 写出来：

```markdown
## A/B Report

### 条件
- 硬件：本地 CPU baseline
- 模型 / dtype：chapter07 minimal demo
- prompt length：5 tokens
- output length：2 tokens
- batch size：1
- commit：当前工作树

### 改动
- baseline：chapter07_benchmark_demo
- experiment：暂空
- 只改变的变量：暂空

### 结果
| 指标 | baseline | experiment | delta |
| --- | --- | --- | --- |
| correctness | pass | - | - |
| throughput | 4000 items/s | - | - |
| prefill latency | 0.30 ms | - | - |
| decode latency | 0.15 ms | - | - |
| sampling latency | 0.05 ms | - | - |
| output latency | 0.05 ms | - | - |
| peak memory | 未记录 | - | - |

### 决策
- 保留 / 回滚 / 继续拆分：继续拆分
- 理由：当前只有 baseline，还没有单变量实验
```

这份记录先固定后续实验需要的字段。下一步如果你改的是 scheduler 或 cache，就必须看它影响的是 `prefill`、`decode`，还是只是 `output`。

## 结构化记录

如果后续要把 benchmark / profiling 结果长期保存下来，纯文本摘要还不够。当前教程基线已经可以进一步导出结构化 JSON 样例，至少包含三层：

- `phases`：阶段汇总
- `trace_events`：事件明细
- `hotspots`：热点排序

这样做的意义是，A/B 记录不再只是手工摘抄，而是可以从统一结构里回填报告、比对版本，或者转成更接近 profiler UI 的视图。

下面这类结构，就是当前教程基线已经能导出的最小样例：

```json
{
  "benchmark": {
    "name": "chapter07_demo",
    "items_per_second": 4000.00,
    "phases": [
      {"phase": "input", "tokens": 5, "elapsed_ms": 0.05},
      {"phase": "prefill", "tokens": 5, "elapsed_ms": 0.30},
      {"phase": "decode", "tokens": 2, "elapsed_ms": 0.15}
    ]
  },
  "trace_events": [
    {"phase": "prefill", "event": "run_prefill_attention", "start_ms": 0.13, "duration_ms": 0.14},
    {"phase": "decode", "event": "decode_step", "start_ms": 0.30, "duration_ms": 0.09}
  ],
  "hotspots": [
    {"phase": "prefill", "symbol": "run_prefill_attention", "self_ms": 0.14, "total_ms": 0.17},
    {"phase": "decode", "symbol": "decode_step", "self_ms": 0.09, "total_ms": 0.15}
  ]
}
```
