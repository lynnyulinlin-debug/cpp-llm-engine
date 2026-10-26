# 08.5 A/B Experiment A/B 对比实验

优化结论需要同时说明条件、收益和代价。A/B 实验固定模型、输入和环境，每次只改变一个变量，再用正确性闸门决定性能数据是否可比较。

**阅读入口**

- 前置阅读：[`07.6 Result Interpretation`](../chapter07_benchmark/06_result_interpretation.md)、[`08.2 Latency Breakdown`](./02_latency_breakdown.md)
- 源码直达：<a href="../_code/chapter07_benchmark/examples/main_benchmark.cpp"><code>main_benchmark.cpp</code></a>、<a href="../_code/chapter07_benchmark/scripts/run.sh"><code>run.sh</code></a>、<a href="../_code/chapter07_benchmark/tests/chapter07_benchmark_test.cpp"><code>chapter07_benchmark_test.cpp</code></a>
- 相关阅读：[`回归测试`](../reference/regression_testing.md)、[`08.6 Optimization Loop`](./06_optimization_loop.md)

## 实验契约

一次可解释的实验按以下顺序执行：

1. 写明假设和唯一变量，例如只替换 Q4_K Linear backend。
2. 固定模型、dtype、prompt、output length、batch、硬件和构建类型。
3. 先运行 correctness、golden 和资源释放检查。
4. 采用相同 warmup、迭代次数和统计方式采集两组性能。
5. 根据预先约定的阈值决定保留、回滚或继续拆分。

同时修改 scheduler、cache 和 kernel 时，结果只能说明组合发生了变化，无法归因到单个实现。组合实验应放在各单变量实验通过之后。

| 项目 | Baseline | Experiment | 判定问题 |
| --- | --- | --- | --- |
| Correctness | pass / fail | pass / fail | 两组结果是否可比较 |
| Prefill / decode | 延迟与波动 | 延迟与波动 | 改动影响哪个阶段 |
| Throughput | token/s | token/s | 稳态产出是否提高 |
| Memory | 峰值与稳定值 | 峰值与稳定值 | 资源代价是否可接受 |
| Complexity | 原实现 | 新实现 | 收益是否覆盖维护成本 |

## 记录模板

实验开始前创建记录并填写条件与保留阈值，运行后再填写结果。`commit`、编译器、构建类型和 CPU/GPU 型号用于判断两组数据是否来自同一环境。

```markdown
# A/B Report

## 条件
- 硬件 / 编译器 / 构建类型：
- 模型 / dtype：
- prompt / output / batch：
- warmup / iterations：
- commit：

## 唯一变量
- baseline：
- experiment：

## 结果
| 指标 | baseline | experiment | delta |
| --- | ---: | ---: | ---: |
| correctness | pass | pass | 可比较 |
| prefill latency |  |  |  |
| decode latency |  |  |  |
| throughput |  |  |  |
| peak memory |  |  |  |

## 决策
- KEEP / REVERT / SPLIT：
- 阈值与理由：
```

## 从 Baseline 开始

尚未修改实现时，先运行 `bash code/chapter07_benchmark/scripts/run.sh`，登记当前输出。下一次实验复用相同字段并补上 experiment，避免优化完成后才回忆环境。

下面的数值是 Chapter07 minimal demo 的格式示例，不是跨机器性能承诺：

| 条件或指标 | Baseline |
| --- | ---: |
| workload | prompt 5 tokens、output 2 tokens、batch 1 |
| correctness | pass |
| throughput | `4000 items/s` |
| prefill latency | `0.30 ms` |
| decode latency | `0.15 ms` |
| sampling / output | `0.05 / 0.05 ms` |
| peak memory | 待采集 |
| 当前决策 | `SPLIT`，等待单变量实验 |

## 结构化结果

长期保存的结果使用 JSON，而不是从日志文本反向提取数字。最小结构包含 `benchmark.phases`、`trace_events` 和 `hotspots`；比较器据此生成报告，并在 workload 或环境字段不匹配时拒绝计算 delta。

```json
{
  "benchmark": {
    "name": "chapter07_demo",
    "items_per_second": 4000.0,
    "phases": [
      {"phase": "prefill", "tokens": 5, "elapsed_ms": 0.30},
      {"phase": "decode", "tokens": 2, "elapsed_ms": 0.15}
    ]
  },
  "trace_events": [
    {"phase": "prefill", "event": "run_prefill_attention", "duration_ms": 0.14}
  ],
  "hotspots": [
    {"phase": "prefill", "symbol": "run_prefill_attention", "self_ms": 0.14}
  ]
}
```
