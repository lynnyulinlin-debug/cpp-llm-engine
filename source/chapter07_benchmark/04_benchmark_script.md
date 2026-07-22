# 07.4 Benchmark Script Benchmark 脚本


## 这一页要解决什么

编写 `benchmark.sh`，自动运行 benchmark 并记录硬件与运行信息。

## 读完后应能做什么

- 理解为什么 benchmark 需要自动化
- 理解应该记录哪些环境信息
- 理解脚本如何支持重复测量

## 代码直达

- [`scripts/run.sh`](/code/chapter07_benchmark#run-sh)
- [`examples/main_benchmark.cpp`](/code/chapter07_benchmark#main-benchmark-cpp)
- [`tests/chapter07_benchmark_test.cpp`](/code/chapter07_benchmark#chapter07-benchmark-test-cpp)

## 为什么这一步重要

没有脚本化的 benchmark 很难复现，也很难比较不同版本的性能。

## 这一页的核心内容

- 自动运行测试
- 记录硬件信息
- 保存运行结果
- 输出可比较的 benchmark 报告

脚本化 benchmark 的价值在于把“运行条件”固定下来。没有固定输入、固定模型、固定环境信息，测出来的性能数据很难真正比较。

因此脚本不仅要跑测试，还要把版本、硬件和配置一起写进结果里。

建议脚本输出同时包含两种格式：一种给人看，另一种给后续自动分析用。只留文本日志的话，后面做回归比较会很费劲。

脚本流程可以先写成：

```bash
#!/usr/bin/env bash
set -euo pipefail
./run_tests
./run_benchmark > benchmark.json
```

在本教程中，`run_tests` 和 `run_benchmark` 对应的是 `build/` 目录下的测试与 benchmark 可执行文件，比如 `./build/bin/test_all` 和 `./build/bin/benchmark_matmul`。脚本应当位于项目根目录，并默认构建已经完成，或者由脚本内部自动触发构建。

## 实现建议

- 让脚本能独立执行
- 固定输入和配置
- 把环境信息和结果一起保存

## 怎么验证

- 脚本能自动完成一次测量
- 硬件信息被正确记录
- 结果可重复生成

如果同一脚本在同一环境里多次运行，输出结果格式稳定且便于横向比较，这一页就达标了。

## 常见问题

- 测量条件没有固定
- 输出结果没有归档
- 硬件信息采集不完整
