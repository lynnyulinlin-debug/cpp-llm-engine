# 使用说明

这个仓库面向学习者，是一条教学路径。

## 三个入口怎么分工

- `foundation`：补主线前的基础认知
- `chapter00_quickstart`：尽快跑通第一个可执行基线
- `USAGE.md`：告诉你从哪条路径开始读、怎么继续读

`USAGE.md` 负责路线图，`chapter00_quickstart` 负责起跑线。前者告诉你先看什么、后看什么；后者带你把第一条真实推理链路跑通。

如果你不确定自己该从哪里开始，先看这里：

- 先读 `USAGE.md`，确定阅读路径
- 再按需要进入 `foundation`
- 最后进入 `chapter00_quickstart` 和后续主线章节

## 从这里开始

1. 先看 [docs/README.md](docs/README.md) 了解教程总图
2. 如果你更习惯英文导航，可以看 [docs/README.en.md](docs/README.en.md)
3. 需要 C++、构建工具或 CUDA 背景时，先看 `foundation`
4. 想先跑出第一个可执行基线时，先看 `chapter00_quickstart`
5. 然后按照 `chapter01` 到 `chapter07` 的主线往下读

## 阅读路径

- 入门路径：`foundation` -> `chapter00` -> `chapter01` -> `chapter02` -> `chapter03` -> `chapter04` -> `chapter05` -> `chapter06` -> `chapter07`
- 工程路径：`chapter00` -> `chapter01` -> `chapter02` -> `chapter03` -> `chapter04` -> `chapter05` -> `chapter07`
- 性能路径：`chapter02` -> `chapter03` -> `chapter04` -> `chapter05` -> `chapter07`
- 部署路径：`chapter06` -> `chapter07`

## 每章作用

- `chapter00_quickstart`：验证本地推理基线
- `chapter01_tensor_loader`：加载张量数据和模型权重
- `chapter02_tensor_ops`：学习视图语义和存储共享
- `chapter03_matmul`：构建核心矩阵乘法算子
- `chapter04_attention`：组装 decoder block
- `chapter05_generation`：构建生成运行时
- `chapter06_quantization`：增加量化推理路径
- `chapter07_benchmark`：验证正确性并比较性能

## 如何运行代码

优先使用各章节目录下的脚本：

- `bash code/chapter00_quickstart/run.sh`
- `bash code/chapter02_tensor_ops/scripts/run.sh`
- `bash code/chapter03_matmul/scripts/run.sh`
- `bash code/chapter04_attention/scripts/run.sh`
- `bash code/chapter05_generation/scripts/run.sh`
- `bash code/chapter06_quantization/scripts/run.sh`
- `bash code/chapter07_benchmark/scripts/run.sh`

对于 `chapter01_tensor_loader`，使用本地 CMake 构建：

```bash
cmake -S code/chapter01_tensor_loader -B code/chapter01_tensor_loader/build
cmake --build code/chapter01_tensor_loader/build
ctest --test-dir code/chapter01_tensor_loader/build --output-on-failure
```

## 测试怎么理解

- `chapter00_quickstart`：用脚本和命令行确认最小推理链路能跑通
- `chapter01` 到 `chapter06`：主要是章节内最小测试和 `assert` 级验证
- `chapter07`：用统一的测试 harness 收口单测、端到端验证和 benchmark 结果

如果你关心的是“当前章节有没有被正确验证”，优先看对应章节的 `summary.md`。  
如果你关心的是“仓库整体测试体系现在长什么样”，优先看 [docs/maintenance/tutorial_maintenance.md](docs/maintenance/tutorial_maintenance.md)。

## 参考层

- `llama.cpp`：本地推理基线
- `vLLM`：高吞吐 serving 设计
- `SGLang`：结构化 serving 和编排
- `TensorRT-LLM`：GPU 引擎和部署设计

在理解主线章节之后，再看这些参考层。它们是设计对照点，不是替代教程主线的第二条阅读路径。

## 如果你想扩展教程

- 保持主线稳定
- 把新内容放在桥接文字、进阶说明或附录页面里
- 只用 benchmark 做比较，不用它证明正确性
