# F07 Common CUDA Libraries CUDA 常用库

## 目标

给读者一个可以直接用于教程路线判断的 CUDA 库地图，而不是一份松散的名词清单。

## 读完你应该能做什么

- 理解不同 CUDA 库分别解决什么问题
- 理解为什么这套教程先关注 `cuBLAS`
- 理解哪些库是可选的，哪些是推理系统里常见的

## 为什么重要

刚接触 CUDA 推理代码时，库名往往比代码本身更让人迷糊。这里要解决的不是“背全生态”，而是“知道在这套教程里，哪些名字对应哪类工程问题”。

这章不是完整生态导览，而是一个面向主线章节的决策地图：先知道谁负责 matmul，谁负责通信，谁负责 launch overhead，再决定哪些工具现在需要，哪些可以后面再看。

## 后面会用到什么

- `cuBLAS`：矩阵乘法
- `CUDA Graph`：减少 launch overhead
- `memory pool` 风格的分配管理
- 以及对 `cuDNN`、`CUTLASS`、`Thrust`、`CUB`、`NCCL` 的可选认识

## 在教程里哪里会出现

- `chapter03_matmul`
  - `cuBLAS` 是 dense matrix multiplication 的主要 GPU backend 参考
  - 如果你想看自定义 GPU kernel 的思路，`CUTLASS` 是一个可选参考

- `chapter05_generation`
  - 如果以后扩展到多 GPU serving，`NCCL` 才会变得相关

- `chapter07_benchmark`
  - `CUDA Graph` 会作为 launch overhead 优化思路出现
  - `memory pool` 会作为内存管理优化思路出现
  - `async copy` 会作为数据移动优化思路出现

## 章节关系

| 章节 | 关注层 | 问题 |
| --- | --- | --- |
| `00.3` | 环境层 | GPU 可见吗？driver / runtime 可用吗？ |
| `00.7` | 库层 | `cuBLAS` 解决什么问题？`CUDA Graph` 解决什么问题？ |
| `chapter03_matmul` | 应用层 | 如何用 `cuBLAS` 实现 MatMul？ |

三层连起来就是一条完整路径：环境检查 -> 库选择 -> 代码实现。

## 建议阅读顺序

1. 先看 `cuBLAS`，因为它最接近教程里的 `MatMul` backend 路径。
2. 再看 `CUDA Graph`，如果你想理解后面的性能优化思路。
3. `memory pool`、`CUTLASS` 和 `NCCL` 可以按兴趣作为扩展阅读。

## 这一页的核心内容

### 1. 先把运行时边界分清

runtime 负责让程序发起 GPU 工作，driver 负责让 GPU 对系统可见并可用。

在教程里，这个边界很重要，因为后续所有“库选择”都建立在它之上。先把 runtime 和 driver 混在一起，后面的错误定位就会偏。

### 2. `cuBLAS` 对应 dense matmul 主路径

`cuBLAS` 是 NVIDIA GPU 上的 dense linear algebra 标准库。在这套教程里，它是最直接相关的库，因为 `MatMul` 通常会映射到 GEMM 风格执行。

这也是它为什么会在 CUDA 故事里出现得比较早：一旦知道 baseline matmul 路径，`cuBLAS` 就是自然的高性能对照点。

对本教程来说，`cuBLAS` 不是“GPU 库之一”，而是 `chapter03_matmul` 的主 backend 参照。

### 3. `CUDA Graph` 对应重复 launch 的收口

`CUDA Graph` 可以捕获重复的 GPU launch 模式，从而减少 launch overhead。它通常属于性能优化，不是第一步就该学的东西。

在这套教程里，它更适合放在 `chapter07_benchmark` 的语境里看，而不是拿来替代主线实现。

### 4. `cuDNN`、`CUTLASS`、`Thrust`、`CUB`、`NCCL` 分别对应不同扩展点

- `cuDNN`：深度学习 primitive
- `CUTLASS`：自定义高性能 kernel 参考
- `Thrust`：GPU-side 算法原语
- `CUB`：更底层的并行原语
- `NCCL`：多 GPU 通信

这些名字不需要一次读完，但需要知道它们分别在哪类问题上出现。

更具体一点，可以这样理解它们在这套教程里的角色：

- `cuDNN`：不作为主线依赖，适合当未来扩展到 CNN 等模型时的参考
- `CUTLASS`：不直接依赖，但有助于理解自定义 GPU kernel 的实现生态
- `Thrust`：不直接使用，但如果你需要在 GPU 侧做排序、归约等 STL 风格操作，它是一个参考选项
- `CUB`：不直接使用，但它是更底层的并行原语库，理解它有助于看懂 `CUTLASS` 之类库的依赖链
- `NCCL`：只在多 GPU 通信或分布式场景变得相关

### 5. 选择顺序应该按问题，而不是按名气

这套教程里，库的优先级不是“谁更强”，而是“谁最接近当前章节的问题”。

所以顺序应当是：

- 先 `cuBLAS`
- 再 `CUDA Graph`
- 最后按需要看 `cuDNN`、`CUTLASS`、`Thrust`、`CUB`、`NCCL`

### 6. 异步拷贝是优化手段，不是基础依赖

`async copy` 的作用是让数据搬运和 kernel 执行尽量重叠，从而隐藏一部分数据移动延迟。它不是教程的核心功能，而是 `chapter07_benchmark` 里讨论性能优化时才会用到的概念。

也就是说，它属于“怎么更快”，不是“怎么先跑起来”。

## 最小示例

```text
教程中的关注点：

- `cuBLAS` -> dense matmul / GEMM
- `CUDA Graph` -> 重复 launch overhead 的降低
- `cuDNN` -> 可选 deep-learning primitive
- `CUTLASS` -> 自定义 kernel 学习 / 扩展
- `Thrust` -> GPU-side 排序 / 归约等算法原语
- `CUB` -> 更底层的并行原语
- `NCCL` -> 多 GPU 通信
```

这个地图足够回答两个问题：

- 我应该先学哪个库？
- 哪些库现在可以先忽略？

如果只记住一个映射，那就记住这句：`cuBLAS` 对应 dense matmul，`CUDA Graph` 对应 launch overhead，其余是按需要再看的扩展工具。

## 怎么验证

- 解释 `cuBLAS` 在这套教程里是做什么的
- 解释为什么 `CUDA Graph` 出现得更晚，而不是更早
- 解释哪些库对当前主线是可选的
- 解释每个库对应教程里的哪一章
- 解释为什么先看这个库而不是另一个
- 解释当前问题是在算子层、runtime 层还是通信层

## 常见问题

- 把每个 CUDA 库都看成同样重要
- 还没理解基本 kernel 路径，就先去学 `CUDA Graph`
- 把 `cuBLAS` 和 CUDA runtime 本身混在一起
- 以为单 GPU 教程就一定需要多 GPU 库
- 把“知道名字”当成“理解问题”

## 总结

读完这一章后，读者应该知道 CUDA 常用库的大致地图，并理解为什么这套教程先关注 `cuBLAS`，同时把其他库作为可选参考保留。

如果你能解释为什么每个库会出现在教程路线图里，那么 CUDA 生态对后面的学习路径就已经足够清楚了。
