# 算子、Backend、Executor 与 Runtime

这一页专门把推理引擎里最容易混在一起的四层边界拆开：`operator`、`backend`、`executor`、`runtime`。  
主线已经分别在 `chapter03_matmul` 和 `chapter05_generation` 里碰到这些概念，但如果不把它们显式拆开，读者很容易把“能跑”与“该放在哪一层”混成一件事。

## 1. 这页要解决什么

这页只回答一个问题：一个推理引擎里，谁负责语义，谁负责执行，谁负责调度，谁负责持续服务。

如果四层边界不清楚，后面最容易出现的问题是：

- 上层直接依赖具体实现
- 后端和上层语义绑死
- 调度逻辑散进算子里
- runtime 里混进了太多实现细节

完整系统中的位置见 [推理引擎系统补强图](./inference_engine_systems.md)。本页只展开其中的计算与执行边界。

## 2. 四层分别是什么

### operator

`operator` 负责的是**语义**。

它要回答的是：

- 这个计算做什么
- 输入输出是什么
- shape 和布局约束是什么
- 错误边界在哪里

在教程里，`MatMul` 就是最典型的 `operator`。

### backend

`backend` 负责的是**具体实现路径**。

它要回答的是：

- 这件事用 CPU 还是 GPU 做
- 用参考实现还是库实现
- 用 cuBLAS 还是未来其他 kernel

backend 不应该改写上层语义，只应该实现同一套语义。

### executor

`executor` 负责的是**把某次执行真正跑起来**。

它要回答的是：

- 这次执行调用哪个 backend
- 输入怎么喂进去
- 输出怎么拿出来
- 这次执行的资源边界是什么

你可以把它理解成语义和运行之间的中间层，它把“我要做什么”变成“这一次具体怎么做”。

### runtime

`runtime` 负责的是**持续服务和状态推进**。

它要回答的是：

- 请求怎么进入系统
- `prefill` 和 `decode` 怎么推进
- `KV Cache` 怎么复用
- scheduler 怎么决策
- 多请求怎么持续工作

runtime 关心的是系统，不只是单次执行。

## 3. 一条最小分工链

| 层 | 主要职责 | 典型问题 |
| --- | --- | --- |
| `operator` | 定义语义边界 | 这一步算的是什么 |
| `backend` | 实现具体计算 | 这一步怎么高效算 |
| `executor` | 执行一次任务 | 这一步用哪条路径跑 |
| `runtime` | 持续推进请求 | 系统怎么一直服务下去 |

这四层不是等价替代，而是从“抽象”往“运行”逐层落地。

## 4. 和主线章节的关系

| 章节 | 已经做了什么 | 这四层里更偏哪一层 |
| --- | --- | --- |
| `chapter03_matmul` | 算子边界、CPU reference、cuBLAS | `operator` + `backend` |
| `chapter05_generation` | 生成链路、request、scheduler、cache | `runtime` |
| `chapter07_benchmark` | 正确性和性能收口 | `executor` + `runtime` 的验证结果 |

如果再往回看，`chapter03` 解决的是“语义怎么固定”，`chapter05` 解决的是“系统怎么持续推进”。  
这页补的是两者之间的层次感：谁该放在哪一层。

## 5. 读者应该如何使用这一页

- 先把 `chapter03_matmul` 和 `chapter05_generation` 读完
- 再回到这里看哪些概念其实不是同一层
- 如果后续要扩展 backend、executor 或 runtime，就先对齐这一页的分工

如果只保留一句话：  
`operator` 定义做什么，`backend` 定义怎么做，`executor` 定义这次怎么跑，`runtime` 定义系统怎么持续跑。
