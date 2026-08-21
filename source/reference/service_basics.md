# 推理引擎服务化基础

这一页专门讲“推理引擎什么时候开始不只是本地程序，而是一个持续服务”。  
主线在 `chapter05_generation` 和 `chapter07_benchmark` 里已经碰到一些服务化语义，这一页把它们收成最基础的运行规则。

## 1. 这页要解决什么

如果系统已经有多请求、流式输出和资源复用，那么它就不再只是一个 demo。  
这时就必须开始回答服务化问题。

这一页只回答一个问题：最基础的服务化能力有哪些，以及它们如何落到 `Request`、`Scheduler` 和资源回收边界上。

## 2. 服务化最小集合

一个推理引擎至少要考虑这些基础能力：

- 限流
- SLA
- 请求取消
- 健康检查
- 日志
- 指标

这些能力是系统开始对外提供持续服务时必须具备的东西。

## 3.1 最小服务契约

服务层不应该直接修改模型计算状态，而是把外部事件转换成 runtime 能处理的请求动作：

| 外部事件 | 服务层动作 | runtime 结果 |
| --- | --- | --- |
| 新请求 | 校验输入、检查容量、进入 waiting | `Request::waiting` |
| 队列过载 | 拒绝、排队或返回明确的限流结果 | 不进入执行队列 |
| 客户端取消 | 标记取消并通知 scheduler | `Request::aborted` |
| 超时 | 终止输出并触发资源回收 | `Request::aborted` 或 timeout finish reason |
| 正常完成 | 关闭流、记录指标、归还 cache | `Request::finished` |

这条边界把 API、调度和计算解耦：服务层负责接入和策略，`Scheduler` 负责推进顺序，`Runtime` 负责一次执行，资源管理层负责 KV Cache 和 workspace 的释放。

### 回压和取消的共同约束

回压不是简单地让队列无限变长。达到容量上限时，系统必须明确选择拒绝、等待或降级，并让调用方能看到结果。取消也不能只停止网络输出；它必须阻止请求再次进入调度，并释放已占用的 cache、workspace 和流式输出状态。

## 3. 限流和 SLA

限流解决的是“请求太多时系统怎么自保”。  
SLA 解决的是“系统承诺的响应边界是什么”。

没有这两项，系统很容易在高峰时把资源吃满，或者让延迟失去上限。

### 最小理解方式

- 限流：控制同时进入系统的请求数量
- SLA：控制用户能预期到的响应时限和服务边界

这两者一起决定系统是否可控。

## 4. 请求取消

请求取消不是边角功能，而是 runtime 和服务系统必须共享的行为。

它至少要做到：

- 用户不需要了可以停
- 已经占住的资源可以释放
- 取消后不会把 cache 和队列状态弄乱

如果取消和资源回收没打通，系统就会出现“逻辑上已经结束，资源上还在占着”的问题。

## 5. 健康检查、日志和指标

这三项决定系统是不是能被持续观察：

- 健康检查：系统还活着吗
- 日志：发生了什么
- 指标：系统整体运行得怎么样

推理引擎只要开始服务多个请求，就必须让这些信息可见。  
否则出了问题，只能猜，不能定位。

## 6. 和主线章节的关系

| 章节 | 已经做了什么 | 服务化页继续补什么 |
| --- | --- | --- |
| `chapter05_generation` | 最小 generation runtime、streaming callback | 如何把 runtime 变成可持续服务 |
| `chapter07_benchmark` | 验证、对照和结果收口 | 服务化场景下如何读结果 |
| `reference/runtime_systems.md` | 请求生命周期、continuous batching、backpressure | 面向外部服务的最小规则 |

这页不是要把完整 production serving 写完，而是先把最基础的服务语义说清楚。

相关阅读：[`05.9 Request 生命周期`](../chapter05_generation/08_request_lifecycle.md)、[`05.10 Continuous Batching 入门`](../chapter05_generation/09_continuous_batching.md)、[`推理引擎内存管理`](./memory_management.md)、[`推理引擎回归测试`](./regression_testing.md)

## 7. 使用顺序

- 先看 `reference/runtime_systems.md`
- 再看 `reference/inference_engine_systems.md`
- 系统开始对外接请求后，用这一页检查服务边界是否清楚

如果只保留一句话：  
服务化基础决定了引擎是不是能持续对外工作。
