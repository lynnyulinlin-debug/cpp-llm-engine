# 05.5 Scheduler 调度器


## 决定下一个 Request

把“谁先推进、谁后推进、什么时候推进”从 runtime 的隐式行为，变成显式的调度规则。  
先定义最小调度器，不展开复杂批处理。

**阅读入口**

- 前置阅读：[`05.4 KV Cache`](./04_kv_cache.md)、[`05.3 Naive Generation`](./03_naive_generation.md)
- 源码直达：<a href="../_code/chapter05_generation/src/scheduler.cpp"><code>scheduler.cpp</code></a>、<a href="../_code/chapter05_generation/src/request.cpp"><code>request.cpp</code></a>、<a href="../_code/chapter05_generation/tests/chapter05_scheduler_test.cpp"><code>chapter05_scheduler_test.cpp</code></a>
- 相关阅读：[`05.6 Sampler`](./05_sampler.md)、[`05.7 端到端生成测试`](./06_end_to_end_generation.md)

## 核心设计

当 runtime 只有单个请求时，推进顺序看起来像是“自然发生”的。  
一旦进入多请求场景，系统就必须明确回答：当前该推进谁？

如果调度规则不显式，cache 和 sampler 都会被迫在隐式上下文里工作，性能、延迟和公平性也就很难判断。

最小调度器先只做一件事：在活跃请求里选出下一条要推进的请求。

可以先把它理解成一个简单的队列调度器：

- 请求进入队列
- 调度器选择一个请求推进
- 推进结束后决定是否重新入队

下面的 Python 镜像对应当前 `RoundRobinScheduler::run_with_stats()`，只表达状态推进，不拥有 Request、Executor 或 KV 资源：

```python
def run_round_robin(runtime, scheduled_requests, stream_token):
    for item in scheduled_requests:
        if item.request is not None and item.executor is None:
            runtime.prefill(item.request)

    progress = True
    while progress:
        progress = False
        for item in scheduled_requests:
            request = item.request
            if request is None or item.sampler is None or request.terminal():
                continue

            generated_before = request.generated_size()
            decoded = runtime.decode_one(request, item.sampler, item.executor)
            if decoded == 0:
                continue

            progress = True
            if request.generated_size() > generated_before:
                stream_token(item.request_id, request.generated[-1])
```

最小调度器不追求复杂策略，只先把“推进顺序可观察、可测试”立住。

## Admission 和优先级边界

真实服务不能只决定“下一个是谁”，还要先决定“谁可以进入运行集合”。最小实现可以使用 FIFO，但接口应保留 admission policy 的位置：

1. 检查请求状态、输入长度和可用 token/cache budget。
2. 容量不足时保留在 waiting、拒绝或返回限流结果，不能无条件进入 running。
3. 取消或超时的请求在 admission 前直接丢弃，不得重新排队。
4. 进入 running 后，每个 step 只推进预算内的 token，并在终态释放资源。

优先级队列属于后续策略，不属于当前 FIFO 基线。引入优先级时必须同时定义公平性、饥饿保护和取消后的队列清理，不能只替换一个比较器。

调度器和请求状态的边界：

调度器不负责计算 logits，也不负责维护 K/V。  
它只负责决定哪个请求进入下一轮 runtime 推进。

这样可以把职责拆开：

- `Request` 保存状态
- `Request::cache()` 保存 token ID 历史
- `Scheduler` 决定推进顺序
- `RequestModelExecutor` 保存单个请求的模型 Context、分层 K/V 并提供 logits
- `Sampler` 决定下一步 token

如果这四件事混在一起，后面很难做单独验证。

最小实现可以先用 round-robin：

1. 从队列头取一个请求
2. 推进一次
3. 如果请求没结束，放回队尾
4. 如果请求结束，释放它的 runtime 资源

这个策略的优点是容易理解、容易验证，也容易和单请求路径对照。  
它不是最终优化方案，但足够作为第一个可观察调度模型。

## 每请求 Model Executor

真实多请求不能让 Scheduler 反复调用同一个带状态 Decoder。`ScheduledRequest` 可以携带一个可选的 `shared_ptr<RequestModelExecutor>`；ServiceCore 从 admission 持有它，Scheduler 只在该 Request 被选中时调用它，terminal cleanup 后随 entry 一起释放。

| 调度阶段 | Runtime 动作 | Executor 动作 |
| --- | --- | --- |
| 首次推进 | 校验 context budget，写入 prompt token history | `prefill(prompt)`，建立所属 Context 的分层 K/V |
| 后续推进 | 更新 Request、调用 Sampler、检查 EOS/长度 | `decode(last_token)`，只推进所属 Context |
| 执行错误 | Request 进入 `Failed`，停止调度 | 返回错误，不产生占位 token |
| terminal cleanup | 清理 Request cache，ServiceCore 移除 entry | `shared_ptr` 释放请求 Context |

### 算法对象映射

Python 镜像中的对象与 C++ 接口保持一一对应，方便从控制流回到 ownership 和错误路径。

| Python 对象或条件 | C++ 对应物 | 测试证据 |
| --- | --- | --- |
| `scheduled_requests` | `std::vector<ScheduledRequest>` | 多请求输入顺序固定 |
| `request.terminal()` | `Request::terminal()` | 完成、取消、失败请求不再推进 |
| `runtime.decode_one(...)` | `Runtime::decode_one()` | prefill/decode 与 Executor 隔离 |
| `decoded == 0` | 无进展分支 | 不产生虚假 token，不触发 streaming |
| `progress` | 外层 round-robin 终止条件 | 全部请求无进展时退出 |

### 与 Runtime 的窄接口

和 runtime 的接口：

调度器通常会和 runtime 通过一个很窄的接口交互：

```cpp
class Scheduler {
public:
    Request* next();
    void submit(Request req);
    void complete(RequestId id);
};
```

这个接口的关键是“只暴露调度动作，不暴露内部队列细节”。  
这样 runtime 只需要关心：谁该被推进、谁已经完成、谁还在等待。

## 验证与衔接

验证时重点看：

- 多个请求能按预期轮转
- 单个请求结束后不会再被调度
- 调度顺序和预期一致
- scheduler 不会改写请求内部内容
- 两个 executor 的 prefill/decode 次数、输出 token 和 Context 互不污染
- executor 返回错误时 Request 进入 `Failed`，不会把空 logits 采样成 token `0`

能够通过测试确认“请求推进顺序稳定、结束请求不会再被选中”，这页就算成立。

常见问题：

- 把 scheduler 和 cache 混在一起
- 请求结束状态没有清理干净
- 调度器持有过多业务逻辑
- 多请求时推进顺序不可预测
- 没有 admission 检查就把所有 waiting 请求放入 running
- 只加优先级字段，没有定义公平性和饥饿保护

和后续章节怎么接：

最小调度器把“谁先推进”说清楚了。
下一页再把 sampler 接上去，就能把“推进请求”变成“推进请求并产出 token”。
