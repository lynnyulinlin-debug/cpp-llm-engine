# 05.5 Scheduler 调度器


## 这一页要解决什么

把“谁先推进、谁后推进、什么时候推进”从 runtime 的隐式行为，变成显式的调度规则。  
这一页先定义最小调度器，不展开复杂批处理。

## 阅读入口

- 前置阅读：[`05.4 KV Cache`](/chapter05_generation/04_kv_cache)、[`05.3 Naive Generation`](/chapter05_generation/03_naive_generation)
- 源码直达：[`chapter05_generation 代码入口`](/code/chapter05_generation)
- 相关阅读：[`05.6 Sampler`](/chapter05_generation/05_sampler)、[`05.7 端到端生成测试`](/chapter05_generation/06_end_to_end_generation)

## 代码直达

- [`src/scheduler.cpp`](/code/chapter05_generation#scheduler-cpp)
- [`src/runtime.cpp`](/code/chapter05_generation#runtime-cpp)
- [`src/request.cpp`](/code/chapter05_generation#request-cpp)
- [`tests/chapter05_scheduler_test.cpp`](/code/chapter05_generation#chapter05-scheduler-test-cpp)

## 2. 为什么调度必须显式化

当 runtime 只有单个请求时，推进顺序看起来像是“自然发生”的。  
一旦进入多请求场景，系统就必须明确回答：当前该推进谁？

如果调度规则不显式，cache 和 sampler 都会被迫在隐式上下文里工作，性能、延迟和公平性也就很难判断。

## 3. 最小设计是什么

最小调度器先只做一件事：在活跃请求里选出下一条要推进的请求。

可以先把它理解成一个简单的队列调度器：

- 请求进入队列
- 调度器选择一个请求推进
- 推进结束后决定是否重新入队

最小伪代码可以写成：

```cpp
Request* pick_next(std::deque<Request*>& active) {
    if (active.empty()) return nullptr;
    Request* req = active.front();
    active.pop_front();
    return req;
}
```

这一页不追求复杂策略，只先把“推进顺序可观察、可测试”立住。

### 调度器和请求状态的边界

调度器不负责计算 logits，也不负责维护 K/V。  
它只负责决定哪个请求进入下一轮 runtime 推进。

这样可以把职责拆开：

- `Request` 保存状态
- `KV Cache` 保存历史
- `Scheduler` 决定推进顺序
- `Sampler` 决定下一步 token

如果这四件事混在一起，后面很难做单独验证。

## 4. 最小实现怎么落

最小实现可以先用 round-robin：

1. 从队列头取一个请求
2. 推进一次
3. 如果请求没结束，放回队尾
4. 如果请求结束，释放它的 runtime 资源

这个策略的优点是容易理解、容易验证，也容易和单请求路径对照。  
它不是最终优化方案，但足够作为第一个可观察调度模型。

### 和 runtime 的接口

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

## 5. 怎么验证

- 多个请求能按预期轮转
- 单个请求结束后不会再被调度
- 调度顺序和预期一致
- scheduler 不会改写请求内部内容

能够通过测试确认“请求推进顺序稳定、结束请求不会再被选中”，这页就算成立。

## 6. 常见问题

- 把 scheduler 和 cache 混在一起
- 请求结束状态没有清理干净
- 调度器持有过多业务逻辑
- 多请求时推进顺序不可预测

## 7. 和后续章节怎么接

这一页把“谁先推进”说清楚了。  
下一页再把 sampler 接上去，就能把“推进请求”变成“推进请求并产出 token”。
