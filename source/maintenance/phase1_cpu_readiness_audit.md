# CPU v1 建设与验收状态

本文仅供教程维护人员使用，不进入学习者导航。学习者版本位于 `source/guides/phase1_cpu_learning_plan.md`，只保留运行条件、学习步骤和验收标准。

CPU v1 的范围是单机、单线程模型执行、TinyLlama 1.1B Q4_K_M、Greedy Sampling 的教学型推理服务闭环。统一验收入口为 `scripts/run_cpu_v1_acceptance.sh`；它需要本地 llama.cpp、TinyLlama GGUF 和可执行的 `llama-simple`。

## 已完成范围

| 范围 | 建设状态 | 维护关注点 |
| --- | --- | --- |
| 真实 CPU 基线 | 已完成 | llama.cpp CPU 推理固定真实模型和对照输出 |
| 输入与加载 | 已完成 | 真实 tokenizer、GGUF schema、Tensor index、ModelConfig 和 22 层映射可验证 |
| Tensor 与 MatMul | 已完成 | Tensor view、公共 contract 和 CPU reference backend 已进入回归 |
| 真实 Decoder | 已完成 | 统一脚本串联 Chapter04、Chapter06 和真实 22 层 generation golden |
| Generation Runtime | 已完成 | Request、prefill/decode、KV、采样、chunk 和多请求状态闭环 |
| 最小 CPU 服务 | 已完成 | ServiceCore、真实 HTTP/SSE、metrics、双请求 benchmark 和 profiling 闭环 |
| CPU 全局回归 | 已完成 | 无 GGUF 默认回归与有 GGUF 的 CPU v1 验收分开执行 |

## v1 非目标

这些能力属于后续增强，不阻止 CPU v1 收口。

| 非目标 | 当前边界 |
| --- | --- | --- |
| 并行模型执行 | cpp-httplib 可接收连接，但进入当前 ServiceCore 后串行执行 |
| 完整采样策略 | 可执行路径固定 Greedy；Temperature、Top-K、Top-p 保留为扩展 |
| 真实 Paged KV payload | Block allocator 管理 ownership；真实 K/V 仍由请求 Context 连续持有 |
| 多模型通用性 | 首个完整模型固定为 TinyLlama/Llama 架构，不宣称覆盖所有 GGUF 架构 |
| 生产运维 | Python Gateway、分布式调度、生产限流和故障隔离不在 CPU v1 内 |

## 验收命令

```bash
CPP_LLM_ENGINE_MODEL_PATH=/path/to/tinyllama.gguf \
LLAMA_CPP_ROOT=/path/to/llama.cpp \
bash scripts/run_cpu_v1_acceptance.sh
```

脚本通过后会在 `${CPU_V1_RESULT_DIR:-/tmp/cpp-llm-engine-cpu-v1}` 保存验收清单和真实服务 benchmark。默认 `npm run test:regression` 继续保持无模型、无网络和无 GPU 依赖。

最近一次统一验收使用 GCC 13.1.0 和 638 MiB TinyLlama GGUF，六项任务全部通过；真实双请求 benchmark 的 correctness、Context/cache 隔离和 terminal 资源释放均为 `true`。本地 HTTP/SSE 检查需要允许监听 `127.0.0.1`，受控沙箱禁止 socket 时应在宿主环境执行。
