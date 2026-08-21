# 04.7 vLLM / nano-vLLM 结构阅读桥

本页不复刻 vLLM 或 nano-vLLM 的源码，而是说明本章的 decoder block 在真实 serving runtime 里会落到哪里。读这些项目时，先不要急着追调度器，先确认 model runner 每一步调用的模型结构是什么。

这一页的作用是把 `chapter04` 放回整条主线里：`chapter01` 解决权重入口，`chapter02` 解决 view，`chapter03` 解决 MatMul backend，到了这一章，这些部件第一次被装进一个真正可 forward 的 decoder 结构。后面的 runtime 不是调度抽象公式，而是调度这里已经定义好的结构单元。

## 阅读入口

- 前置阅读：[`04.4 DecoderLayer`](./04_decoder_layer.md)、[`04.5 Load First Layer`](./05_load_first_layer.md)
- 源码直达：<a href="../_code/chapter04_attention/src/decoder_layer.cpp"><code>decoder_layer.cpp</code></a>、<a href="../_code/chapter04_attention/src/attention.cpp"><code>attention.cpp</code></a>
- 相关阅读：[`05.4 KV Cache`](../chapter05_generation/04_kv_cache.md)、[`05.8 vLLM 对照`](../chapter05_generation/07_vllm_comparison.md)、[`vLLM 生态对照`](../reference/ecosystem/vllm.md)

## 本章内容对应到哪里

| 本章概念 | 在 vLLM / nano-vLLM 阅读中对应什么 | 先看懂什么 |
| --- | --- | --- |
| `DecoderLayer` | model runner 每层 forward 的结构单元 | runtime 调度的是一批 token，但最终仍要落到 decoder layer |
| Q/K/V projection | attention backend 的输入准备 | K/V 的 shape 会继续影响 KV cache 布局 |
| MHA / GQA | K/V head 数量和 cache 形状 | 不是所有 head 都一定有独立 K/V cache |
| position / mask | prefill 与 decode 的可见性约束 | token 能看见哪些历史位置，不只由 scheduler 决定 |
| residual / FFN | attention 之后的逐 token 计算路径 | cache 只复用 attention 历史状态，不跳过后续计算 |

## 读源码时先抓四个边界

第一，看模型定义：确认一层 decoder 由哪些子模块组成，模块顺序是否和本章一致。

第二，看 attention forward：确认 Q/K/V 是在哪里生成的，position 或 rotary embedding 是在哪里接入的。

第三，看 runtime 传入的 metadata：真实 serving 项目通常不会只传一个 tensor，还会传位置、slot、block table 或 sequence metadata。

第四，看 KV cache 写入点：cache 不是 attention 的附属变量，而是 prefill / decode 之间共享历史状态的运行时边界。

## 为什么这页放在 chapter04

`chapter05` 会开始讲 generation runtime，但 runtime 调度的是“已经定义好的模型结构”。如果在进入 runtime 前没有看懂 decoder layer、Q/K/V 和 position，后面读 vLLM / nano-vLLM 时很容易把调度问题和模型结构问题混在一起。

所以本页的作用是先建立一个阅读坐标：`chapter04` 负责模型结构，`chapter05` 负责运行时状态，真实项目把二者接在 model runner、attention backend 和 KV cache 写入路径之间。
