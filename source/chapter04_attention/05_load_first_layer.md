# 04.5 First Layer Weight Wiring 首层权重加载与验证


## 这一页要解决什么

从 `chapter01` 的模型加载结果中提取第一层权重，并验证前向传播的张量形状是否正确。

读完后应能做到：

- 理解如何把第 1 章的加载结果接入当前模块
- 理解如何验证第一层权重是否真正可用
- 理解为什么先只验证一层，而不是直接上完整模型

## 阅读入口

- 前置阅读：[`04.4 DecoderLayer`](./04_decoder_layer.md)、[`chapter01_tensor_loader`](../chapter01_tensor_loader/intro.md)
- 源码直达：<a href="../_code/chapter04_attention/src/decoder_layer.cpp"><code>decoder_layer.cpp</code></a>、<a href="../_code/chapter01_tensor_loader/src/model_loader.cpp"><code>model_loader.cpp</code></a>、<a href="../_code/chapter04_attention/tests/chapter04_attention_test.cpp"><code>chapter04_attention_test.cpp</code></a>
- 相关阅读：[`04.7 vLLM / nano-vLLM 结构阅读桥`](./07_vllm_nano_vllm_structure_bridge.md)、[`04.8 本章小结`](./summary.md)、[`chapter05_generation`](../chapter05_generation/intro.md)

## 验证设计

这一页把“能加载模型”进一步推进到“能用加载出来的权重跑前向”。它是从静态数据到动态推理的关键过渡。

验证重点：

- 权重名称可以对应上
- shape 和模块定义一致
- 前向传播输出 shape 正常

这一页的重点不是把整个模型一次性接完，而是先把“加载出来的权重能真实驱动一层前向”这个问题解决掉。只要第一层对了，后面的层只是在重复同一个结构。

数值验证的范围：

这一阶段的“数值验证”目标是有限的：

- 确认输出不是 `NaN` 或 `Inf`
- 确认输出范围合理，没有明显发散
- 不要求和某个 reference 完全一致

这一页先解决的是“模块能跑通”，不是“数值证明完全正确”。更完整的 reference 对比会放到后续的验证流程里。

实现建议：

- 只加载第一层做验证
- 先验证 shape，再验证数值
- 出问题时优先检查加载映射

如果权重名和模块名之间有一层显式映射表，调试通常会比直接硬编码索引更轻松。

实际调试时，建议先把这一层拆成两个检查点：先确认权重是否都能按名字找到，再确认每个张量的 shape 是否和模块定义一致。这样可以把“加载失败”和“模块定义不匹配”区分开。

一开始就尝试整层前向时，出现错误往往很难判断问题来自 loader、张量布局还是 attention 计算本身。

权重缺失时怎么办：

当 `loader.find()` 返回空指针时，教程主线建议先停下来检查映射，而不是直接跳过：

1. 检查名称映射是否正确
2. 检查模型文件是否完整
3. 检查当前层的命名约定是否和模型一致

如果真的需要临时继续排查，可以：

- 用随机初始化只做 shape 验证
- 或者把这一层标记为未验证

但主线教程里更推荐先修复加载映射，因为随机权重很容易让 shape 看起来没问题、数值却完全偏掉，调试成本反而更高。

不同模型的命名约定：

权重名称是模型特定的，不能假设所有模型都用同一套命名：

| 模型 | 命名示例 |
| --- | --- |
| BERT | `bert.encoder.layer.0.attention.self.query` |
| GPT-2 | `transformer.h.0.attn.c_attn` |
| LLaMA | `model.layers.0.self_attn.q_proj` |
| 本教程 | `layer0.attention.wq` |

本教程使用的是简化命名，目的是让“名称映射”和“模块接线”更便于理解。真实项目里通常需要根据具体模型文件调整解析逻辑。

与 Chapter01 的衔接：

`chapter01` 的 `ModelLoader` 负责把权重加载进一个按名称索引的容器里：

```cpp
class ModelLoader {
    std::unordered_map<std::string, Tensor> weights_;
public:
    const Tensor* find(const std::string& name) const;
};
```

`04.5` 就是在这个接口上，去拿第一层权重并接入当前模块：

```cpp
auto loader = ModelLoader("model.bin");
auto layer = build_decoder_layer();
auto wq = loader.find("layer0.attention.wq");
```

关键点是分工清楚：

- `loader` 负责“从文件加载并建立名称索引”
- `layer` 负责“使用权重执行计算”

两者通过名称映射解耦，避免把文件格式和模块结构硬绑在一起。

最小验证代码可以先写成：

```cpp
auto layer = build_decoder_layer();
auto weights = loader.find("layer0.attention.wq");
assert(weights != nullptr);
assert(weights->shape() == std::vector<int64_t>{hidden, hidden});
```

先检查权重能否找到，再检查 shape，再做前向。

## 验证与排查

常见问题：

- 权重名称映射不一致
- 第一层 shape 和模块输入不匹配
- 直接上整模型导致排错成本过高
