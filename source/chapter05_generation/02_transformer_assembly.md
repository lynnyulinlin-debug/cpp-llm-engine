# 05.2 N-layer Transformer Assembly N 层 Transformer 组装


## 1. 这一页要解决什么

把前面做好的单层 Decoder 组合成一个 N 层 Transformer，并让 forward 主流程稳定跑通。  
这一页的重点不是新算子，而是把“单层正确”变成“多层可堆叠”。

## 阅读入口

- 前置阅读：[`05.1 Embedding 与位置编码`](/chapter05_generation/01_embedding_and_position)、[`chapter04_attention`](/chapter04_attention/intro)
- 源码直达：[`chapter05_generation 代码入口`](/code/chapter05_generation)
- 相关阅读：[`05.3 Naive Generation`](/chapter05_generation/03_naive_generation)、[`05.4 KV Cache`](/chapter05_generation/04_kv_cache)

## 2. 为什么这一步重要

单层正确只是基础，真正的生成模型需要多层堆叠。  
这里是从“模块”走向“模型”的关键一步。

如果这一页写得太像配置清单，读者会只知道“要有几层”，却不容易理解“为什么层与层之间能直接复用同一条 hidden state”。

## 3. 最小设计是什么

最小组装只需要两件事：

- 层与层之间传递同一种 hidden state
- 用固定层数把单层 forward 循环起来

只要每层输入输出 shape 一致，组装逻辑就主要变成循环和容器管理。  
这意味着组装层的职责不是重新定义模型结构，而是把已经正确的单层稳定地串起来。

### 每层独立权重

标准 Transformer 里，每一层 DecoderLayer 都有自己的独立参数：

```cpp
for (std::size_t i = 0; i < num_layers; ++i) {
    layers_.emplace_back(config);  // 每层独立初始化
}
```

这意味着：

- 第 0 层的 `Wq` 和第 1 层的 `Wq` 不是同一组参数
- 每层学习的是不同层次的表示
- 总参数量大致等于单层参数量乘以层数

这和 ALBERT 那种跨层共享权重不是一回事。本教程采用的是标准方案：每层独立权重。

### 残差路径如何跨层传播

单层 DecoderLayer 里的残差连接会把输入直接带到输出：

```text
hidden = x + attention(layernorm(x))
hidden = hidden + ffn(layernorm(hidden))
```

在 N 层堆叠里，这条路径会继续跨层传播：

```text
layer0 output -> layer1 input -> layer2 input -> ...
```

这意味着输入信号可以通过残差路径一直传到更深层，是深层 Transformer 能稳定工作的关键。

### Hidden state 在内存里怎么流转

在 C++ 实现里，hidden state 通常就是一个 `Tensor`：

```cpp
Tensor hidden = input;
for (auto& layer : layers_) {
    hidden = layer.forward(hidden);
}
```

从内存角度看：

- 每层输出会变成下一层输入
- 前一层的临时结果在下一层计算结束后可以释放或复用
- 不需要为每一层保留独立副本

这也是推理里很重要的工程点，因为它可以减少分配和拷贝。

## 4. 为什么要先固定配置

如果从工程实现倒推，先要把层配置固定下来：

- `num_layers`
- `hidden_size`
- `num_heads`
- `ffn_hidden_size`

这些参数最好来自统一配置对象。这样后面切换模型规模时，组装代码不会跟着改。  
组装层只负责读取配置和循环调用，不负责重新解释每一层的语义。

最小伪代码可以写成：

```cpp
Tensor hidden = input;
for (std::size_t i = 0; i < num_layers; ++i) {
    hidden = layers[i].forward(hidden);
}
```

这一段先不处理 cache 和采样，只表达多层堆叠。  
如果这里已经混进了 runtime 状态，后面的 generation 逻辑会更难拆开。

## 5. 怎么验证

- 整体 forward 可以跑通
- 每层输入输出 shape 正常
- 组装后没有破坏单层接口
- 不需要重新验证 04 章里已经确认过的单层内部计算

如果逐层打印 shape 后没有发现异常变化，这个组装过程就算可信。

### 和 04 章怎么分工

`chapter04` 已经验证了：

- 单层 DecoderLayer 的 forward 正确性
- 单层的 shape 一致性

`05.2` 只需要验证：

- 多层循环是否正确
- 每层的输入输出是否正确传递

这就是分层验证的原则：**本层只关注本层新增的逻辑，已经验证过的子模块不重复展开**。

## 6. 常见问题

- 层数配置错误
- 传递的 hidden state shape 不一致
- 组装逻辑破坏了单层验证结果

## 7. 和后续章节怎么接

这一页把 decoder 组织成完整 forward，下一步才适合进入 `03_naive_generation`，把 forward 变成真正的生成循环。
