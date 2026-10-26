# 01.7 Input Contract and Tokenizer 输入契约与 Tokenizer

## 输入链路

前面的页面解决了“模型权重怎样进入程序”，本节补上另一条入口链：用户文本怎样变成后续 `Request` 能消费的 token IDs。

本节先固定 tokenizer 接口和教学用最小版本，再通过可选 llama.cpp reference adapter 验证真实 TinyLlama tokenization；本教程暂不重复实现完整 BPE/SentencePiece 算法。

### 两条输入边界

Chapter 01 需要同时区分两条链路：

```text
用户文本
  -> Tokenizer
  -> Token IDs
  -> Input Contract
  -> Request
```

```text
模型配置 / 权重文件
  -> ModelLoader
  -> Tensor
  -> Decoder / Runtime
```

Tokenizer 不负责加载权重，ModelLoader 也不负责理解用户 prompt。两条链路会在后续 Runtime 的 `Request` 处汇合。

### 最小接口

代码入口：`code/chapter01_tensor_loader/src/tokenizer.h`

```cpp
class Tokenizer {
public:
    TokenSequence encode(std::string_view text) const;
    std::string decode(const TokenSequence& tokens) const;
    std::size_t vocab_size() const;
};
```

默认实现仍是 toy tokenizer：字母映射到固定整数，未知字符映射到 `0`。它只用于验证接口、词表边界和 Request 输入，不产生有语言意义的模型输出。启用 `CHAPTER01_ENABLE_LLAMA_CPP` 后，`RealTokenizer` 使用真实 GGUF vocabulary 和 llama.cpp tokenizer API。

### 真实 tokenizer 还需要什么

接入真实模型时，Tokenizer 还必须处理：

- vocabulary 和 token IDs
- BPE merge rules 或 SentencePiece model
- normalization 和 Unicode 行为
- BOS、EOS、PAD、UNK 等 special tokens
- chat template 和 generation prompt
- encode / decode 的 golden test
- `vocab_size` 与 `ModelConfig` 的一致性

这些规则必须来自模型对应的 tokenizer 配置，不能用自定义字母映射替代。

真实 TinyLlama 对照命令：

```bash
bash code/chapter01_tensor_loader/run_real_tokenizer.sh
```

当前固定检查 `Hello TinyLlama` 的 6 个 token：`15043,18439,1460,29931,5288,29874`，并要求 detokenize 回原文。这个结果来自真实 GGUF vocabulary，不适用于其他模型；换模型时必须重新生成对照基线。

## 配置与 Request

代码入口：`code/chapter01_tensor_loader/src/model_config.h`

`ModelConfig` 先保存后续结构和输入需要的最小静态约束：`vocab_size`、`hidden_size`、`max_position`、`layer_count` 和 `max_new_tokens`。

至少要验证三条关系：

```text
ModelConfig.vocab_size == Tokenizer.vocab_size
Token ID < vocab_size
embedding.weight / lm_head.weight shape == [vocab_size, hidden_size]
```

真实模型接入时，配置通常来自模型 metadata 或 config 文件；本章当前实现只负责把一致性校验边界固定下来。

### 输入契约

Tokenizer 输出进入 Runtime 前，还需要固定：

| 字段 | 作用 |
| --- | --- |
| token IDs | 交给 embedding 和 decoder 的离散输入 |
| prompt length | 决定 prefill 工作量和 KV Cache 起点 |
| max input length | 防止超出模型上下文限制 |
| max new tokens | 决定 decode 上限和资源预算 |
| EOS / stop criteria | 决定请求何时结束 |

`chapter05_generation` 会把这些输入状态写入 `Request`、Scheduler 和 KV Cache。本节固定进入 Request 之前的语义。

## 验证结果

```bash
cmake -S code/chapter01_tensor_loader -B /tmp/cpp-llm-engine-chapter01-build
cmake --build /tmp/cpp-llm-engine-chapter01-build
ctest --test-dir /tmp/cpp-llm-engine-chapter01-build --output-on-failure
```

最小验证应确认：

- token 数量与输入字符数符合 toy 规则
- token ID 不超过词表范围
- 未知字符有明确行为
- decode 不越界
- Tokenizer 的 `vocab_size` 可以和模型配置做后续一致性检查
