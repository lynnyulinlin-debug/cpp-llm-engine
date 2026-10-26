# 00.3 Download TinyLlama 下载 TinyLlama 模型

**阅读入口**

- 前置阅读：[`00.2 Clone and Build`](./02_clone_and_build_llamacpp.md)
- 源码直达：<a href="../_code/chapter00_quickstart/main_quick.cpp"><code>main_quick.cpp</code></a>、<a href="../_code/chapter00_quickstart/run.sh"><code>run.sh</code></a>
- 相关阅读：[`00.4 First Inference`](./04_first_inference.md)、[`00.5 Troubleshooting`](./05_troubleshooting.md)

## 准备模型

准备一个小尺寸的 GGUF 模型，用来验证推理链路是否正常。

**完成后：**

- 知道为什么要用小模型做第一次验证
- 知道模型文件应该放在哪里
- 知道 GGUF 文件是后续推理验证的输入

**前置条件：**

- 已完成 `llama.cpp` 的克隆与编译
- 已有可用的 `models` 目录

## 为什么选择 TinyLlama Q4_K_M

第一次真实推理既要能在普通 CPU 环境运行，也要保留足够完整的模型结构供后续拆解。TinyLlama 1.1B 不是 toy model：它仍有 22 个 Decoder Block、真实 tokenizer、RMSNorm、RoPE、GQA、SwiGLU、KV Cache 和 LM Head。约 638 MiB 的 Q4_K_M GGUF 又能降低下载、内存和重复回归成本。

Q4_K_M 还会让后续章节遇到真实的 Q4_K/Q6_K 权重，而不是只处理 Float32 教学矩阵。`llama.cpp` 能直接加载同一文件，因此 Chapter00 的输出可以作为自研 Loader、Decoder 和 Runtime 的外部参考。这里选择它是为了工程可复现性，不表示 TinyLlama 的生成质量代表更大模型。

| 选择条件 | TinyLlama Q4_K_M 提供的价值 |
| --- | --- |
| CPU 可运行 | 无 GPU 也能完成第一期主线和真实回归 |
| 结构真实 | 22 层 Llama 路径覆盖模型加载、Attention、FFN、KV 和生成 |
| 量化真实 | Q4_K/Q6_K payload 可以进入 Chapter06 的正确性和性能实验 |
| 外部可对照 | 同一 GGUF 可由 `llama.cpp` 产生 tokenizer、logits 和生成基线 |
| 成本可控 | 文件和内存规模适合反复构建、调试与执行 golden |

## 下载步骤

先确认 `llama.cpp` 已经完成克隆。目录名中的 `.` 是项目目录名的一部分，不是文件扩展名，也不是隐藏目录标记；不要在尚未克隆时先创建 `llama.cpp` 空目录。

如果之前已经误执行过 `mkdir -p llama.cpp/models`，保留这个目录即可，先把它整体改名为占位目录，再克隆 llama.cpp 源码仓库：

```bash
cd /data/cpp-llm-engine
mv llama.cpp llama.cpp-before-clone
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
mkdir -p llama.cpp/models
```

检查 Git 仓库时必须从项目根目录执行：

```bash
cd /data/cpp-llm-engine
test -d llama.cpp/.git && echo "llama.cpp 已克隆" || echo "llama.cpp 不是 Git 仓库"
```

确认新仓库正常后，`llama.cpp-before-clone` 可以暂时保留，确认其中没有需要的文件后再由你手动清理。

```bash
cd /data/cpp-llm-engine
test -d llama.cpp/.git || git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
cd llama.cpp
mkdir -p models
wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## 验证和排错

- 文件下载完成
- `models/` 目录下存在 `.gguf` 文件
- 文件大小与预期接近

如果模型文件能被正确下载并放到统一目录，后续推理命令只需要关心“模型路径”和“参数”，而不需要再处理下载细节。

**常见问题：**

- 下载中断
- URL 失效
- 本地磁盘空间不足
- 把模型放错目录
