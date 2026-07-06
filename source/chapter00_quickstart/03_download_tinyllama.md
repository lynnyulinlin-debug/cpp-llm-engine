# 00.3 Download TinyLlama 下载 TinyLlama 模型

## 阅读入口

- 前置阅读：[`00.2 Clone and Build`](/chapter00_quickstart/02_clone_and_build_llamacpp)
- 源码直达：[`main_quick.cpp`](/code/chapter00_quickstart)、[`run.sh`](/code/chapter00_quickstart)
- 相关阅读：[`00.4 First Inference`](/chapter00_quickstart/04_first_inference)、[`00.5 Troubleshooting`](/chapter00_quickstart/05_troubleshooting)

## 这一页要解决什么

准备一个小尺寸的 GGUF 模型，用来验证推理链路是否正常。

## 读完你应该能做什么

- 知道为什么要用小模型做第一次验证
- 知道模型文件应该放在哪里
- 知道 GGUF 文件是后续推理验证的输入

## 前置条件

- 已完成 `llama.cpp` 的克隆与编译
- 已有可用的 `models` 目录

## 推荐模型

建议先使用 TinyLlama 的 GGUF 量化版本，因为它体积较小，适合做快速验证。

这里选择小模型的原因很现实：第一次验证只需要确认链路通了，不需要追求模型能力。小模型下载更快、占用更少、出错更容易定位，是做 smoke test 的理想对象。

## 步骤

```bash
cd /data/cpp-llm-engine/llama.cpp
mkdir -p models
wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## 怎么验证

- 文件下载完成
- `models/` 目录下存在 `.gguf` 文件
- 文件大小与预期接近

如果模型文件能被正确下载并放到统一目录，后续推理命令只需要关心“模型路径”和“参数”，而不需要再处理下载细节。

## 常见问题

- 下载中断
- URL 失效
- 本地磁盘空间不足
- 把模型放错目录
