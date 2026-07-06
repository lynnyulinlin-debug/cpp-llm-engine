# Glossary 术语表

这个页面固定教程里的核心术语，帮助中文读者统一写法，也帮助英文学习者对齐概念。

## 计算与张量

- `Tensor`：张量，带有形状、类型、设备和视图语义的数据容器
- `shape`：形状，张量每个维度的大小
- `dtype`：数据类型，例如 `float32`、`int8`
- `device`：设备，例如 CPU 或 GPU
- `view`：视图，共享底层存储但改变索引或形状
- `reshape`：重塑，只改变形状，不复制底层数据
- `slice`：切片，返回共享底层存储的局部视图

## 模型与算子

- `ModelLoader`：模型加载器，负责把权重读入程序
- `MatMul`：矩阵乘法算子，是线性计算的核心边界
- `Attention`：注意力机制，负责在上下文中选择相关信息
- `LayerNorm`：层归一化，稳定网络中的数值分布
- `FFN`：前馈网络，通常位于 Transformer block 内部
- `decoder`：解码器，生成式模型中逐步产出 token 的结构
- `KV Cache`：K/V 缓存，保存历史 token 的 key 和 value，避免重复计算

## 生成与调度

- `token`：模型输出的最小离散单位
- `prompt`：输入提示词
- `sampling`：采样，从候选 token 中选择下一步输出
- `sampler`：采样器，负责实现采样策略
- `runtime`：运行时，负责把模型执行组织成连续流程
- `scheduler`：调度器，负责决定下一个被推进的请求
- `prefill`：预填充，先把 prompt 处理进缓存
- `decode`：解码，基于已有缓存逐步生成后续 token
- `continuous batching`：连续批处理，批大小随请求状态动态变化
- `streaming`：流式输出，生成过程中持续返回结果

## 部署与评估

- `Quantization`：量化，用更低精度表示权重或激活，以减少内存和带宽开销
- `benchmark`：性能测试，用来比较延迟、吞吐和运行特征
- `throughput`：吞吐量，单位时间内完成的 token 或请求数量
- `latency`：延迟，单个请求或 token 完成所需时间
- `profiling`：性能剖析，用来解释性能结果来自哪里
- `A/B benchmark`：对比测试，在一致条件下比较两个实现或配置

## 参考框架

- `llama.cpp`：本地推理基线
- `vLLM`：高吞吐 serving 参考
- `SGLang`：结构化 serving 和应用编排参考
- `TensorRT-LLM`：GPU engine 和部署优化参考
- `Ollama`：本地打包与交付参考
- `Gemma`：模型家族参考

## 使用规则

- 优先使用这里列出的写法
- 第一次出现术语时，正文里给出中文解释和英文原词
- 不要在不同章节里为同一个概念随意换写法
- 如果后面要扩展术语，优先补充这里，再改正文
