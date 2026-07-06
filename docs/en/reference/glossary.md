# Glossary

This page fixes the core terms used in the tutorial. It helps readers keep the wording consistent and gives English readers a stable concept map.

## Tensor and data

- `Tensor`: a data container with shape, type, device, and view semantics
- `shape`: the size of each tensor dimension
- `dtype`: data type, such as `float32` or `int8`
- `device`: execution device, such as CPU or GPU
- `view`: a shared-storage view with a different index or shape
- `reshape`: change shape without copying the underlying data
- `slice`: return a local view backed by the same storage

## Model and operators

- `ModelLoader`: loads model weights into the program
- `MatMul`: matrix multiplication operator
- `Attention`: mechanism that selects relevant context information
- `LayerNorm`: layer normalization
- `FFN`: feed-forward network inside a Transformer block
- `decoder`: the structure that generates tokens step by step
- `KV Cache`: key-value cache for historical tokens

## Generation and scheduling

- `token`: the smallest discrete output unit
- `prompt`: the input prompt
- `sampling`: choose the next token from candidates
- `sampler`: the component that implements sampling policy
- `runtime`: the system that organizes model execution as a continuous flow
- `scheduler`: decides which request advances next
- `prefill`: process the prompt and fill the cache
- `decode`: generate the next token step by step
- `continuous batching`: dynamic batching as requests change state
- `streaming`: emit results while generation is still running

## Deployment and evaluation

- `Quantization`: represent weights or activations in lower precision
- `benchmark`: performance testing for latency, throughput, and behavior
- `throughput`: how many tokens or requests finish per unit time
- `latency`: how long one request or token takes
- `profiling`: inspect where performance time comes from
- `A/B benchmark`: compare two implementations under the same conditions

## Reference frameworks

- `llama.cpp`: local inference baseline
- `vLLM`: high-throughput serving reference
- `SGLang`: structured serving and orchestration reference
- `TensorRT-LLM`: GPU engine and deployment optimization reference
- `Ollama`: local packaging and delivery reference
- `Gemma`: model-family reference

## Usage rules

- Prefer the names listed here
- Explain the Chinese term and the English original the first time it appears
- Do not rename the same concept differently across chapters
- If you need to extend the glossary later, update this page first
