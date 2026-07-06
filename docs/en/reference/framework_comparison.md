# Training vs Inference Frameworks

This page answers a practical question:

- What does a training framework do?
- What does an inference framework do?
- Which layer does this from-scratch C++ inference tutorial study?

## 1. One-sentence distinction

- **Training framework**: builds the model
- **Inference framework**: runs the model, keeps it fast, and keeps it stable
- **This tutorial**: breaks down the main inference ideas and builds a minimal implementation that can be studied, verified, and extended

## 2. Main differences

| Dimension | Training framework | Inference framework | This tutorial |
| --- | --- | --- | --- |
| Goal | Train a usable model | Serve inference with low latency and high throughput | Build a verifiable, explainable inference path from scratch |
| Core tasks | Forward, backward, gradients, optimizer, distributed training | Forward-only, scheduling, KV cache, quantization, streaming | Tensor, loader, operators, decoder, generation, benchmark |
| Main focus | Training stability, differentiability, throughput, memory efficiency | Throughput, latency, concurrency, caching, deployment | Principles, boundaries, implementation, verification, migration |
| Typical objects | PyTorch, JAX, TensorFlow | `llama.cpp`, `vLLM`, `SGLang`, `TensorRT-LLM` | This repository’s chapter line |
| Output | checkpoint, optimizer state, logs | runtime / engine / serving endpoint | A readable, runnable, editable teaching implementation |

## 3. Training framework traits

Training frameworks care about learning the model.

They usually provide:

- automatic differentiation and backpropagation
- optimizers and learning-rate schedules
- eager or graph-style training execution
- mixed precision training
- distributed training and communication
- checkpoint save and restore

Their strength is flexibility.  
They are not built to minimize single-token latency.

## 4. Inference framework traits

Inference frameworks care about running the trained model efficiently.

They usually emphasize:

- forward-only execution
- KV cache management
- batching and scheduling
- streaming output
- quantization and deployment
- balancing throughput, latency, and resource use

## 5. This tutorial is in the middle layer

This tutorial is neither a training framework nor a production inference system.  
It is a learning path that separates the main ideas and then builds a minimal runtime.

It keeps:

- `Tensor` object modeling
- weight loading
- view semantics and shared storage
- operator boundaries
- generation runtime
- quantization path
- benchmark and verification discipline

It does not focus on:

- full training graphs
- backward propagation
- large-scale distributed training
- production multi-tenant serving
- complex scheduling or fault tolerance

## 6. The chain of responsibility

1. **Training framework**
   - trains the model
   - produces weights and training artifacts
2. **This tutorial**
   - starts from weights
   - organizes tensor, operator, decoder, and runtime layers
   - explains the key ideas in inference
3. **Inference framework**
   - turns inference into a deployable, scalable, schedulable service

## 7. What this tutorial borrows

### 7.1 From training frameworks

- unified object representation
- tensor and shape thinking
- model structure layering
- verification-oriented engineering discipline

### 7.2 From inference frameworks

- forward-only execution
- KV cache and request state
- scheduler, batching, and streaming
- quantization and deployment constraints

### 7.3 This tutorial’s own trade-off

We compress those ideas into a more teachable path:

- understand `Tensor`
- understand loader and view
- understand `MatMul` and attention
- understand generation runtime
- understand quantization
- finish with benchmark and verification

## 8. Concrete examples

### Training framework examples

- `PyTorch`
  - the most common training reference
  - useful for tensor, autograd, module, optimizer, and training-loop concepts
- `JAX`
  - useful for functional style and transformation thinking
- `TensorFlow`
  - useful for graph execution and deployment layering
- `DeepSpeed`
  - useful for large-model training systems

### Inference framework examples

- `llama.cpp`
  - local inference baseline
- `gemma.cpp`
  - a more specific and shorter C++ example
- `vLLM`
  - high-throughput serving
- `SGLang`
  - structured serving and orchestration
- `TensorRT-LLM`
  - GPU engine and deployment

### Model-family example

- `Gemma`
  - not a training framework and not an inference framework
  - it is the object being inferred

## 9. How to use this page

- Read it first if you want to understand why training and inference look different
- Use it to locate which layer the tutorial is actually teaching
- Treat it as a map, not as a replacement for the main line

## 10. Related chapters

- `chapter01_tensor_loader`
- `chapter02_tensor_ops`
- `chapter03_matmul`
- `chapter04_attention`
- `chapter05_generation`
- `chapter06_quantization`
- `chapter07_benchmark`
