# C++ LLM Inference Engine Tutorial

_A Practical Tutorial for Building a C++ LLM Inference Engine_

This is the legacy English navigation entry for readers who prefer an
English-first guide. The canonical English entry is now [English tutorial
entry](/en/README). The main tutorial content is still Chinese-first in
[README.md](README.md).

## Entry Roles

- `foundation`: prepare the background you need before the main line
- `chapter00_quickstart`: get the first runnable baseline as quickly as possible
- `guides`: practical engineering references for build, environment, and migration
- `reference`: design references, terminology, and ecosystem comparisons
- `maintenance`: contributor-only maintenance rules and templates
- `guide.md`: choose the reading path and understand how the tutorial is organized

## Current English Coverage

- English navigation and high-level orientation: available
- English entry pages and chapter landing pages: available
- Foundation pages: available
- Key reference pages: available
- Full chapter-by-chapter tutorial content in English: not yet complete
- If you need the full learning path today, use the Chinese main line together
  with this English entry as a guide

If you are unsure where to start, read `guide.md` first, then decide whether
you need `foundation` before moving on to `chapter00_quickstart`.

## Start Here

1. Read [README.md](README.md) for the Chinese-first project entry
2. Read [guide.md](guide.md) for reading tracks and chapter roles
3. Read [guides/README.md](guides/README.md) if you need build, environment, or migration help
4. Read [maintenance/tutorial_maintenance.md](maintenance/tutorial_maintenance.md) for maintenance rules
5. Read [reference/glossary.md](reference/glossary.md) for core terminology

## Main Line

- `foundation`
- `chapter00_quickstart`
- `chapter01_tensor_loader`
- `chapter02_tensor_ops`
- `chapter03_matmul`
- `chapter04_attention`
- `chapter05_generation`
- `chapter06_quantization`
- `chapter07_benchmark`

## Reference Layers

- `llama.cpp` for the local inference baseline
- `vLLM` for high-throughput serving design
- `SGLang` for structured serving and orchestration
- `TensorRT-LLM` for GPU engine and deployment design
- `Ollama` for packaging and local delivery
- `Gemma` for model-family examples

## Reading Advice

- Start with `chapter00_quickstart` if you want a runnable baseline first
- Use `foundation` only if you need background on C++, CMake, or CUDA
- Follow the chapter summaries as checkpoints
- Use the reference layers for comparison, not as an alternate main line
- If you do not read Chinese comfortably, this page is not enough by itself to
  learn the whole tutorial yet

## Document Layers

- Must-read for learners: `guide.md`, `chapter00_quickstart`, `foundation`
- Must-read for engineering help: `guides/README.md`
- Must-read for future contributors: `maintenance/tutorial_maintenance.md`, `maintenance/tutorial_writing_guidelines.md`
- Optional reference: `reference/glossary.md`, `reference/framework_comparison.md`, `reference/ecosystem/`
