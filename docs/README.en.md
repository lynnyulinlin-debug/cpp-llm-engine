# C++ LLM Inference Engine Tutorial

_A Practical Tutorial for Building a C++ LLM Inference Engine_

This is the canonical English entry for the tutorial. The Chinese main line
remains the source of truth, and the English side starts with entry pages
first.

## Roles

- foundation: background you should know before the main line
- chapter00_quickstart: the first runnable baseline
- guide.md: how to read the tutorial
- guides: engineering references for build, environment, and migration
- reference: terminology, concept mapping, and ecosystem comparison
- maintenance: contributor-only maintenance rules and templates

## Coverage status

- English navigation and entry pages: available
- Chapter intros and summaries: rolling out in Phase 1
- Full chapter body translation: planned for later phases

## Start here

1. Read [guide.md](/en/guide) first if you want a route map.
2. Read [foundation/README.md](/en/foundation/README) if you need background.
3. Open [chapter00_quickstart](/en/chapter00_quickstart/intro) when you want the first runnable baseline.
4. Use [guides/README.md](/en/guides/README) for build or environment help.
5. Use [reference/README.md](/en/reference/README) when you need concept mapping or ecosystem context.

## Main line

- foundation
- chapter00_quickstart
- chapter01_tensor_loader
- chapter02_tensor_ops
- chapter03_matmul
- chapter04_attention
- chapter05_generation
- chapter06_quantization
- chapter07_benchmark

## Reference Layers

- `llama.cpp` for the local inference baseline
- `vLLM` for high-throughput serving design
- `SGLang` for structured serving and orchestration
- `TensorRT-LLM` for GPU engine and deployment design
- `Ollama` for packaging and local delivery
- `Gemma` for model-family examples

## Notes

- If you only need the route map, start from [guide.md](/en/guide).
- If you need the full Chinese walkthrough, switch back to the Chinese main entry.
## Document Layers

- Must-read for learners: `guide.md`, `chapter00_quickstart`, `foundation`
- Must-read for engineering help: `guides/README.md`
- Must-read for future contributors: `maintenance/tutorial_maintenance.md`, `maintenance/tutorial_writing_guidelines.md`
- Optional reference: `reference/glossary.md`, `reference/framework_comparison.md`, `reference/ecosystem/`
