# Contributing to cpp-llm-engine

Thank you for helping improve this tutorial.

## What This Project Is

`cpp-llm-engine` is a practical tutorial for building a C++ LLM inference engine.
The main line runs from `Tensor` and `MatMul` to `Generation`, `Quantization`,
and `Benchmark`, with a clear path toward vLLM-like high-throughput runtime ideas.

## What Contributions Are Welcome

- Fixes to documentation, wording, and chapter bridges
- Improvements to examples, scripts, and validation steps
- Small code corrections that keep the tutorial runnable
- Better cross-references between chapters and `guides` / `reference` / `maintenance` pages

## What to Keep Stable

- Keep the main chapter order intact unless there is a strong reason to change it
- Keep terminology consistent with the writing guidelines
- Keep benchmark and correctness checks separated
- Keep external framework references as comparison points, not as the main line

## Before You Submit

- Read [docs/maintenance/tutorial_writing_guidelines.md](/data/cpp-llm-engine/docs/maintenance/tutorial_writing_guidelines.md)
- Read [docs/maintenance/tutorial_maintenance.md](/data/cpp-llm-engine/docs/maintenance/tutorial_maintenance.md)
- Check whether your change affects chapter boundaries or bridge text
- Verify the relevant build or test command still works

## Suggested Workflow

1. Make the smallest change that solves the problem.
2. Update the related doc or code path.
3. Run the relevant chapter script or test.
4. Keep the wording short and consistent.

## Testing Notes

- `chapter00_quickstart` uses smoke checks
- `chapter01` to `chapter06` use chapter-local tests or assert-based checks
- `chapter07` uses unit tests and end-to-end checks
- Benchmark results should not be used as correctness evidence

When you change a chapter, keep the relevant verification layer aligned with the
kind of change you made: correctness changes need tests, while performance
changes need comparable benchmark conditions.

## If You Are Unsure

- Prefer a small patch over a broad rewrite
- Prefer a bridge sentence over a large structural change
- Prefer updating existing sections over adding new sections
