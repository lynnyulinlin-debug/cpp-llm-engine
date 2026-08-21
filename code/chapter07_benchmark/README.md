# Chapter 07: Benchmark

This directory contains the minimal implementation for `chapter07_benchmark`.

## Scope

- Lightweight test harness setup
- End-to-end correctness checks
- Benchmark metrics
- Benchmark scripts
- Optimization notes

## How to run

- `bash scripts/run.sh`

## Current status

- Library, demo, and test targets are available
- Named unit checks, e2e checks, and benchmark summary are wired together
- This chapter is the verification and performance closure for the tutorial
- The demo now emits stage-level timing for `input`, `prefill`, `decode`, `sampling`, and `output`
- The demo also emits a minimal event-level trace so chapter08 can teach evidence before full profilers
- The demo also emits a minimal hotspot summary so chapter08 can connect traces to function-level ranking
- The demo also emits a minimal JSON trace artifact so chapter08 can teach structured evidence, not just console text
- The demo also emits a minimal Chrome-trace-style view so chapter08 can teach timeline-shaped evidence
- The Chrome-trace-style view now includes minimal host/compute track separation and one sync-point marker
- The demo now also includes a tiny multi-request timeline to explain overlap and batch-merge waits
- The current code is intentionally small: it provides a baseline for correctness and performance comparison, not a full profiling or serving system
- The intended output is decision evidence, not just raw numbers: compare baseline, optimized, and reference behavior under the same conditions
- Benchmark comparisons should keep inputs, hardware, and runtime flags aligned so the result reflects the change being measured
- Results should be treated as a reusable record: input shape, device, flags, and script version matter when you revisit an A/B comparison later
- If this chapter's tests grow more complex later, the migration plan to Google Test is documented in `docs/appendix/gtest_migration.md`
