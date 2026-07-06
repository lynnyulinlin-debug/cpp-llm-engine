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
- The current code is intentionally small: it provides a baseline for correctness and performance comparison, not a full profiling or serving system
- The intended output is decision evidence, not just raw numbers: compare baseline, optimized, and reference behavior under the same conditions
- Benchmark comparisons should keep inputs, hardware, and runtime flags aligned so the result reflects the change being measured
- Results should be treated as a reusable record: input shape, device, flags, and script version matter when you revisit an A/B comparison later
- If this chapter's tests grow more complex later, the migration plan to Google Test is documented in `docs/appendix/gtest_migration.md`
