# F04 Debugging and Testing Basics

## Goal

Introduce the debugging and testing habits used by the tutorial.

## What you should be able to do after reading

- Isolate a failing step
- Use small tests and incremental verification
- Distinguish shape checks, numerical checks, and performance checks
- Use logs as a debugging aid

## Why this matters

The tutorial builds a small inference engine step by step. That only works if every step can be checked independently. Without debugging habits, it becomes hard to tell whether a failure came from the current change or from an earlier layer.

## What comes next

- Unit tests for tensor ops and operators
- End-to-end correctness checks
- Benchmark measurements
- Shape and numerical checks for intermediate tensors

## Core ideas

### 1. Change one thing at a time

Make one change, check one result, then move on. Changing too much at once makes failures hard to localize.

### 2. Check intermediate results early

Do not wait until the whole model is assembled. Start checking as soon as tensor shapes, values, or simple forward outputs exist.

### 3. Use assertions to catch invalid states

Assertions are cheap and effective when they run close to the source of a bug.

```cpp
#include <cassert>
#include <cmath>
#include <vector>

bool close_enough(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

void check_matmul_shape(std::size_t a_rows, std::size_t a_cols,
                        std::size_t b_rows, std::size_t b_cols) {
    assert(a_cols == b_rows && "MatMul shape mismatch");
}

void check_index(std::size_t idx, const std::vector<float>& data) {
    assert(idx < data.size() && "Index out of bounds");
}
```

### 4. Prefer reproducible small tests

A small test with fixed inputs is easier to debug than a large scenario with many moving parts.

The tutorial’s testing flow can be understood in three layers:

- simple assertions for one-file checks
- standalone test programs for each chapter
- CMake integration with `add_executable(test_xxx test_xxx.cpp)`

### 5. Distinguish test types

- Shape checks: dimensions are correct
- Numerical checks: values are close enough
- Performance checks: latency or throughput meets expectations

## Minimal example

```cpp
#include <cassert>

int add_one(int x) {
    return x + 1;
}

int main() {
    assert(add_one(1) == 2);
    assert(add_one(0) == 1);
    return 0;
}
```

For floating-point values, use approximate equality instead of exact equality.

## How to validate

- Tests fail when the assumption is wrong
- Tests pass when the assumption is correct
- A bug can be reproduced with a tiny input
- You know whether the failure is shape, value, or performance related
- Once a failing case is found, keep it as a regression test

## Logging layers

1. Development: `std::cout` for quick iteration
2. Debugging: structured logs with context
3. Runtime: only key messages to avoid noise

## Summary

This page prepares the testing workflow used in the tutorial. If you can break a problem into small debug steps instead of guessing from the final failure, the page has done its job.
