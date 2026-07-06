# F06 Data Structures and Algorithms Basics

## Goal

Align the data-layout, container selection, and algorithm intuition that appear repeatedly in the tutorial.

## What you should be able to do after reading

- Understand why contiguous storage matters for tensors
- Understand when `vector`, `unordered_map`, or a simple array is enough
- Understand complexity as an engineering intuition, not a theory class
- Understand why the tutorial starts with the simplest algorithm

## Why this matters

The tutorial repeatedly uses a few data-structure choices: tensors use contiguous buffers, name lookup uses hash maps, state records use vectors, and correctness-first paths start with simple loops.

## What comes next

- tensor storage and view semantics
- loader lookup tables
- attention and matmul loops
- KV cache and runtime state containers
- benchmark result storage

## Core ideas

### 1. Prefer contiguous storage for numeric data

When you need dense numeric data, predictable iteration, and shared raw buffers, contiguous storage should be the default.

### 2. Prefer lookup structures for named access

When you need to find a tensor or symbol by name, use a hash map.

### 3. Prefer ordered containers for ordered state

Use a vector when order matters, such as layer stacks, token sequences, request progress, or benchmark records.

### 4. Start with the simplest baseline algorithm

For this tutorial, a clear `O(n^3)` CPU reference is better than a clever but unreadable optimization.

### 5. Data layout changes future cost

The same algorithm can have very different downstream cost depending on layout.

## Lookup structure choices

| Need | Structure | Why |
| --- | --- | --- |
| Name lookup, order not important | `unordered_map` | Fast lookup, good for loader paths |
| Name lookup with deterministic order | `std::map` | Ordered and reproducible |
| Very small lookup table | `vector` + linear search | Simple and low overhead |

## Selection table

| Need | Preferred structure | Why |
| --- | --- | --- |
| Contiguous numeric buffer | `vector` / contiguous array | Great for sequential access and transfer |
| Name to object mapping | `unordered_map` | Good for loading and lookup |
| Ordered state or logs | `vector` | Clear sequence semantics |
| Small fixed-size data | `array` / direct fields | Low overhead and explicit bounds |
| Verification pass | Plain loop | Easiest to read and test |

## Composition view

`F05` is about how objects are composed; `F06` is about how data structures are chosen inside those objects.

```cpp
class Model {
public:
    std::unordered_map<std::string, Tensor> tensors_;  // name lookup
    std::vector<std::string> layer_order_;             // execution order
    std::vector<float> shared_buffer_;                 // contiguous storage
};
```

## Minimal example

```cpp
#include <unordered_map>
#include <vector>
#include <string>

std::unordered_map<std::string, int> make_lookup() {
    return {{"weight", 1}, {"bias", 2}};
}

int sum_vector(const std::vector<int>& values) {
    int sum = 0;
    for (int v : values) {
        sum += v;
    }
    return sum;
}
```

## How to validate

- Explain why tensor uses contiguous storage
- Explain why model parameters use a lookup table
- Explain why CPU reference starts with a simple loop
- Explain which data should be ordered and which should be name-addressed
- Explain complexity using intuition instead of proof

## Summary

If you can explain access patterns first and only then choose a structure, this page has done its job.
