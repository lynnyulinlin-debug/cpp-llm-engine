# F01 C++ Basics

## Goal

Cover the C++ basics used throughout the tutorial, with emphasis on ownership, object lifetime, and the kind of code organization a small inference engine needs.

## What you should be able to do after reading

- Understand classes, constructors, destructors, and RAII
- Understand references, pointers, and basic ownership
- Understand `std::vector`, `std::string`, and smart pointers
- Understand headers, source files, and namespaces
- Reason about copy and move semantics in real code

## Why this matters

The main tutorial uses modern C++. That means the code cares about more than syntax: it cares about clear ownership, predictable object lifetime, and small abstractions that are easy to debug and extend.

If these basics are shaky, `Tensor`, `ModelLoader`, operator wrappers, and KV cache management become much harder to follow.

## What comes next

- `Tensor` design
- memory management
- operator and loader implementations
- shared weights and cache buffers
- testing and benchmark helper code

## Core ideas

### 1. A class should bundle state and behavior

When data and behavior belong together, use a class. Keep the public API small and make ownership obvious.

### 2. Use constructors and destructors to define valid lifetime

Constructors create valid state. Destructors release resources. That is the foundation of RAII.

### 3. Copy and move are part of the performance story

Copying duplicates data or ownership. Moving transfers ownership. For large buffers and tensors, move semantics often decide whether a design is acceptable.

### 4. Smart pointers are tools, not defaults

| Ownership | Tool | Typical use |
| --- | --- | --- |
| Single ownership | Direct member / `unique_ptr` | Tensor storage, exclusive resources |
| Shared ownership | `shared_ptr` | Explicitly shared storage later in the tutorial |
| Non-owning observation | `weak_ptr` / raw pointer | Temporary access, observation only |

In this tutorial, `Tensor` defaults to single ownership. Shared ownership is introduced only when the tutorial needs shared storage semantics.

### 5. STL containers are part of design

- `vector` for contiguous buffers and random access
- `string` for textual metadata and names
- `unordered_map` for lookup-heavy metadata tables

### 6. Headers and source files separate interface from implementation

Put declarations in headers and implementations in source files. Use namespaces to avoid symbol collisions as the project grows.

## Minimal example

```cpp
#include <vector>
#include <string>

class Tensor {
public:
    Tensor(std::string name, std::size_t size)
        : name_(std::move(name)), data_(size) {}

    const std::string& name() const { return name_; }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

private:
    std::string name_;
    std::vector<float> data_;
};

Tensor make_tensor() {
    Tensor t("weight", 1024);
    t.data()[0] = 1.0f;
    return t;
}
```

The important points are:

- ownership is explicit
- tensor payload is managed by RAII
- returning a large object does not require manual `new` / `delete`

## How to validate

- Explain who owns the tensor storage
- Explain why returning the object does not leak memory
- Explain when move is better than copy
- Explain why this page does not need `shared_ptr`

## Summary

This page gives the C++ base layer for the tutorial. If you can explain why a `Tensor` should explicitly own its storage and how copy versus move changes behavior, the page has done its job.
