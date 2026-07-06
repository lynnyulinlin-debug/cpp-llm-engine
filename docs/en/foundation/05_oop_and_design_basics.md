# F05 OOP and Design Basics

## Goal

Align the object-design principles that appear repeatedly in the tutorial, so ownership, boundaries, and state management stay clear later on.

## What you should be able to do after reading

- Understand a class as state plus behavior
- Understand encapsulation and why the public API should stay small
- Understand when composition is better than inheritance
- Understand why the tutorial prefers simple, explicit objects over fancy abstractions

## Why this matters

The tutorial uses C++ classes to represent `Tensor`, `ModelLoader`, operators, cache state, runtime state, and benchmark helpers. When object design is unclear, the reader usually gets stuck on ownership and responsibility, not on math.

## What comes next

- `Tensor` and `ModelLoader`
- operator and backend classes
- decoder layer composition
- cache and runtime state objects

## Responsibility map

| Class | Role | Notes |
| --- | --- | --- |
| `Tensor` | Data container | Holds storage and metadata |
| `ModelLoader` | File entry point | Parses and constructs data objects |
| `MatMul` | Compute node | Performs the actual operation |
| `KVCache` | State shell | Manages runtime cache |
| `Runtime` | State shell | Manages inference state |
| `DecoderLayer` | Composite object | Combines attention, norm, and FFN |

## Core ideas

### 1. Separate responsibilities first

A good class should carry one clear responsibility. In this tutorial, the common split is:

- data container
- file entry point
- compute node
- state shell

### 2. Expose only the next step

Small interfaces are easier to understand and test. The best interface is not the one with the most methods; it is the one that is just enough for the current chapter.

### 3. State, behavior, and ownership belong together

If an object owns memory, file handles, device buffers, or shared storage, its lifetime is part of the design.

### 4. Composition beats deep inheritance

This tutorial prefers composition for practical reasons:

- easier local replacement
- easier single-layer validation
- easier separation of decoder, operator, and backend

```cpp
class DecoderLayer {
public:
    Tensor forward(const Tensor& input) {
        auto x = attention_->forward(input);
        x = norm_->forward(x);
        return ff_->forward(x);
    }

private:
    std::unique_ptr<Attention> attention_;
    std::unique_ptr<LayerNorm> norm_;
    std::unique_ptr<FeedForward> ff_;
};
```

### 5. Inheritance is only for narrow, explicit boundaries

Use inheritance mainly for interface abstraction, not for code reuse.

Rules of thumb:

- ✅ pure virtual base classes
- ✅ strategy abstraction
- ❌ deep inheritance trees
- ❌ inheritance just to reuse code

### 6. Design should support verification

If an object looks elegant but cannot be checked independently, it is not good enough for this tutorial.

## Minimal example

```cpp
class Counter {
public:
    explicit Counter(int value) : value_(value) {}

    void add(int delta) { value_ += delta; }
    int value() const { return value_; }

private:
    int value_;
};
```

## How to validate

- Explain what the object owns
- Explain what the object does
- Explain why the interface is small enough
- Explain who owns, moves, or shares the resources
- Explain where composition is used and where inheritance is justified

## Summary

If you can explain why an object exists, what it owns, and what it intentionally does not own, the page has completed its job.
