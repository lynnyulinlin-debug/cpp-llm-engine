# F08 PyTorch to C++ Bridge

## Goal

Help readers who already know PyTorch translate their mental model into this tutorial’s C++ objects, loading path, and runtime boundaries.

## What you should be able to do after reading

- Map common PyTorch concepts to the tutorial’s C++ concepts
- Read the tutorial using PyTorch thinking as a bridge
- Understand what is “the same idea, different language”
- Understand why this tutorial treats PyTorch as a reference, not as the runtime itself

## Why this matters

Many readers already know tensors, modules, state dicts, and forward passes. The hard part is switching to C++ ownership, file layout, backend selection, and explicit runtime structure.

## What comes next

- `Tensor` and `ModelLoader`
- `MatMul`, `Attention`, and `Generation`
- `Quantization` and benchmark validation

## Responsibility map

| `F05` category | PyTorch concept | Tutorial class |
| --- | --- | --- |
| Data container | `torch.Tensor` | `Tensor` |
| File entry point | `state_dict` | `ModelLoader` |
| Compute node | `nn.Linear` | `MatMulOp` |
| State shell | runtime state | `KVCache` / `Runtime` |
| Composite structure | `nn.Module` | `DecoderLayer` |

## Core ideas

### 1. Tensor is a semantic data object

In PyTorch, a tensor is a first-class object with shape, dtype, device, and view behavior. In this tutorial, those ideas become explicit C++ fields and ownership rules.

### 2. `nn.Module` maps to operator / layer classes

The forward-path idea is the same, but the C++ version makes ownership, interfaces, and backend choice explicit.

### 3. `state_dict` maps to naming-based loading

The loader does not only answer “what did we read?” It also answers “what should these weights become in the program?”

### 4. `view` maps to a lightweight view object

PyTorch view semantics mean shared storage with a different shape. In this tutorial, the natural equivalent is a lightweight `TensorView`.

### 5. Reference, baseline, and backend are three different questions

```text
① Reference
   -> simplest C++ loop
   -> Is the algorithm correct?

② Baseline
   -> real data structures
   -> Can the real-sized input run?

③ Backend
   -> optimized implementation such as cuBLAS
   -> Is it faster while staying correct?
```

### 6. `torch.no_grad()` has no direct counterpart here

This tutorial is inference-only. There is no autograd engine and no training path, so `.no_grad()` is not needed.

### 7. `dtype` and `device` become explicit implementation choices

| PyTorch | Tutorial |
| --- | --- |
| `dtype=torch.float32` | `float` type or runtime type tag |
| `device='cuda'` | backend choice: CPU vs CUDA |
| `to(device)` | explicit data migration or copy |

## Minimal example

```python
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

This maps to:

- `nn.Module` -> layer / operator class
- `nn.Linear` -> `MatMul` plus optional bias
- `forward()` -> `forward()` or `compute()`
- `state_dict` -> `ModelLoader`

## How to validate

- Explain how a PyTorch module maps to a C++ layer
- Explain how a PyTorch tensor maps to the tutorial’s `Tensor`
- Explain how `state_dict` maps to `ModelLoader`
- Explain why backend and storage remain explicit in C++
- Explain when the difference is semantic and when it is implementation-level
- Explain why the tutorial does not need autograd

## Summary

If you can translate `torch.Tensor`, `nn.Module`, and `state_dict` into `Tensor`, operator classes, and `ModelLoader`, the bridge is in place.
