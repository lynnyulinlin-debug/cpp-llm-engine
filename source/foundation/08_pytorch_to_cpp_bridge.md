# F08 PyTorch to C++ Bridge PyTorch 到 C++ 的桥接

## 目标

帮助已经熟悉 PyTorch 的读者，把心智模型翻译成这套教程里的 C++ 对象、加载链路和 runtime 边界。

## 读完你应该能做什么

- 把常见 PyTorch 概念映射到教程里的 C++ 概念
- 理解如何用 PyTorch 的思维方式阅读这些 C++ 章节
- 理解哪些地方是“同样的想法，不同的语言”
- 理解为什么这套教程把 PyTorch 当参考，而不是运行时本身

## 为什么重要

很多读者已经会 tensor、module、state dict 和 forward pass。真正难的不是机器学习概念，而是切换到 C++ 后的所有权、文件布局、backend 选择和显式 runtime 结构。

这一章就是桥接层：帮助读者复用已有认知，而不是从零再学一遍。但它不是“术语对照表”，而是“工程语义翻译器”。

目的不是让 PyTorch 消失，而是把熟悉的概念翻成这套教程里更显式的 C++ 形式，这样读者就能从“我懂概念”走到“我知道它在代码里哪儿、由谁负责、在哪一层发生变化”。

## 后面会用到什么

- `Tensor` 和 `ModelLoader`
- `MatMul`、`Attention` 和 `Generation`
- `Quantization` 和 benchmark 验证
- 用 PyTorch 心智模型阅读教程的 C++ 源码

## 与 00.5 的职责呼应

| 00.5 职责分类 | PyTorch 概念 | 教程类 |
| --- | --- | --- |
| 数据容器 | `torch.Tensor` | `Tensor` |
| 文件入口 | `state_dict` | `ModelLoader` |
| 计算节点 | `nn.Linear` | `MatMulOp` |
| 状态壳 | runtime 状态 | `KVCache` / `Runtime` |
| 组合结构 | `nn.Module` | `DecoderLayer` |

## 这一页的核心内容

### 1. Tensor 不是“数组”，而是“带语义的数据对象”

在 PyTorch 里，tensor 天生就是一个带 shape、dtype、device 和 view 行为的一等对象。在这套教程里，这些概念会变成显式的 C++ 字段和所有权规则。

工程上最重要的变化是：C++ 教程不会把 storage 藏在框架后面。读者需要直接看到：

- 所有权在哪儿
- view 从哪儿来
- tensor 如何共享数据
- tensor 在哪一层完成 device 转换

`dtype` 和 `device` 也要按 C++ 语义来理解：

| PyTorch | 教程 |
| --- | --- |
| `dtype=torch.float32` | 使用 `float` 类型或运行时类型标记 |
| `device='cuda'` | 通过 backend 选择决定 CPU vs CUDA |
| `to(device)` | 显式内存拷贝或数据迁移 |

关键变化是：在这套教程里，device 通常是对象级别的运行路径选择，而不是 `Tensor` 自己的隐式属性。

### 2. `nn.Module` 对应 operator / layer / runtime unit

`nn.Module` 在这套教程里通常对应一个 operator 或 layer 类。forward path 的思路是一样的，但 C++ 版本会把所有权、接口和 backend 选择写得更显式。

这意味着：

- 计算逻辑和状态要分开
- 参考实现和 backend 实现要分开
- 层级组合和底层算子要分开

### 3. `state_dict` 对应命名加载和对象重建

PyTorch 的 `state_dict` 很适合帮助理解这套教程里的 `ModelLoader` 是如何把名称映射到 tensor 的。

这套教程保留这个映射，不只是为了读模型文件，而是为了从文件里重建一套一致的对象模型。换句话说，加载链路要回答的不只是“读到了什么”，还有“这些东西在程序里应该变成什么对象”。

### 4. `view` 在 C++ 里对应轻量视图对象

PyTorch 的 view 语义是“共享 storage，但 shape 不同”。在这套教程里，最自然的对应就是一个轻量的 `TensorView`：

- 持有底层 buffer 的指针或引用
- 保存 `offset` 和 `shape`
- 自己不拥有数据，只是观察它

这也是为什么 `02 Tensor Ops` 里再讨论 view 会更自然：它会和 slice、reshape、共享 buffer 一起出现。

### 5. reference、baseline 和 backend 是三层不同问题

在 PyTorch 里，你通常会先写一个干净的 reference 版本。这套教程也是一样：先写 CPU reference baseline，再写像 cuBLAS 这样的 backend。

这三层各自解决不同问题：

- reference：算法语义是否清楚
- baseline：当前实现是否能跑通
- backend：是否有可验证的加速路径

可以把这个演进过程理解成下面三步：

```text
① Reference
   -> 用最简单的 C++ 循环实现
   -> 验证：这个小输入算对了吗？

② Baseline
   -> 用真实数据结构重写
   -> 验证：真实尺寸能跑通吗？

③ Backend
   -> 用 cuBLAS 等库优化
   -> 验证：比 baseline 快吗？正确性一致吗？
```

### 6. forward pass 在 C++ 里会变成显式 runtime flow

PyTorch 会隐藏很多 runtime 细节，而这套教程会把这些流转显式化，让读者看清 memory、layout、cache 和 scheduler 到底在什么时候起作用。

这种显式化对后面几章很关键，因为 generation runtime、quantization path 和 benchmark 都依赖这条流转是清楚的。

### 7. autograd 在 C++ 中没有对应物

`torch.no_grad()` 在这套教程里不需要对应，因为 C++ 教程写的是推理引擎，根本没有训练路径。

这意味着：

- 没有 autograd engine
- 没有梯度计算
- 所有 forward 计算天然都是推理模式

如果后面看到某段 C++ 代码没有 `.no_grad()`，不是忘了加，而是压根不需要加。

## 最小示例

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

这个例子在教程里可以这样映射：

- `nn.Module` -> layer / operator class
- `nn.Linear` -> `MatMul` 加可选 bias
- `forward()` -> `forward` / `compute`
- `state_dict` -> `ModelLoader` 的名称查找

再往工程层面翻一层，就是：

- module 的参数归属由 loader 决定
- layer 的执行边界由 runtime 决定
- backend 的替换由章节目标决定

重要补充是：`torch.Tensor` 的 `device` 在教程里通常不是 `Tensor` 的隐式属性，而是由 backend 和数据迁移显式决定的。

重要结论是：大多数概念都能对上，但这套教程会刻意把所有权、storage 和 backend 选择写得更明确，而不是藏起来。

## 对照表

| PyTorch 概念 | 教程概念 | C++ 里变化了什么 |
| --- | --- | --- |
| `torch.Tensor` | `Tensor` | 所有权和 storage 显式化 |
| `nn.Module` | operator / layer 类 | 接口手写，更明确 |
| `state_dict` | `ModelLoader` | 文件解析和名称查找显式化 |
| `forward()` | `forward()` | 思路相同，但 runtime 流更显式 |
| `torch.no_grad()` | 不需要对应 | 教程本身没有训练路径 |
| `torch.matmul` | `MatMulOp` | backend 选择和 layout 变得重要 |
| `torch.nn.functional` | 底层 helper ops | 更多样板，少一点魔法 |

## 怎么验证

- 解释 PyTorch module 如何映射到 C++ layer
- 解释 PyTorch tensor 如何映射到教程里的 `Tensor`
- 解释 state dict 如何映射到 `ModelLoader`
- 解释为什么教程仍然需要显式 backend 和 storage 逻辑
- 解释什么时候是语义映射，什么时候是实现差异
- 解释 `dtype`、`device` 和 `view` 在 C++ 里分别怎么落地
- 解释为什么教程里不需要 autograd

## 常见问题

- 以为 C++ 会像 PyTorch 一样把细节都藏起来
- 只找 API 一一对应，而不是找概念对应
- 把 PyTorch 当成实现，而不是参考
- 因为 PyTorch 会隐藏内存，就忽略了所有权问题
- 把“能运行”误当成“语义已经对齐”
- 没区分 loader、layer 和 backend 的职责
- 以为 C++ 推理路径里也需要 `.no_grad()`
- 把 device 当成 Tensor 的隐式属性，而不是 backend 选择

如果你本来就会 PyTorch，这一章的目标是翻译你的心智模型，而不是重新学一遍机器学习。

## 总结

读完这一章后，读者应该能带着 PyTorch 的心智模型阅读这套 C++ 教程，并知道哪里是概念相同、哪里是 C++ 故意更显式。

如果你能把 `torch.Tensor`、`nn.Module` 和 `state_dict` 翻译成这套教程里的 `Tensor`、operator 类和 `ModelLoader`，那这座桥就算搭好了。
