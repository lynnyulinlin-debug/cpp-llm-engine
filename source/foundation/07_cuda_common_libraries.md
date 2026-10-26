# F07 Common CUDA Libraries CUDA 常用库

CUDA 库按问题选择，不按名气堆叠。Foundation 只建立映射，具体 kernel 和 benchmark 放在 Chapter 03、06、07、08。

## 1. 主线位置与目标

| 问题 | 常见工具 | 教程中的位置 |
| --- | --- | --- |
| dense MatMul | cuBLAS | Chapter 03 backend |
| kernel 模板和 tile | CUTLASS | Chapter 03/08 扩展阅读 |
| 设备端容器和排序 | Thrust、CUB | 算子或辅助路径 |
| 重复 launch 开销 | CUDA Graph | Chapter 08 |
| 多 GPU 通信 | NCCL | 服务扩展，不是 CPU 主线 |
| 深度学习算子集合 | cuDNN | 外部框架对照 |

## 2. 选择顺序

```text
CPU reference
  -> CUDA kernel / cuBLAS
  -> 正确性对照
  -> benchmark
  -> 再考虑 Graph、异步拷贝或通信
```

异步拷贝和 CUDA Graph 是优化手段，不能替代一个可验证的同步实现。单 GPU 的教程主线也不需要先引入 NCCL。

## 3. 最小例子

```cpp
// CPU reference 和 CUDA kernel 使用相同的输入、输出和边界条件。
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
```

FQ10 用这个最小问题建立 CPU/CUDA 对照；它不覆盖 cuBLAS、CUDA Graph、CUB 和 NCCL 的完整 API。

## 4. 验证与练习

验证顺序是：解释库解决的问题、指出它位于算子还是 runtime、用 CPU 结果对照 GPU 结果。对应练习见 [Foundation 正文与练习映射](./exercise_map.md)，完成 FQ10；完整库实践题后续再补。

## 5. 总结

库选择必须由瓶颈和验证证据驱动。先完成 CPU 正确性，再引入一个 CUDA 后端或一个高性能库，最后测量是否真的减少了时间或内存成本。
