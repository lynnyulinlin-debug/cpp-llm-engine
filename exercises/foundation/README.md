# Foundation Exercises 基础小题库

这套题库把 `foundation` 的最低门槛落实为可编译、可测试的小题。题目默认不需要模型文件或 GPU；FQ10 提供 CPU 参考实现和可选 CUDA starter。

## 当前题目

| 编号 | 难度 | 主题 | 验证重点 | 对应主线 |
| --- | --- | --- | --- | --- |
| `FQ01` | 入门 | move-only Buffer | 所有权、移动、释放 | `foundation/01` |
| `FQ02` | 入门 | Tensor shape | shape、numel、边界 | `chapter01` |
| `FQ03` | 主线 | Tensor view | offset、共享存储 | `chapter02` |
| `FQ04` | 入门 | CMake + CTest | target、构建、测试注册 | `foundation/02` |
| `FQ05` | 入门 | 安全缓冲区 | 输入检查、失败路径 | `foundation/04` |
| `FQ06` | 主线 | CPU MatMul | shape 检查、数值结果 | `chapter03` |
| `FQ07` | 主线 | Token queue | FIFO、状态转移 | `foundation/06` / `chapter05` |
| `FQ08` | 主线 | Mini Tensor | ownership、view、索引组合 | `foundation/09` / `chapter01` |
| `FQ09` | 主线 | Request State | 状态迁移、终态、取消 | `chapter05` |
| `FQ10` | 进阶选修 | Vector Add / CUDA 映射 | CPU 参考、kernel 索引、边界 | `chapter03` / `foundation/03` |
| `FQ11` | 主线 | Request Queue | 容量、FIFO、取消、backpressure | `chapter05` |

每道题都包含问题说明、完成标准、验证命令、常见错误和主线链接。题目入口：

- [`FQ01 move-only Buffer`](./problems/fq01_move_buffer/README.md)
- [`FQ02 Tensor shape`](./problems/fq02_tensor_shape/README.md)
- [`FQ03 Tensor view`](./problems/fq03_tensor_view/README.md)
- [`FQ04 CMake + CTest`](./problems/fq04_cmake_ctest/README.md)
- [`FQ05 安全缓冲区`](./problems/fq05_debug_safe_buffer/README.md)
- [`FQ06 CPU MatMul`](./problems/fq06_cpu_matmul/README.md)
- [`FQ07 Token queue`](./problems/fq07_token_queue/README.md)
- [`FQ08 Mini Tensor`](./problems/fq08_mini_tensor/README.md)
- [`FQ09 Request State`](./problems/fq09_request_state/README.md)
- [`FQ10 Vector Add / CUDA Mapping`](./problems/fq10_vector_add/README.md)
- [`FQ11 Request Queue`](./problems/fq11_request_queue/README.md)

当前目录中的 `.cpp` 是可运行参考实现；每道题的 `starter/` 目录提供不含答案的最小起点。

## 推荐题库顺序

- 入门：`FQ01`、`FQ02`、`FQ04`、`FQ05`
- 主线：`FQ03`、`FQ06`、`FQ07`、`FQ08`、`FQ09`、`FQ11`
- 进阶选修：`FQ10`

初学者完成入门组即可进入 `chapter00`；希望理解 generation runtime 的学习者继续完成主线组；已有 GPU 基础的学习者再做 FQ10。

## 运行全部题目

```bash
bash exercises/foundation/run_all.sh
```

也可以手动构建：

```bash
cmake -S exercises/foundation -B exercises/foundation/build
cmake --build exercises/foundation/build
ctest --test-dir exercises/foundation/build --output-on-failure
npm run test:regression
```

内存安全题可以额外启用 Sanitizer：

```bash
cmake -S exercises/foundation -B exercises/foundation/build-asan -DENABLE_ASAN=ON
cmake --build exercises/foundation/build-asan
ctest --test-dir exercises/foundation/build-asan --output-on-failure
```

当前受控执行环境会限制 AddressSanitizer 的运行时和 LeakSanitizer；如果出现 `DEADLYSIGNAL` 或 `LeakSanitizer has encountered a fatal error`，应在普通本地环境重跑。当前环境使用 UBSan 做可运行验证：

```bash
cmake -S exercises/foundation -B exercises/foundation/build-ubsan \
  -DCMAKE_CXX_FLAGS=-fsanitize=undefined
cmake --build exercises/foundation/build-ubsan
ctest --test-dir exercises/foundation/build-ubsan --output-on-failure
```

## 题目规范

学习目标统一写成“动作 + 对象 + 条件 + 验证结果”。例如：完成 `FQ03` 后，应能在不复制底层数组的前提下创建 view，并通过测试确认修改 view 会反映到原始 storage。
