# Foundation Exercises 基础小题库

这套题库把 `foundation` 的最低门槛落实为可编译、可测试的小题。题目默认不需要模型文件或 GPU，CUDA 题目后续单独放入选修目录。

## 当前题目

| 编号 | 主题 | 验证重点 | 对应主线 |
| --- | --- | --- | --- |
| `FQ01` | move-only Buffer | 所有权、移动、释放 | `foundation/01` |
| `FQ02` | Tensor shape | shape、numel、边界 | `chapter01` |
| `FQ03` | Tensor view | offset、共享存储 | `chapter02` |
| `FQ04` | CMake + CTest | target、构建、测试注册 | `foundation/02` |
| `FQ05` | 安全缓冲区 | 输入检查、失败路径 | `foundation/04` |
| `FQ06` | CPU MatMul | shape 检查、数值结果 | `chapter03` |
| `FQ07` | Token queue | FIFO、状态转移 | `foundation/06` / `chapter05` |
| `FQ08` | Mini Tensor | ownership、view、索引组合 | `foundation/09` / `chapter01` |

每道题都包含问题说明、完成标准、验证命令、常见错误和主线链接。当前目录中的 `.cpp` 是可运行参考实现，后续可以增加 `starter/` 和分级提示而不改变测试接口。

## 运行全部题目

```bash
bash exercises/foundation/run_all.sh
```

也可以手动构建：

```bash
cmake -S exercises/foundation -B exercises/foundation/build
cmake --build exercises/foundation/build
ctest --test-dir exercises/foundation/build --output-on-failure
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
