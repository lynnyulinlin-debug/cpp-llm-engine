# 01.1 Tensor Object Tensor 对象

## 学习目标

模型文件中的权重进入程序后，需要统一的对象表达名称、形状、类型、设备和数据存储。本节将接口与实现放在一起，读完后应能：

- 区分逻辑形状与物理存储
- 说明 `shared_ptr<Storage>` 和 `offset_bytes` 的作用
- 检查 Tensor 的构造、复制、移动和原始数据访问

代码入口：

- <a href="../_code/chapter01_tensor_loader/src/tensor.h"><code>tensor.h</code></a>
- <a href="../_code/chapter01_tensor_loader/src/tensor.cpp"><code>tensor.cpp</code></a>
- <a href="../_code/chapter01_tensor_loader/tests/chapter01_tensor_loader_test.cpp"><code>chapter01_tensor_loader_test.cpp</code></a>

## 数据表示与所有权

`Tensor` 用一组稳定字段连接 loader、算子和 runtime：

| 信息 | 当前实现 | 后续用途 |
| --- | --- | --- |
| 名称 | `name_` | 按模型 schema 定位权重 |
| 逻辑布局 | `shape_`、`dtype_` | 校验元素数量和算子输入 |
| 执行位置 | `device_`，当前为 CPU | 后续选择 CPU/GPU backend |
| 物理数据 | `shared_ptr<vector<byte>>` | 共享 storage 并管理生命周期 |
| 视图起点 | `offset_bytes_` | Chapter02 的 slice/view |
| 有效载荷 | `payload_bytes_` | 支持量化 tensor 的真实字节数 |

`shape` 描述逻辑视图，storage 保存物理字节。两者分开后，多个 Tensor 才能共享同一块内存，并用不同 shape 或 offset 解释数据。Chapter02 会在此基础上实现 `reshape` 和 `slice`。

构造路径也反映了两类输入：`from_vector<T>()` 适合测试和普通数值，`from_raw_bytes()` 保留 GGUF 量化 payload。复制 Tensor 时共享 storage；移动 Tensor 时转移对象状态，资源释放由 RAII 完成。

## 验证

从仓库根目录运行：

```bash
bash code/chapter01_tensor_loader/run.sh
```

在测试中确认以下结果：

1. `numel()` 与 shape 一致。
2. `bytes()` 和 raw payload 大小一致。
3. `raw_data()` 已包含 `offset_bytes`。
4. 复制和移动后数据仍可访问。
5. 非法 shape、dtype 或越界 offset 会明确失败。

