# 00.2 Clone and Build `llama.cpp` 克隆并编译 `llama.cpp`

**阅读入口**

- 前置阅读：[`00.1 Environment Setup`](./01_environment_setup.md)
- 源码直达：<a href="../_code/chapter00_quickstart/main_quick.cpp"><code>main_quick.cpp</code></a>、<a href="../_code/chapter00_quickstart/run.sh"><code>run.sh</code></a>
- 相关阅读：[`00.3 Download TinyLlama`](./03_download_tinyllama.md)、[`00.4 First Inference`](./04_first_inference.md)

## 开始前

把 `llama.cpp` 编译出来，并准备好后续推理所需的执行文件。

**完成后：**

- 知道如何获取源码
- 知道 CPU 版和 CUDA 版怎么切换
- 知道编译成功后应该得到什么产物

**前置条件：**

- 已完成环境准备
- 本地磁盘空间足够
- 网络可用，至少能访问 GitHub 或镜像源

## 编译 `llama.cpp`

```bash
cd /data/cpp-llm-engine
test -d llama.cpp/.git || git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
cd llama.cpp
```

### CPU 版本

```bash
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

这个路径最适合先做 smoke test。只要 CPU 路径能跑通，就说明大部分环境问题都排除了。

这里建议每次都使用独立的 `build` 目录，而不是在源码目录里混着编译产物。这样做的好处是：清理简单、切换编译选项方便、出问题时更容易重建。

### 可选 CUDA 版本

如果你确认本机具备 CUDA 环境，可以切到 GPU 构建：

```bash
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j"$(nproc)"
```

如果 CUDA 构建失败，不要立刻怀疑教程内容本身。先确认驱动、toolkit、编译器版本和环境变量是否一致，再决定是不是要退回 CPU 版本继续往下走。

## 验证和排错

- `build` 目录下生成了可执行文件
- 当前版本通常会生成 `build/bin/llama-simple`；不同版本的示例程序名称可能变化，应以 `build/bin` 中的实际文件为准
- CPU 版本可以启动
- 如果启用了 CUDA，编译过程没有报错

如果编译产物已经生成，说明你已经跨过了“环境是否可用”的第一道门槛。后面的模型下载和推理启动才有意义。

**常见问题：**

- 编译器版本不满足要求
- 依赖库缺失
- CUDA 开关打开后构建失败
- Git clone 太慢或失败
