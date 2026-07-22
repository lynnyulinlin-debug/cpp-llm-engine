# cpp-llm-engine

用 C++ 构建 LLM 推理引擎

_A Practical Tutorial for Building a C++ LLM Inference Engine_

这是一个面向中文学习者优先的工程教程仓库。主教程入口在 [docs/README.md](docs/README.md)，英文入口在 [docs/README.en.md](docs/README.en.md) 和 [/en/README](/en/README)。如果先看阅读路径，先看 [USAGE.md](USAGE.md)。

## 项目简介

- 中文主线优先，英文入口逐步补齐
- 主线已经完成收口，当前重点是英文正文、残余句式清理，以及参考层和维护层补齐
- 教程目标是把一条可顺着读、顺着跑、顺着改的 C++ LLM 推理路径讲清楚

## 如何开始

1. 先看 [USAGE.md](USAGE.md)
2. 再看 [docs/README.md](docs/README.md)
3. 想先看站点首页，可以打开 [docs/index.md](docs/index.md)
4. 想先看中文主入口，可以打开 [docs/README.md](docs/README.md)
5. 想看英文入口，可以打开 [docs/README.en.md](docs/README.en.md)

## 教程地图

- `foundation`：基础准备
- `chapter00_quickstart`：第一次真实推理
- `chapter01` - `chapter03`：输入、张量语义、算子边界
- `chapter04` - `chapter05`：结构与运行时
- `chapter06`：表示与成本
- `chapter07`：验证与证据
- `guides` / `reference` / `maintenance`：支撑层、参考层和维护层

## 当前状态

- 中文主线已经收口到“导读 / 机制 / 收口”三层
- 代码直达已经统一到具体实现文件
- `docs:check-links` 已纳入维护流程
- 英文正文仍在推进中

## 相关入口

- [docs/README.md](docs/README.md)：中文主入口
- [docs/README.en.md](docs/README.en.md)：英文入口
- [docs/guide.md](docs/guide.md)：阅读指南
- [source/reference/README.md](source/reference/README.md)：设计参考入口
- [source/maintenance/README.md](source/maintenance/README.md)：维护文档入口

## 构建与检查

```bash
npm install
npm run docs:check-links
npm run docs:build
npm run docs:preview
```

如果你的环境需要指定镜像：

```bash
npm install --registry=https://registry.npmjs.org
```
