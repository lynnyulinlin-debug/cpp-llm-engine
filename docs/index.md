---
layout: home

hero:
  name: cpp-llm-engine
  text: 用 C++ 构建 LLM 推理引擎
  tagline: 从基础准备到基本完成的可验证 LLM 推理主线
  actions:
    - theme: brand
      text: 进入教程主入口
      link: /README
    - theme: alt
      text: 看阅读指南
      link: /guide
    - theme: alt
      text: English entry
      link: /en/README

features:
  - title: 中文主线
    details: 先把主线讲清楚，再用英文入口和术语表照顾非中文母语学习者。
  - title: 工程化入门
    details: 从 foundation 到 chapter07 顺着读，按工程问题推进，不要求一次吃完整站。
  - title: 可验证主线
    details: 单测、e2e 和 benchmark 分层推进，最后在验证章统一证据链。
---

## 这是什么

这是一个面向中文学习者优先的 C++ LLM 推理教程。  
如果你想先知道怎么读，先看 [阅读指南](/guide)；如果你想直接看主线，从 [教程主入口](/README) 进入。

中文主线已经基本完成，当前工作重点是英文正文、残余句式清理，以及参考层和维护层的持续补齐。

## 教程主线

这套教程按一条推理引擎主线往前推进，而不是按主题合集散着讲。

| 阶段 | 主要职责 |
| --- | --- |
| `chapter01` - `chapter03` | 输入、张量语义、算子边界 |
| `chapter04` - `chapter05` | 结构与运行时 |
| `chapter06` | 表示与成本 |
| `chapter07` | 验证与证据 |

`chapter07` 之后，主线转入 `reference`、`guides` 和 `maintenance`：

- `reference` 讲概念解释、术语和生态对照
- `guides` 讲构建、环境、迁移和工程方法
- `maintenance` 讲站点维护、写作规范和长期收口

想先看“指挥台 / 发动机”的分工，可以直接进 [training vs inference frameworks](/reference/framework_comparison)。

## 你会看到什么

- `foundation` 到 `chapter07_benchmark` 的主线
- 训练框架、推理框架和本教程的对照层
- 术语表、写作规范和维护规范
- 图示、表格和收口页

## 适合谁

- 想从零看懂推理引擎的人
- 想理解 runtime、KV Cache、benchmark 证据链的人
- 想把教程内容迁移到自己项目里的人

## 英文覆盖状态

- 英文导航和入口页：完成
- 章节导读和收口页：基本完成
- 章节正文完整翻译：待完善
