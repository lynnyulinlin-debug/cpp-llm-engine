# Chapter Template 章节模板

章节标题应直接表达这一章要解决的工程问题，避免只写内部备注式标题。

## What Problem Does This Chapter Solve?

用一句话说明本章要解决什么问题。

示例：

- How can `reshape` and `slice` work as view-based tensor operations without copying data?

## 为什么这个问题重要？

- 解释这个问题为什么会出现在推理引擎里
- 解释为什么必须在这一章解决，而不是后面再说
- 解释后续章节为什么依赖这一章

## 背景是什么？

先给初学者一个足够短但足够有用的背景。

- 这一章的概念从哪里来
- 它在推理引擎里对应哪一类问题
- 为什么读者现在需要它，而不是以后再学

## 这一章会产出什么？

- 这一章完成后会得到什么具体产物
- 什么可见行为可以证明它已经工作

## 最小设计是什么？

先讲思路，再讲实现。

- 这一章解决的核心问题是什么
- 为什么选择这种方式
- 接受了什么 tradeoff

## 中间推导是什么？

把“为什么会想到这个方案”写出来，而不是只给结论。

- 从问题出发，先得到哪些约束
- 从约束出发，为什么会走到这个设计
- 哪一步是关键转折点
- 如果换一种设计，会失去什么

## 实现路径是什么？

### 1. 第一步

说明第一步做什么，以及它启用了什么能力。

```cpp
// 示例代码块
```

### 2. 第二步

说明下一步做什么，以及为什么它是自然延伸。

## 怎么验证？

说明如何确认本章已经工作。

- shape 检查
- 单元测试
- 数值对比
- 端到端运行

## 常见问题是什么？

- 问题 1
- 问题 2

## 本章解决了什么？

总结本章完成了什么，以及它和下一章的关系。

## 下一步是什么？

说明教程中的自然下一步。

## English anchors

If you want to help non-Chinese readers, keep this section very short and stable:

- Problem
- Why it matters
- Minimal design
- Verification
- Next step
