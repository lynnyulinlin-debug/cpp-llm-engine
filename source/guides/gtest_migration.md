# Google Test Migration 迁移方案

这页只回答一个问题：如果未来要把 `chapter07` 从当前的测试 harness 升级到 Google Test，应该怎么做。

## 当前前提

当前仓库的测试体系是：

- `chapter00_quickstart`
  - smoke test / 命令行验证
- `chapter01` 到 `chapter06`
  - 章节内最小可执行测试和 `assert` 级验证
- `chapter07`
  - 命名测试用例 + 统一测试入口的 harness

这意味着 Google Test 现在还不是已实现方案，而是未来的可选升级方向。

## 什么时候迁移

只有当 `chapter07` 开始明显出现下面这些信号时，才建议迁移：

- 需要共享 fixture
- 需要参数化测试
- 需要更细的测试分组和过滤
- 需要更标准化的 CI 输出
- 需要减少重复测试样板代码

如果这些信号还不明显，继续使用当前 harness 更轻，也更符合教程的阶段目标。

## 迁移范围

第一阶段只迁移 `chapter07`，不动前面章节。

建议迁移对象：

- `code/chapter07_benchmark/tests/`
- `code/chapter07_benchmark/CMakeLists.txt`
- `code/chapter07_benchmark/README.md`
- `docs/chapter07_benchmark/intro.md`
- `docs/chapter07_benchmark/summary.md`
- `docs/maintenance/tutorial_maintenance.md`

## 迁移顺序

1. 在 `chapter07` 中引入 GTest 依赖
2. 把当前命名测试用例映射成 `TEST` 或 `TEST_F`
3. 保留 benchmark 相关逻辑和输出格式
4. 让 `ctest` 继续作为统一执行入口
5. 再更新文档里的测试描述

## 不做什么

- 不先迁移 `chapter01` 到 `chapter06`
- 不把 benchmark 和测试混成一层
- 不为了引入 GTest 而重构整个教程测试体系

## 这页的作用

这不是现在就要执行的改动清单，而是一个预案。

它的作用是把未来升级节点、升级范围和升级顺序固定下来，避免后面真的要迁移时临时决策。
