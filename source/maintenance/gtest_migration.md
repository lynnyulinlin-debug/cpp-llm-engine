# Google Test 迁移预案

本文仅供维护人员评估测试框架迁移，不进入学习者导航。

## 触发条件

只有 Chapter07 明显需要共享 fixture、参数化测试、更细过滤或标准 CI 输出时，才考虑从当前轻量 harness 迁移到 Google Test。

## 迁移范围

第一阶段只处理 `code/chapter07_benchmark/tests/`、对应 `CMakeLists.txt`、代码 README 和 Chapter07 测试说明。Chapter01-06 保持原有测试入口，`ctest` 继续作为统一执行命令。

## 执行顺序

1. 引入 GTest 依赖并固定离线/在线获取策略。
2. 将现有命名测试映射为 `TEST` 或 `TEST_F`。
3. 保留 benchmark 逻辑和输出格式，不把性能测量混入 correctness test。
4. 用 `ctest` 验证新旧入口行为一致。
5. 最后更新学习者页面中的实际测试命令。
