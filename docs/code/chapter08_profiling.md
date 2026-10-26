# chapter08_profiling 代码导航

这一页定位 profiling evidence、A/B 判定、测试和运行脚本。Chapter 08 消费 Chapter 07 的实测 benchmark，不重新实现 Runtime。

## 阅读顺序

1. 先看 `profile_record.h`，确认教学与真实服务输入字段和验证规则
2. 再看 `ab_compare.h`，确认 correctness-first 判定
3. 最后看报告示例、测试和脚本

## 核心实现

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter08_profiling/include/chapter08_profiling/profile_record.h"><code>profile_record.h</code></a> | 实测字段和报告接口 |
| <a href="/_code/chapter08_profiling/include/chapter08_profiling/ab_compare.h"><code>ab_compare.h</code></a> | A/B 接受策略 |
| <a href="/_code/chapter08_profiling/src/profile_record.cpp"><code>src/profile_record.cpp</code></a> | 解析和指标计算 |
| <a href="/_code/chapter08_profiling/src/ab_compare.cpp"><code>src/ab_compare.cpp</code></a> | 正确性与延迟回退判断 |

## 验证与脚本

| 文件 | 作用 |
| --- | --- |
| <a href="/_code/chapter08_profiling/examples/main_profiling.cpp"><code>examples/main_profiling.cpp</code></a> | 报告和 A/B 命令行入口 |
| <a href="/_code/chapter08_profiling/tests/chapter08_profiling_test.cpp"><code>tests/chapter08_profiling_test.cpp</code></a> | 章节测试 |
| <a href="/_code/chapter08_profiling/scripts/run.sh"><code>scripts/run.sh</code></a> | 消费 Chapter 07 实测 benchmark |
| <a href="/_code/chapter08_profiling/scripts/select_real_service_profile.mjs"><code>select_real_service_profile.mjs</code></a> | 适配真实 ServiceCore JSON |
| <a href="/_code/chapter08_profiling/scripts/run_real_service_ab.sh"><code>run_real_service_ab.sh</code></a> | 真实 baseline/experiment 回归 A/B |
| <a href="/_code/chapter08_profiling/scripts/run_real_q4k_attention_ab.sh"><code>run_real_q4k_attention_ab.sh</code></a> | Q4_K QKV/out 三轮 scalar/AVX2 A/B |
| <a href="/_code/chapter08_profiling/scripts/compare_q4k_attention_profiles.mjs"><code>compare_q4k_attention_profiles.mjs</code></a> | 校验作用域、正确性、计时闭合并生成判定 |
| <a href="/_code/engine/examples/real_tinyllama_decoder_profile.cpp"><code>real_tinyllama_decoder_profile.cpp</code></a> | 真实 Decoder 子阶段计时与 golden 验证 |

## 对应文档

- [chapter08 导读](../chapter08_profiling/intro.md)
- [真实 TinyLlama 服务 Profiling](../chapter08_profiling/08_real_service_profile.md)
- [CPU / GPU 统一架构](../reference/cpu_gpu_architecture.md)
