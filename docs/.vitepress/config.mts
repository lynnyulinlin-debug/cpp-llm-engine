import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'cpp-llm-engine',
  description: '用 C++ 构建 LLM 推理引擎',
  lang: 'zh-CN',
  cleanUrls: true,
  lastUpdated: true,
  locales: {
    '/': {
      lang: 'zh-CN',
      label: '中文',
      title: 'cpp-llm-engine',
      description: '用 C++ 构建 LLM 推理引擎',
    },
    '/en/': {
      lang: 'en-US',
      label: 'English',
      title: 'C++ LLM Inference Engine Tutorial',
      description: 'Build an LLM inference engine from scratch',
    },
  },
  themeConfig: {
    sidebar: createSidebar(),
    nav: [
      { text: '首页', link: '/' },
      { text: '教程主入口', link: '/README' },
      { text: '英文入口', link: '/en/README' },
      { text: '使用文档', link: '/guide' },
      { text: '开发参考', link: '/guides/README' },
      { text: '设计参考', link: '/reference/README' },
      { text: '维护文档', link: '/maintenance/README' },
      { text: '代码索引', link: '/code/README' },
    ],
    locales: {
      '/en/': {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Tutorial Entry', link: '/en/README' },
          { text: 'Guide', link: '/en/guide' },
          { text: 'Development', link: '/en/guides/README' },
          { text: 'Reference', link: '/en/reference/README' },
          { text: 'Maintenance', link: '/en/maintenance/README' },
        ],
        sidebar: createEnglishSidebar(),
      },
    },
  },
})

type SidebarItem = {
  text: string
  link?: string
  collapsed?: boolean
  items?: SidebarItem[]
}

function numberedText(index: string | number, text: string) {
  return `${index} ${text}`
}

function chapterGroup(chapterCode: string, text: string, base: string, pages: Array<{ text: string; file: string }>) {
  return {
    text,
    collapsed: true,
    items: [
      { text: numberedText(`${chapterCode}.0`, '本章导读'), link: `${base}/intro` },
      ...pages.map((page, index) => ({ text: numberedText(`${chapterCode}.${index + 1}`, page.text), link: `${base}/${page.file.replace(/\.md$/, '')}` })),
      { text: numberedText(`${chapterCode}.${pages.length + 1}`, '本章小结'), link: `${base}/summary` },
    ],
  }
}

function codeGroup(chapterCode: string, text: string, base: string) {
  return {
    text,
    collapsed: true,
    items: [
      { text: `${chapterCode} 代码索引`, link: `${base}` },
    ],
  }
}

function chapterLandingGroup(chapterCode: string, text: string, base: string) {
  return {
    text,
    collapsed: true,
    items: [
      { text: numberedText(`${chapterCode}.0`, 'Intro'), link: `/en${base}/intro` },
      { text: numberedText(`${chapterCode}.1`, 'Summary'), link: `/en${base}/summary` },
    ],
  }
}

function createEnglishSidebar() {
  return {
    '/en/': [
      {
        text: 'Tutorial Guide',
        collapsed: false,
        items: [
          { text: 'Home', link: '/en/' },
          { text: 'Tutorial Entry', link: '/en/README' },
          { text: 'Guide', link: '/en/guide' },
          { text: 'Development References', link: '/en/guides/README' },
          { text: 'Design References', link: '/en/reference/README' },
          { text: 'Maintenance', link: '/en/maintenance/README' },
          { text: 'Foundation', link: '/en/foundation/README' },
          chapterLandingGroup('00', 'Chapter 00 Quickstart', '/chapter00_quickstart'),
          chapterLandingGroup('01', 'Chapter 01 Tensor Loader', '/chapter01_tensor_loader'),
          chapterLandingGroup('02', 'Chapter 02 Tensor Ops', '/chapter02_tensor_ops'),
          chapterLandingGroup('03', 'Chapter 03 MatMul', '/chapter03_matmul'),
          chapterLandingGroup('04', 'Chapter 04 Attention', '/chapter04_attention'),
          chapterLandingGroup('05', 'Chapter 05 Generation', '/chapter05_generation'),
          chapterLandingGroup('06', 'Chapter 06 Quantization', '/chapter06_quantization'),
          chapterLandingGroup('07', 'Chapter 07 Benchmark', '/chapter07_benchmark'),
        ],
      },
    ],
  }
}

function createSidebar() {
  return {
    '/': [
      {
        text: '教程主线',
        collapsed: false,
        items: [
          { text: '主入口', link: '/README' },
          { text: '英文入口', link: '/en/README' },
          { text: '使用文档', link: '/guide' },
          { text: '开发参考', link: '/guides/README' },
          { text: '设计参考', link: '/reference/README' },
          { text: '维护文档', link: '/maintenance/README' },
          { text: '代码索引', link: '/code/README' },
          chapterGroup('00', '第 00 章 起跑线', '/chapter00_quickstart', [
            { text: '环境准备', file: '01_environment_setup.md' },
            { text: '克隆并编译 llama.cpp', file: '02_clone_and_build_llamacpp.md' },
            { text: '下载 TinyLlama', file: '03_download_tinyllama.md' },
            { text: '第一次推理', file: '04_first_inference.md' },
            { text: '排查问题', file: '05_troubleshooting.md' },
          ]),
          {
            text: '基础准备',
            collapsed: true,
            items: [
              { text: numberedText(1, '总览'), link: '/foundation/README' },
              { text: numberedText(2, 'C++ 基础'), link: '/foundation/01_cxx_basics' },
              { text: numberedText(3, 'CMake 和编译基础'), link: '/foundation/02_cmake_build_basics' },
              { text: numberedText(4, 'CUDA 和环境基础'), link: '/foundation/03_cuda_environment_basics' },
              { text: numberedText(5, '调试与测试基础'), link: '/foundation/04_debugging_testing_basics' },
              { text: numberedText(6, 'OOP 和设计基础'), link: '/foundation/05_oop_and_design_basics' },
              { text: numberedText(7, '数据结构和算法基础'), link: '/foundation/06_data_structure_algorithm_basics' },
              { text: numberedText(8, 'CUDA 常用库'), link: '/foundation/07_cuda_common_libraries' },
              { text: numberedText(9, 'PyTorch 到 C++ 桥接'), link: '/foundation/08_pytorch_to_cpp_bridge' },
              { text: numberedText(10, '最小 C++ / CUDA 项目'), link: '/foundation/09_minimal_cpp_cuda_project' },
            ],
          },
          chapterGroup('01', '第 01 章 权重加载', '/chapter01_tensor_loader', [
            { text: 'Tensor 接口', file: '01_tensor_interface.md' },
            { text: 'Tensor 实现', file: '02_tensor_implement.md' },
            { text: '自定义二进制格式', file: '03_custom_binary_format.md' },
            { text: 'Python 转换器', file: '04_python_converter.md' },
            { text: '模型加载器', file: '05_model_loader.md' },
            { text: '验证与打印', file: '06_verify_and_print.md' },
          ]),
          chapterGroup('02', '第 02 章 张量视图', '/chapter02_tensor_ops', [
            { text: '共享存储', file: '01_shared_ptr_memory.md' },
            { text: 'reshape', file: '02_reshape.md' },
            { text: 'slice', file: '03_slice.md' },
            { text: '单元测试', file: '04_uint_tests.md' },
          ]),
          chapterGroup('03', '第 03 章 矩阵乘法', '/chapter03_matmul', [
            { text: '算子基类', file: '01_operator_base.md' },
            { text: 'MatMul 设计', file: '02_matmul_design.md' },
            { text: 'CPU 参考实现', file: '03_cpu_reference.md' },
            { text: 'cuBLAS 实现', file: '04_cublas_implement.md' },
            { text: 'PyTorch 对照', file: '05_test_with_pytorch.md' },
            { text: '复杂度直觉', file: '06_complexity_intuition.md' },
          ]),
          chapterGroup('04', '第 04 章 Attention', '/chapter04_attention', [
            { text: '多头注意力', file: '01_multihead_attention.md' },
            { text: 'LayerNorm 和激活', file: '02_layernorm_and_activation.md' },
            { text: '前馈网络', file: '03_feedforward.md' },
            { text: 'DecoderLayer', file: '04_decoder_layer.md' },
            { text: '加载第一层权重', file: '05_load_first_layer.md' },
            { text: 'Gemma / TensorRT-LLM 对照', file: '06_gemma_tensorrt_comparison.md' },
          ]),
          chapterGroup('05', '第 05 章 生成运行时', '/chapter05_generation', [
            { text: 'Embedding 与位置编码', file: '01_embedding_and_position.md' },
            { text: 'Transformer 组装', file: '02_transformer_assembly.md' },
            { text: '朴素生成', file: '03_naive_generation.md' },
            { text: 'KV Cache', file: '04_kv_cache.md' },
            { text: 'Sampler', file: '05_sampler.md' },
            { text: '端到端生成测试', file: '06_end_to_end_generation.md' },
            { text: 'vLLM 对照', file: '07_vllm_comparison.md' },
          ]),
          chapterGroup('06', '第 06 章 量化', '/chapter06_quantization', [
            { text: '量化基础', file: '01_quantization_basics.md' },
            { text: '扩展 Tensor', file: '02_extend_tensor.md' },
            { text: '加载量化权重', file: '03_load_quantized_weights.md' },
            { text: '反量化算子', file: '04_dequantize_op.md' },
            { text: '量化 Linear', file: '05_linear_with_quant.md' },
            { text: '精度验证', file: '06_accuracy_validation.md' },
            { text: 'TensorRT-LLM / Gemma / vLLM 对照', file: '07_tensorrt_gemma_vllm_comparison.md' },
          ]),
          chapterGroup('07', '第 07 章 验证与基准测试', '/chapter07_benchmark', [
            { text: '测试框架搭建', file: '01_test_harness_setup.md' },
            { text: '端到端正确性', file: '02_e2e_correctness.md' },
            { text: 'Benchmark 指标', file: '03_benchmark_metrics.md' },
            { text: 'Benchmark 脚本', file: '04_benchmark_script.md' },
            { text: '优化建议', file: '05_optimization_tips.md' },
            { text: '主流对照', file: '07_framework_comparison.md' },
          ]),
        ],
      },
      {
        text: '查阅与维护',
        collapsed: true,
        items: [
          { text: '开发参考总入口', link: '/guides/README' },
          { text: 'CMake 常用技巧', link: '/guides/cmake_guide' },
          { text: 'CUDA 安装与配置', link: '/guides/setup_cuda' },
          { text: 'GTest 迁移预案', link: '/guides/gtest_migration' },
          { text: 'GGUF 解析', link: '/guides/gguf_parser' },
          { text: '语言策略', link: '/guides/language_strategy' },
          { text: '设计参考总入口', link: '/reference/README' },
          { text: '训练框架 vs 推理框架', link: '/reference/framework_comparison' },
          { text: '术语表', link: '/reference/glossary' },
          { text: '进一步阅读', link: '/reference/further_reading' },
          { text: 'FAQ', link: '/reference/faq' },
          { text: '生态对照', link: '/reference/ecosystem/README' },
          { text: '维护文档总入口', link: '/maintenance/README' },
          { text: '教程维护文档', link: '/maintenance/tutorial_maintenance' },
          { text: '写作规范', link: '/maintenance/tutorial_writing_guidelines' },
          { text: '章节模板', link: '/maintenance/chapter_template' },
          { text: '收口模板', link: '/maintenance/summary_template' },
        ],
      },
    ],
    '/code/': [
      {
        text: '代码索引',
        collapsed: false,
        items: [
          { text: '总览', link: '/code/README' },
          codeGroup('00', '第 00 章 起跑线', '/code/chapter00_quickstart'),
          codeGroup('01', '第 01 章 权重加载', '/code/chapter01_tensor_loader'),
          codeGroup('02', '第 02 章 张量视图', '/code/chapter02_tensor_ops'),
          codeGroup('03', '第 03 章 矩阵乘法', '/code/chapter03_matmul'),
          codeGroup('04', '第 04 章 Attention', '/code/chapter04_attention'),
          codeGroup('05', '第 05 章 生成运行时', '/code/chapter05_generation'),
          codeGroup('06', '第 06 章 量化', '/code/chapter06_quantization'),
          codeGroup('07', '第 07 章 验证与基准测试', '/code/chapter07_benchmark'),
        ],
      },
    ],
  }
}
