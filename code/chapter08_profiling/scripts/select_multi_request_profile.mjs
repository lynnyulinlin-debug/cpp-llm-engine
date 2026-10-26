#!/usr/bin/env node

// Chapter 08：从 Chapter 07 多请求 JSON 中选择一个 profiling 场景。
// 职责：把稳定的 JSON 结果适配为现有 C++ ProfileRecord 的 key=value 输入。
// 验证：检查请求数、correctness、完成请求数和生成 token 数后再输出记录。

import fs from "node:fs";

const [inputPath, requestCountText, outputPath] = process.argv.slice(2);
if (!inputPath || !requestCountText || !outputPath) {
  console.error("usage: select_multi_request_profile.mjs INPUT_JSON REQUEST_COUNT OUTPUT_TXT");
  process.exit(2);
}

const requestCount = Number(requestCountText);
const document = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const scenario = document.scenarios?.find((item) => item.request_count === requestCount);
if (!scenario) {
  throw new Error(`request_count=${requestCount} not found`);
}
if (!scenario.correctness || scenario.completed_requests !== requestCount) {
  throw new Error(`scenario request_count=${requestCount} failed correctness`);
}
if (scenario.generated_tokens !== requestCount * scenario.max_new_tokens) {
  throw new Error(`scenario request_count=${requestCount} has incomplete output`);
}

const lines = [
  `runtime_benchmark=chapter07_cpu_multi_request_${requestCount}`,
  "backend=cpu",
  `correctness=${scenario.correctness ? "PASS" : "FAIL"}`,
  `iterations=${document.iterations}`,
  `request_count=${scenario.request_count}`,
  `generated_tokens_per_iteration=${scenario.generated_tokens}`,
  `prefill_ms=${scenario.prefill_ms}`,
  `decode_ms=${scenario.decode_ms}`,
  `total_ms=${scenario.total_ms}`,
  `ttft_ms=${scenario.ttft_ms}`,
  "peak_memory_mb=0",
];
fs.writeFileSync(outputPath, `${lines.join("\n")}\n`);
