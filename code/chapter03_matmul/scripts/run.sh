#!/usr/bin/env bash
# Chapter 03：MatMul CPU 验证脚本
# 执行：隔离配置、构建 demo，并运行 MatMul 与 core CTest。
# 参数：CHAPTER03_BUILD_DIR 可指定持久 build 目录。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
cleanup_build=false
if [[ -n "${CHAPTER03_BUILD_DIR:-}" ]]; then
    build_dir="${CHAPTER03_BUILD_DIR}"
else
    build_dir="$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter03.XXXXXX")"
    cleanup_build=true
fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then
        rm -rf "${build_dir}"
    fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
"${build_dir}/chapter03_matmul_demo"
ctest --test-dir "${build_dir}" --output-on-failure
