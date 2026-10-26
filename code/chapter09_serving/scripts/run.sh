#!/usr/bin/env bash
# Chapter 09：构建并运行最小 C++ Service Core。
# 验证 admission、batch dispatch、取消/超时和完成后 cache 回收。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${CHAPTER09_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter09.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER09_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

cmake -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
"${build_dir}/chapter09_serving_demo"
ctest --test-dir "${build_dir}" --output-on-failure

if [[ "${CHAPTER09_ENABLE_CPPHTTPLIB:-OFF}" == "ON" ]]; then
    echo "cpp-httplib transport is enabled; start with: ${build_dir}/chapter09_cpp_httplib_server 18080"
fi
