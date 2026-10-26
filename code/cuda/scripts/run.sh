#!/usr/bin/env bash
# CUDA 基础层：使用指定 Toolkit 构建并在真实 GPU 上运行 CTest。
# 默认选择 /usr/local/cuda-12.8；CUDA_HOME 和 CUDA_BUILD_DIR 可覆盖。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_dir="${CUDA_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-cuda.XXXXXX")}"
cleanup_build=false
if [[ -z "${CUDA_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
trap 'if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi' EXIT

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "nvcc not found: ${cuda_home}/bin/nvcc" >&2
    echo "set CUDA_HOME to a compatible CUDA Toolkit" >&2
    exit 2
fi

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -j2
ctest --test-dir "${build_dir}" --output-on-failure
