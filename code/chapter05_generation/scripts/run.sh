#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
# Keep this tutorial build isolated from caches created in older workspace paths.
build_dir="${source_dir}/build-cpp-llm-engine"

cmake -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
"${build_dir}/chapter05_generation_demo"
ctest --test-dir "${build_dir}" --output-on-failure
"${build_dir}/chapter05_streaming_demo"
