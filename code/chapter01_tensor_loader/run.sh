#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${CHAPTER01_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-chapter01.XXXXXX)}"

cmake -S "${script_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
ctest --test-dir "${build_dir}" --output-on-failure
