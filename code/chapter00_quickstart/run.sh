#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep the smoke check independent from build caches copied from older paths.
build_dir="${CHAPTER00_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-chapter00.XXXXXX)}"

cmake -S "${script_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
"${build_dir}/chapter00_quickstart" --check
