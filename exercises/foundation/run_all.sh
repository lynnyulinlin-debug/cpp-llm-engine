#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep exercise verification independent from caches created in another checkout.
build_dir="${FOUNDATION_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-foundation.XXXXXX)}"

cmake -S "${root_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
ctest --test-dir "${build_dir}" --output-on-failure
