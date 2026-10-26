#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${ENGINE_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-engine.XXXXXX")}"
cleanup_build=false
if [[ -z "${ENGINE_BUILD_DIR:-}" ]]; then cleanup_build=true; fi

cleanup() {
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

cmake -S "${source_dir}" -B "${build_dir}"
cmake --build "${build_dir}"
"${build_dir}/cpp_llm_engine_demo" --prompt "hello" --max-new-tokens 4
ctest --test-dir "${build_dir}" --output-on-failure
