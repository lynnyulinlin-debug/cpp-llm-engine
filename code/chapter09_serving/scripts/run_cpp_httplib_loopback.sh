#!/usr/bin/env bash
# Chapter 09：构建 cpp-httplib transport 并执行本地 HTTP/SSE 闭环。
# 验证 health、metrics、同步 completion 和 streaming event，不改变 ServiceCore 契约。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${CHAPTER09_HTTP_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/cpp-llm-engine-chapter09-http.XXXXXX")}"
cleanup_build=false
if [[ -z "${CHAPTER09_HTTP_BUILD_DIR:-}" ]]; then cleanup_build=true; fi
port="${CHAPTER09_HTTP_PORT:-18081}"
include_dir="${CHAPTER09_CPPHTTPLIB_INCLUDE_DIR:-}"
server_pid=""

cleanup() {
    if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; fi
    if [[ "${cleanup_build}" == true ]]; then rm -rf "${build_dir}"; fi
}
trap cleanup EXIT

cmake_args=(-DCHAPTER09_ENABLE_CPPHTTPLIB=ON)
if [[ -n "${include_dir}" ]]; then
    cmake_args+=("-DCHAPTER09_CPPHTTPLIB_INCLUDE_DIR=${include_dir}")
fi
cmake -S "${source_dir}" -B "${build_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}"

"${build_dir}/chapter09_cpp_httplib_server" "${port}" &
server_pid=$!
sleep 1

health="$(curl --fail --silent --show-error --max-time 10 "http://127.0.0.1:${port}/health")"
[[ "${health}" == '{"status":"ok"}' ]]

metrics="$(curl --fail --silent --show-error --max-time 10 "http://127.0.0.1:${port}/metrics")"
grep -q '^serving_accepted_total 0$' <<<"${metrics}"

completion="$(curl --fail --silent --show-error --max-time 10 \
    -H 'Content-Type: application/json' \
    -d '{"request_id":21,"prompt_tokens":[1,2]}' \
    "http://127.0.0.1:${port}/v1/completions")"
grep -q '"finish_reason":"stop"' <<<"${completion}"

stream="$(curl --fail --silent --show-error --max-time 10 -N \
    -H 'Content-Type: application/json' \
    -d '{"request_id":22,"prompt_tokens":[1,2],"stream":true}' \
    "http://127.0.0.1:${port}/v1/completions")"
grep -q 'data: {"token":' <<<"${stream}"
grep -q 'finish_reason":"stop' <<<"${stream}"

echo "cpp-httplib loopback passed: health, metrics, completion, streaming"
