#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

run_and_check() {
    local name="$1"
    local command="$2"
    shift 2

    echo "[regression] ${name}"
    bash -c "${command}" >"${tmp_dir}/${name}.log" 2>&1
    for expected in "$@"; do
        if ! grep -Fq "${expected}" "${tmp_dir}/${name}.log"; then
            echo "[regression] missing '${expected}' in ${name}" >&2
            cat "${tmp_dir}/${name}.log" >&2
            exit 1
        fi
    done
}

run_and_check \
    foundation \
    "cd '${repo_root}' && bash exercises/foundation/run_all.sh" \
    "100% tests passed, 0 tests failed out of 11"

run_and_check \
    chapter05 \
    "cd '${repo_root}' && bash code/chapter05_generation/scripts/run.sh" \
    "100% tests passed, 0 tests failed out of 5" \
    "request_a finished: true" \
    "request_b finished: true"

run_and_check \
    chapter07 \
    "cd '${repo_root}' && bash code/chapter07_benchmark/scripts/run.sh" \
    "unit checks: true" \
    "e2e checks: true" \
    "100% tests passed, 0 tests failed out of 1"

echo "[regression] baseline passed"
