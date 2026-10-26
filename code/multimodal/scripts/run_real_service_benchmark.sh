#!/usr/bin/env bash
# 多模态 CPU：构建并重复采集真实 SmolVLM 动态图片服务 benchmark。
# 口径：每轮是独立进程；阶段计时不含模型加载，wall time 包含完整进程成本。
# 验证：每轮先通过 sync/SSE token、服务计数和资源释放闸门，再写 JSON。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${source_dir}/../.." && pwd)"
build_dir="${MULTIMODAL_SERVICE_BENCHMARK_BUILD_DIR:-/tmp/cpp-llm-engine-mm-service-benchmark}"
model_path="${SMOLVLM_MODEL_PATH:-${repo_dir}/llama.cpp/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf}"
mmproj_path="${SMOLVLM_MMPROJ_PATH:-${repo_dir}/llama.cpp/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf}"
image_path="${SMOLVLM_IMAGE_PATH:-${repo_dir}/llama.cpp/tools/mtmd/test-1.jpeg}"
llama_cpp_root="${LLAMA_CPP_ROOT:-${repo_dir}/llama.cpp}"
llama_cpp_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_cpp_root}/build}"
output_path="${MULTIMODAL_SERVICE_BENCHMARK_OUTPUT:-${source_dir}/results/smolvlm_service_cpu_benchmark.json}"
warmup="${MULTIMODAL_SERVICE_BENCHMARK_WARMUP:-0}"
iterations="${MULTIMODAL_SERVICE_BENCHMARK_ITERATIONS:-3}"

cmake_args=(
    -S "${source_dir}"
    -B "${build_dir}"
    -DCMAKE_BUILD_TYPE=Release
    -DMULTIMODAL_ENABLE_CBLAS=ON
    -DMULTIMODAL_ENABLE_LLAMA_CPP_TOKENIZER=ON
    -DMULTIMODAL_ENABLE_SERVICE_INTEGRATION=ON
    -DLLAMA_CPP_ROOT="${llama_cpp_root}"
    -DLLAMA_CPP_BUILD_DIR="${llama_cpp_build_dir}"
)
if [[ -n "${MULTIMODAL_CBLAS_LIBRARY:-}" ]]; then
    cmake_args+=("-DMULTIMODAL_CBLAS_LIBRARY=${MULTIMODAL_CBLAS_LIBRARY}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --target multimodal_vision_projector_inspect \
    -j "${BUILD_JOBS:-2}"

python3 "${source_dir}/reference/benchmark_smolvlm_service.py" \
    "${build_dir}/multimodal_vision_projector_inspect" \
    "${mmproj_path}" \
    "${model_path}" \
    "${image_path}" \
    --warmup "${warmup}" \
    --iterations "${iterations}" \
    --output "${output_path}"
