#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${MULTIMODAL_BUILD_DIR:-$(mktemp -d /tmp/cpp-llm-engine-multimodal.XXXXXX)}"

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -j "${BUILD_JOBS:-4}"
ctest --test-dir "${build_dir}" --output-on-failure
"${build_dir}/multimodal_image_inspect" "${source_dir}/tests/fixtures/two_by_two.ppm"

if [[ -n "${MMPROJ_PATH:-}" ]]; then
    "${build_dir}/multimodal_mmproj_inspect" "${MMPROJ_PATH}"
fi

if [[ -n "${MMPROJ_PATH:-}" && -n "${MULTIMODAL_IMAGE_PATH:-}" ]]; then
    "${build_dir}/multimodal_vision_embedding_inspect" \
        "${MMPROJ_PATH}" "${MULTIMODAL_IMAGE_PATH}"
    "${build_dir}/multimodal_vision_block_inspect" \
        "${MMPROJ_PATH}" "${MULTIMODAL_IMAGE_PATH}" "${MULTIMODAL_BLOCK_PATCHES:-4}"
    "${build_dir}/multimodal_vision_encoder_inspect" \
        "${MMPROJ_PATH}" "${MULTIMODAL_IMAGE_PATH}" "${MULTIMODAL_ENCODER_PATCHES:-4}"
    if [[ -n "${MULTIMODAL_MODEL_PATH:-}" ]]; then
        if [[ -n "${MULTIMODAL_QUESTION:-}" ]]; then
            "${build_dir}/multimodal_vision_projector_inspect" \
                "${MMPROJ_PATH}" "${MULTIMODAL_IMAGE_PATH}" "${MULTIMODAL_MODEL_PATH}" \
                "${MULTIMODAL_DECODER_LAYERS:-1}" "${MULTIMODAL_QUESTION}"
        else
            "${build_dir}/multimodal_vision_projector_inspect" \
                "${MMPROJ_PATH}" "${MULTIMODAL_IMAGE_PATH}" "${MULTIMODAL_MODEL_PATH}" \
                "${MULTIMODAL_DECODER_LAYERS:-1}"
        fi
    else
        "${build_dir}/multimodal_vision_projector_inspect" \
            "${MMPROJ_PATH}" "${MULTIMODAL_IMAGE_PATH}"
    fi
fi
