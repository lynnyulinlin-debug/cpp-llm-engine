#!/usr/bin/env python3
"""核对动态 tiles、448 个视觉行和 486-position Decoder golden。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

MULTIMODAL_DIR = Path(__file__).resolve().parents[1]
GOLDEN = MULTIMODAL_DIR / "results" / "smolvlm_dynamic_tiles_q8_0_golden.json"
REFERENCE = Path(__file__).with_name("siglip_projector_reference.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def parse(output: str) -> dict[str, object]:
    result: dict[str, object] = {}
    floats = {
        "encoder_checksum", "merged_checksum", "merged_first", "output_checksum", "first",
        "text_embedding_checksum", "sequence_checksum", "logits_checksum", "top1_logit",
    }
    integers = {
        "visual_count", "language_hidden", "prompt_prefix_tokens", "prompt_suffix_tokens",
        "prompt_visual_count", "prompt_visual_rows", "decoder_layers", "decoder_tokens",
        "logits_size", "top1_token_id",
    }
    strings = {
        "source_image", "tile_grid", "vision_mode", "selected_positions", "encoder_shape",
        "merged_shape", "output_shape", "language_architecture", "text_token_ids",
        "sequence_shape", "prompt_mode", "prompt_token_ids",
    }
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in floats:
            result[key] = float(value)
        elif key in integers:
            result[key] = int(value)
        elif key in strings:
            result[key] = value
        elif key == "visual_output_checksums":
            result[key] = [float(item) for item in value.split(",")]
    return result


def compare_to_golden(actual: dict, expected: dict, golden: dict) -> None:
    exact_keys = (
        "source_image", "tile_grid", "visual_count", "vision_mode", "selected_positions",
        "encoder_shape", "merged_shape", "output_shape", "language_architecture",
        "language_hidden", "text_token_ids", "sequence_shape", "prompt_mode",
        "prompt_token_ids", "prompt_prefix_tokens", "prompt_suffix_tokens",
        "prompt_visual_count", "prompt_visual_rows", "decoder_layers", "decoder_tokens",
        "logits_size", "top1_token_id",
    )
    for key in exact_keys:
        if actual.get(key) != golden[key]:
            raise RuntimeError(f"{key} mismatch: {actual.get(key)!r} != {golden[key]!r}")
    for key in ("encoder_checksum", "merged_checksum", "output_checksum",
                "text_embedding_checksum", "sequence_checksum"):
        if abs(float(actual[key]) - float(expected[key])) > golden["checksum_absolute_tolerance"]:
            raise RuntimeError(f"{key} mismatch")
    for key in ("merged_first", "first"):
        if abs(float(actual[key]) - float(expected[key])) > golden["value_absolute_tolerance"]:
            raise RuntimeError(f"{key} mismatch")
    actual_visual = actual["visual_output_checksums"]
    expected_visual = expected["visual_output_checksums"]
    if len(actual_visual) != len(expected_visual):
        raise RuntimeError("visual_output_checksums length mismatch")
    for index, (actual_value, expected_value) in enumerate(zip(actual_visual, expected_visual)):
        if abs(float(actual_value) - float(expected_value)) > golden["checksum_absolute_tolerance"]:
            raise RuntimeError(f"visual_output_checksums[{index}] mismatch")
    if (abs(float(actual["logits_checksum"]) - float(expected["logits_checksum"])) >
            golden["logits_checksum_absolute_tolerance"]):
        raise RuntimeError("logits_checksum mismatch")
    if abs(float(actual["top1_logit"]) - float(expected["top1_logit"])) > golden["logit_absolute_tolerance"]:
        raise RuntimeError("top1_logit mismatch")


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: verify_smolvlm_dynamic_tiles_golden.py "
              "<openblas-tokenizer-projector-inspect> <mmproj.gguf> <model.gguf> <image>",
              file=sys.stderr)
        return 2
    executable, mmproj, model, image = map(lambda value: Path(value).resolve(), sys.argv[1:])
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if (sha256(model) != golden["model_sha256"] or
            sha256(mmproj) != golden["mmproj_sha256"] or
            sha256(image) != golden["input_sha256"]):
        raise RuntimeError("model, mmproj, or image SHA256 does not match the dynamic golden")

    layers = str(golden["decoder_layers"])
    cpp_output = run([
        str(executable), str(mmproj), str(image), str(model), layers,
        golden["question"], "--dynamic-tiles",
    ])
    numpy_output = run([
        sys.executable, str(REFERENCE), str(mmproj), str(image), str(model), layers,
        golden["prompt_token_ids"], "--dynamic-tiles",
    ])
    cpp_values = parse(cpp_output)
    numpy_values = parse(numpy_output)
    compare_to_golden(cpp_values, golden["cpp_openblas"], golden)
    compare_to_golden(numpy_values, golden["numpy_reference"], golden)
    print(cpp_output, end="")
    print(numpy_output, end="")
    print("MM-P6D golden comparison passed: 7 visual chunks and 486 decoder positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
