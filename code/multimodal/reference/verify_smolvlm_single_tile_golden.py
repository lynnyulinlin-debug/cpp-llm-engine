#!/usr/bin/env python3
"""核对完整单 tile 的视觉中间值、真实 prompt 与 32 层 Decoder golden。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

MULTIMODAL_DIR = Path(__file__).resolve().parents[1]
FIXTURE = MULTIMODAL_DIR / "tests" / "fixtures" / "two_by_two.ppm"
GOLDEN = MULTIMODAL_DIR / "results" / "smolvlm_single_tile_q8_0_golden.json"
REFERENCE = Path(__file__).with_name("siglip_projector_reference.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def parse(output: str) -> dict[str, str | float | int]:
    result: dict[str, str | float | int] = {}
    float_keys = {
        "encoder_checksum", "merged_checksum", "merged_first", "output_checksum", "first",
        "text_embedding_checksum", "sequence_checksum", "logits_checksum", "top1_logit",
    }
    int_keys = {
        "language_hidden", "prompt_prefix_tokens", "prompt_suffix_tokens",
        "prompt_visual_rows", "decoder_layers", "decoder_tokens", "logits_size",
        "top1_token_id",
    }
    string_keys = {
        "vision_mode", "selected_positions", "encoder_shape", "merged_shape", "output_shape",
        "language_architecture", "text_token_ids", "sequence_shape", "prompt_mode",
        "prompt_token_ids",
    }
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in float_keys:
            result[key] = float(value)
        elif key in int_keys:
            result[key] = int(value)
        elif key in string_keys:
            result[key] = value
    return result


def compare(actual: dict, expected: dict, golden: dict) -> None:
    exact_keys = (
        "vision_mode", "selected_positions", "encoder_shape", "merged_shape", "output_shape",
        "language_architecture", "language_hidden", "text_token_ids", "sequence_shape",
        "prompt_mode", "prompt_token_ids", "prompt_prefix_tokens", "prompt_suffix_tokens",
        "prompt_visual_rows", "decoder_layers", "decoder_tokens", "logits_size", "top1_token_id",
    )
    for key in exact_keys:
        if actual.get(key) != golden[key]:
            raise RuntimeError(f"{key} mismatch: {actual.get(key)!r} != {golden[key]!r}")
    for key in ("encoder_checksum", "merged_checksum", "output_checksum"):
        if abs(float(actual[key]) - float(expected[key])) > golden["vision_checksum_absolute_tolerance"]:
            raise RuntimeError(f"{key} mismatch")
    for key in ("merged_first", "first"):
        if abs(float(actual[key]) - float(expected[key])) > golden["vision_value_absolute_tolerance"]:
            raise RuntimeError(f"{key} mismatch")
    for key in ("text_embedding_checksum", "sequence_checksum"):
        if abs(float(actual[key]) - float(expected[key])) > golden["sequence_checksum_absolute_tolerance"]:
            raise RuntimeError(f"{key} mismatch")
    if (abs(float(actual["logits_checksum"]) - float(expected["logits_checksum"])) >
            golden["logits_checksum_absolute_tolerance"]):
        raise RuntimeError("logits_checksum mismatch")
    if abs(float(actual["top1_logit"]) - float(expected["top1_logit"])) > golden["logit_absolute_tolerance"]:
        raise RuntimeError("top1_logit mismatch")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: verify_smolvlm_single_tile_golden.py "
              "<cblas-tokenizer-projector-inspect> <mmproj.gguf> <model.gguf>",
              file=sys.stderr)
        return 2
    executable = Path(sys.argv[1]).resolve()
    mmproj = Path(sys.argv[2]).resolve()
    model = Path(sys.argv[3]).resolve()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if (sha256(model) != golden["model_sha256"] or
            sha256(mmproj) != golden["mmproj_sha256"] or
            sha256(FIXTURE) != golden["input_sha256"]):
        raise RuntimeError("model, mmproj, or fixture SHA256 does not match the single-tile golden")

    layers = str(golden["decoder_layers"])
    cpp_output = run([
        str(executable), str(mmproj), str(FIXTURE), str(model), layers,
        golden["question"], "--full-tile",
    ])
    reference_output = run([
        sys.executable, str(REFERENCE), str(mmproj), str(FIXTURE), str(model), layers,
        golden["prompt_token_ids"], "--full-tile",
    ])
    cpp_values = parse(cpp_output)
    reference_values = parse(reference_output)
    compare(cpp_values, golden["cpp"], golden)
    compare(reference_values, golden["numpy_reference"], golden)
    compare(cpp_values, reference_values, golden)
    print(cpp_output, end="")
    print(reference_output, end="")
    print("MM-P6C-3 golden comparison passed: complete 64-row visual tile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
