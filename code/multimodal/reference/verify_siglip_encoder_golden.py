#!/usr/bin/env python3
"""核对 C++ 与 NumPy 的 SmolVLM 12 层 SigLIP Encoder golden。"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

MULTIMODAL_DIR = Path(__file__).resolve().parents[1]
FIXTURE = MULTIMODAL_DIR / "tests" / "fixtures" / "two_by_two.ppm"
GOLDEN = MULTIMODAL_DIR / "results" / "siglip_encoder_q8_0_golden.json"
REFERENCE = Path(__file__).with_name("siglip_block_reference.py")
LAYER_PATTERN = re.compile(
    r"^layer=(\d+) checksum=([-+0-9.eE]+) first=([-+0-9.eE]+)(?: weight_bytes=(\d+))?$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def parse(output: str) -> dict:
    result = {"layer_checksums": [], "layer_first_values": []}
    for line in output.splitlines():
        match = LAYER_PATTERN.match(line)
        if match:
            expected_index = len(result["layer_checksums"])
            if int(match.group(1)) != expected_index:
                raise RuntimeError("encoder layer output is not contiguous")
            result["layer_checksums"].append(float(match.group(2)))
            result["layer_first_values"].append(float(match.group(3)))
            if match.group(4) is not None:
                result.setdefault("weight_bytes", []).append(int(match.group(4)))
        elif line.startswith("post_norm_checksum="):
            result["post_norm_checksum"] = float(line.split("=", 1)[1])
        elif line.startswith("first="):
            result["first"] = float(line.split("=", 1)[1])
        elif line.startswith("peak_block_weight_bytes="):
            result["peak_block_weight_bytes"] = int(line.split("=", 1)[1])
    return result


def compare(actual: dict, expected: dict, golden: dict) -> None:
    checksum_tolerance = golden["checksum_absolute_tolerance"]
    value_tolerance = golden["value_absolute_tolerance"]
    if len(actual["layer_checksums"]) != golden["block_count"]:
        raise RuntimeError("encoder output does not contain all layers")
    for index, expected_value in enumerate(expected["layer_checksums"]):
        if abs(actual["layer_checksums"][index] - expected_value) > checksum_tolerance:
            raise RuntimeError(f"layer {index} checksum mismatch")
    for index, expected_value in enumerate(expected["layer_first_values"]):
        if abs(actual["layer_first_values"][index] - expected_value) > value_tolerance:
            raise RuntimeError(f"layer {index} first value mismatch")
    if abs(actual["post_norm_checksum"] - expected["post_norm_checksum"]) > checksum_tolerance:
        raise RuntimeError("post norm checksum mismatch")
    if abs(actual["first"] - expected["first"]) > value_tolerance:
        raise RuntimeError("post norm first value mismatch")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_siglip_encoder_golden.py <vision-encoder-inspect> <mmproj.gguf>",
              file=sys.stderr)
        return 2
    executable = Path(sys.argv[1]).resolve()
    mmproj = Path(sys.argv[2]).resolve()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if sha256(mmproj) != golden["model_sha256"] or sha256(FIXTURE) != golden["input_sha256"]:
        raise RuntimeError("model or fixture SHA256 does not match the encoder golden")

    cpp_output = run([str(executable), str(mmproj), str(FIXTURE), str(golden["patch_count"])])
    reference_output = run([
        sys.executable, str(REFERENCE), str(mmproj), str(FIXTURE),
        str(golden["patch_count"]), str(golden["block_count"]),
    ])
    cpp_values = parse(cpp_output)
    reference_values = parse(reference_output)
    compare(cpp_values, golden["cpp"], golden)
    compare(reference_values, golden["numpy_reference"], golden)
    compare(cpp_values, reference_values, golden)
    if cpp_values.get("peak_block_weight_bytes") != golden["decoded_single_block_weight_bytes"]:
        raise RuntimeError("decoded single-block weight footprint changed")
    if any(value != golden["decoded_single_block_weight_bytes"]
           for value in cpp_values.get("weight_bytes", [])):
        raise RuntimeError("one or more layers retained an unexpected weight footprint")
    print(cpp_output, end="")
    print(reference_output, end="")
    print("MM-P5A golden comparison passed: 12 layers, post norm, hashes, and weight footprint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
