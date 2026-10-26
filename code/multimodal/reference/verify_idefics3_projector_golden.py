#!/usr/bin/env python3
"""核对 C++ 与 NumPy 的 SmolVLM IDEFICS3 Projector golden。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

MULTIMODAL_DIR = Path(__file__).resolve().parents[1]
FIXTURE = MULTIMODAL_DIR / "tests" / "fixtures" / "two_by_two.ppm"
GOLDEN = MULTIMODAL_DIR / "results" / "idefics3_projector_q8_0_golden.json"
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
        "encoder_checksum", "merged_checksum", "merged_first", "output_checksum", "first"
    }
    integer_keys = {"projector_weight_bytes"}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in float_keys:
            result[key] = float(value)
        elif key in integer_keys:
            result[key] = int(value)
        elif key in {"selected_positions", "encoder_shape", "merged_shape", "output_shape"}:
            result[key] = value
    return result


def compare(actual: dict, expected: dict, golden: dict) -> None:
    for key in ("selected_positions", "encoder_shape", "merged_shape", "output_shape"):
        if actual.get(key) != golden[key]:
            raise RuntimeError(f"{key} mismatch")
    for key in ("encoder_checksum", "merged_checksum", "output_checksum"):
        if abs(float(actual[key]) - float(expected[key])) > golden["checksum_absolute_tolerance"]:
            raise RuntimeError(f"{key} mismatch")
    for key in ("merged_first", "first"):
        if abs(float(actual[key]) - float(expected[key])) > golden["value_absolute_tolerance"]:
            raise RuntimeError(f"{key} mismatch")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_idefics3_projector_golden.py <projector-inspect> <mmproj.gguf>",
              file=sys.stderr)
        return 2
    executable = Path(sys.argv[1]).resolve()
    mmproj = Path(sys.argv[2]).resolve()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if sha256(mmproj) != golden["model_sha256"] or sha256(FIXTURE) != golden["input_sha256"]:
        raise RuntimeError("model or fixture SHA256 does not match the projector golden")

    cpp_output = run([str(executable), str(mmproj), str(FIXTURE)])
    reference_output = run([sys.executable, str(REFERENCE), str(mmproj), str(FIXTURE)])
    cpp_values = parse(cpp_output)
    reference_values = parse(reference_output)
    compare(cpp_values, golden["cpp"], golden)
    compare(reference_values, golden["numpy_reference"], golden)
    compare(cpp_values, reference_values, golden)
    if cpp_values.get("projector_weight_bytes") != golden["decoded_projector_weight_bytes"]:
        raise RuntimeError("decoded projector weight footprint changed")
    print(cpp_output, end="")
    print(reference_output, end="")
    print("MM-P5B golden comparison passed: spatial order, projector values, hashes, and shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
