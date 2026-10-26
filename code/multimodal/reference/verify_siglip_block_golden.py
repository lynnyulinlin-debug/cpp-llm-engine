#!/usr/bin/env python3
"""核对 C++ 与 NumPy 的 SmolVLM 第 0 个 SigLIP Block golden。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

MULTIMODAL_DIR = Path(__file__).resolve().parents[1]
FIXTURE = MULTIMODAL_DIR / "tests" / "fixtures" / "two_by_two.ppm"
GOLDEN = MULTIMODAL_DIR / "results" / "siglip_block_q8_0_golden.json"
REFERENCE = Path(__file__).with_name("siglip_block_reference.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def parse(output: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        try:
            values[key] = float(raw)
        except ValueError:
            pass
    return values


def compare(actual: dict[str, float], expected: dict[str, float], golden: dict) -> None:
    for key, expected_value in expected.items():
        tolerance = (golden["value_absolute_tolerance"]
                     if key in ("first", "embedding_first")
                     else golden["checksum_absolute_tolerance"])
        if key not in actual or abs(actual[key] - expected_value) > tolerance:
            raise RuntimeError(
                f"golden mismatch for {key}: actual={actual.get(key)} "
                f"expected={expected_value} tolerance={tolerance}"
            )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_siglip_block_golden.py <vision-block-inspect> <mmproj.gguf>",
              file=sys.stderr)
        return 2
    executable = Path(sys.argv[1]).resolve()
    mmproj = Path(sys.argv[2]).resolve()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if sha256(mmproj) != golden["model_sha256"]:
        raise RuntimeError("mmproj SHA256 does not match the recorded golden")
    if sha256(FIXTURE) != golden["input_sha256"]:
        raise RuntimeError("PPM fixture SHA256 does not match the recorded golden")

    cpp_output = run([str(executable), str(mmproj), str(FIXTURE), str(golden["patch_count"])])
    reference_output = run([
        sys.executable, str(REFERENCE), str(mmproj), str(FIXTURE),
        str(golden["patch_count"]),
    ])
    cpp_values = parse(cpp_output)
    reference_values = parse(reference_output)
    compare(cpp_values, golden["cpp"], golden)
    compare(reference_values, golden["numpy_reference"], golden)
    compare(cpp_values, reference_values, golden)
    print(cpp_output, end="")
    print(reference_output, end="")
    print("MM-P4B golden comparison passed: hashes and 10 numeric fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
