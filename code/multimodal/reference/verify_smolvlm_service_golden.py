#!/usr/bin/env python3
"""核对真实 SmolVLM embedding 进入同步与 SSE 服务的固定结果。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

MULTIMODAL_DIR = Path(__file__).resolve().parents[1]
GOLDEN = MULTIMODAL_DIR / "results" / "smolvlm_dynamic_tiles_q8_0_golden.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: verify_smolvlm_service_golden.py "
            "<service-projector-inspect> <mmproj.gguf> <model.gguf> <image>",
            file=sys.stderr,
        )
        return 2

    executable, mmproj, model, image = map(
        lambda value: Path(value).resolve(), sys.argv[1:]
    )
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if (
        sha256(model) != golden["model_sha256"]
        or sha256(mmproj) != golden["mmproj_sha256"]
        or sha256(image) != golden["input_sha256"]
    ):
        raise RuntimeError("model, mmproj, or image SHA256 does not match the service golden")

    completed = subprocess.run(
        [
            str(executable),
            str(mmproj),
            str(image),
            str(model),
            str(golden["decoder_layers"]),
            golden["question"],
            "--dynamic-tiles-service",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    values = parse(completed.stdout)
    service = golden["service_smoke"]
    expected = {
        "tile_grid": golden["tile_grid"],
        "visual_count": str(golden["visual_count"]),
        "sequence_shape": golden["sequence_shape"],
        "service_sync_status": "200",
        "service_sync_token_match": "true",
        "service_sse_status": "200",
        "service_sse_token_match": "true",
        "service_factory_calls": str(service["request_count"]),
        "service_accepted": str(service["accepted"]),
        "service_completed": str(service["completed"]),
        "service_released": str(service["released"]),
        "service_generated_tokens": str(service["generated_tokens"]),
        "service_inflight": str(service["inflight_after_release"]),
        "multimodal_service": "valid",
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise RuntimeError(
                f"{key} mismatch: {values.get(key)!r} != {expected_value!r}"
            )
    print(completed.stdout, end="")
    print("MM-P7B-2 service golden passed: sync/SSE token 17600 and resources released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
