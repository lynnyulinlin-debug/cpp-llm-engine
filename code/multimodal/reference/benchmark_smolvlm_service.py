#!/usr/bin/env python3
"""重复采集真实 SmolVLM 服务阶段数据，并在正确性通过后输出 JSON。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

MULTIMODAL_DIR = Path(__file__).resolve().parents[1]
GOLDEN = MULTIMODAL_DIR / "results" / "smolvlm_dynamic_tiles_q8_0_golden.json"
STAGES = (
    "image_decode",
    "dynamic_tiling",
    "prompt_tokenization",
    "vision_encoder",
    "projector",
    "embedding_composition",
    "language_prefill",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def summarize(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("benchmark summary requires finite non-negative samples")
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "min": min(values),
        "mean": mean,
        "p50": statistics.median(values),
        "max": max(values),
        "cv_percent": 0.0 if mean == 0.0 else deviation / mean * 100.0,
    }


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def linked_blas(executable: Path) -> str:
    completed = subprocess.run(
        ["ldd", str(executable)], text=True, capture_output=True, check=False
    )
    for line in completed.stdout.splitlines():
        if "blas" in line.lower():
            return line.strip()
    return "not-reported"


def validate_output(values: dict[str, str], golden: dict[str, Any]) -> None:
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
                f"correctness gate failed for {key}: "
                f"{values.get(key)!r} != {expected_value!r}"
            )

    expected_counts = {
        "image_decode": 1,
        "dynamic_tiling": 1,
        "prompt_tokenization": 1,
        "vision_encoder": golden["visual_count"],
        "projector": golden["visual_count"],
        "embedding_composition": 1,
        "language_prefill": 1,
    }
    for stage, expected_count in expected_counts.items():
        key = f"stage_{stage}_count"
        if int(values.get(key, "-1")) != expected_count:
            raise RuntimeError(f"correctness gate failed for {key}")


def to_sample(values: dict[str, str], wall_seconds: float) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        stages[stage] = {
            "seconds": float(values[f"stage_{stage}_seconds"]),
            "invocations": int(values[f"stage_{stage}_count"]),
            "peak_live_bytes": int(values[f"stage_{stage}_peak_live_bytes"]),
        }
    return {
        "wall_seconds": wall_seconds,
        "input_pipeline_seconds": float(values["input_pipeline_seconds"]),
        "request_peak_live_bytes": int(values["request_peak_live_bytes"]),
        "stages": stages,
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one measured sample is required")
    stage_summary = {}
    for stage in STAGES:
        stage_summary[stage] = {
            "seconds": summarize(
                [sample["stages"][stage]["seconds"] for sample in samples]
            ),
            "invocations_per_iteration": samples[0]["stages"][stage]["invocations"],
            "peak_live_bytes": max(
                sample["stages"][stage]["peak_live_bytes"] for sample in samples
            ),
        }
    return {
        "wall_seconds": summarize([sample["wall_seconds"] for sample in samples]),
        "input_pipeline_seconds": summarize(
            [sample["input_pipeline_seconds"] for sample in samples]
        ),
        "request_peak_live_bytes": max(
            sample["request_peak_live_bytes"] for sample in samples
        ),
        "stages": stage_summary,
    }


def self_test() -> int:
    samples = []
    for scale in (1.0, 2.0, 3.0):
        samples.append(
            {
                "wall_seconds": 10.0 * scale,
                "input_pipeline_seconds": 8.0 * scale,
                "request_peak_live_bytes": 4096,
                "stages": {
                    stage: {
                        "seconds": scale,
                        "invocations": 7 if stage in ("vision_encoder", "projector") else 1,
                        "peak_live_bytes": 1024,
                    }
                    for stage in STAGES
                },
            }
        )
    result = aggregate(samples)
    assert result["wall_seconds"]["mean"] == 20.0
    assert result["input_pipeline_seconds"]["p50"] == 16.0
    assert result["stages"]["vision_encoder"]["invocations_per_iteration"] == 7
    assert result["request_peak_live_bytes"] == 4096
    print("SmolVLM service benchmark collector self-test passed")
    return 0


def validate_result(path: Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    samples = document.get("samples", [])
    if (
        document.get("schema_version") != 1
        or document.get("benchmark") != "smolvlm_dynamic_tiles_service_cpu"
        or document.get("iterations") != len(samples)
        or len(samples) < 2
        or document.get("summary") != aggregate(samples)
    ):
        raise RuntimeError("benchmark result structure or summary is invalid")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    expected_assets = {
        "model_sha256": golden["model_sha256"],
        "mmproj_sha256": golden["mmproj_sha256"],
        "image_sha256": golden["input_sha256"],
    }
    correctness = document.get("correctness", {})
    if (
        document.get("assets") != expected_assets
        or correctness.get("all_iterations") is not True
        or correctness.get("sync_and_sse_token")
        != golden["service_smoke"]["generated_token_ids"]
        or correctness.get("terminal_resources_released") is not True
        or correctness.get("inflight_after_release") != 0
    ):
        raise RuntimeError("benchmark result assets or correctness gate is invalid")
    print(
        f"SmolVLM service benchmark result passed: {len(samples)} iterations, "
        f"pipeline CV {document['summary']['input_pipeline_seconds']['cv_percent']:.6f}%"
    )
    return 0


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", nargs="?", type=Path)
    parser.add_argument("mmproj", nargs="?", type=Path)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--warmup", type=non_negative_integer, default=0)
    parser.add_argument("--iterations", type=positive_integer, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-result", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.validate_result is not None:
        return validate_result(args.validate_result)
    if None in (args.executable, args.mmproj, args.model, args.image) or args.output is None:
        parser.error("executable, mmproj, model, image and --output are required")

    executable = args.executable.resolve()
    mmproj = args.mmproj.resolve()
    model = args.model.resolve()
    image = args.image.resolve()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    hashes = {
        "model_sha256": sha256(model),
        "mmproj_sha256": sha256(mmproj),
        "image_sha256": sha256(image),
    }
    expected_hashes = {
        "model_sha256": golden["model_sha256"],
        "mmproj_sha256": golden["mmproj_sha256"],
        "image_sha256": golden["input_sha256"],
    }
    if hashes != expected_hashes:
        raise RuntimeError("model, mmproj, or image SHA256 does not match the golden")

    command = [
        str(executable),
        str(mmproj),
        str(image),
        str(model),
        str(golden["decoder_layers"]),
        golden["question"],
        "--dynamic-tiles-service",
    ]
    samples: list[dict[str, Any]] = []
    for index in range(args.warmup + args.iterations):
        started_at = time.perf_counter()
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        wall_seconds = time.perf_counter() - started_at
        values = parse_key_values(completed.stdout)
        validate_output(values, golden)
        if index >= args.warmup:
            samples.append(to_sample(values, wall_seconds))
        phase = "warmup" if index < args.warmup else "iteration"
        phase_index = index + 1 if phase == "warmup" else index - args.warmup + 1
        print(
            f"{phase}={phase_index} wall_seconds={wall_seconds:.6f} "
            f"pipeline_seconds={float(values['input_pipeline_seconds']):.6f}",
            flush=True,
        )

    document = {
        "schema_version": 1,
        "benchmark": "smolvlm_dynamic_tiles_service_cpu",
        "measurement_scope": "repeated-cold-process; model loading excluded from stage timers",
        "workload": {
            "question": golden["question"],
            "decoder_layers": golden["decoder_layers"],
            "tile_grid": golden["tile_grid"],
            "visual_inputs": golden["visual_count"],
            "visual_rows": golden["prompt_visual_rows"],
            "sequence_shape": golden["sequence_shape"],
            "requests_per_iteration": golden["service_smoke"]["request_count"],
            "generated_tokens_per_iteration": golden["service_smoke"]["generated_tokens"],
        },
        "assets": hashes,
        "environment": {
            "cpu_model": cpu_model(),
            "hardware_concurrency": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "linked_blas": linked_blas(executable),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "library-default"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "library-default"),
            "build_type": "Release",
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples": samples,
        "summary": aggregate(samples),
        "correctness": {
            "all_iterations": True,
            "sync_and_sse_token": golden["service_smoke"]["generated_token_ids"],
            "terminal_resources_released": True,
            "inflight_after_release": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = document["summary"]
    print(
        f"result={args.output}\n"
        f"wall_mean_seconds={summary['wall_seconds']['mean']:.6f}\n"
        f"pipeline_mean_seconds={summary['input_pipeline_seconds']['mean']:.6f}\n"
        f"pipeline_cv_percent={summary['input_pipeline_seconds']['cv_percent']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
