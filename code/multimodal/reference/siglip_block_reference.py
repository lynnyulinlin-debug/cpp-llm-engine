#!/usr/bin/env python3
"""SmolVLM 首个 SigLIP Block 的可选 NumPy 参考实现。

该脚本不进入默认回归；它使用 llama.cpp/gguf-py 的独立 Q8_0 解码，核对 C++
真实 payload 路径。运行：python3 siglip_block_reference.py <mmproj.gguf> <image> [patch-count]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "llama.cpp" / "gguf-py"))

from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def resize_bilinear_half_pixel(image: np.ndarray,
                               width: int,
                               height: int | None = None) -> np.ndarray:
    if height is None:
        height = width
    source_height, source_width, _ = image.shape
    scale_x = np.float32(source_width) / np.float32(width)
    scale_y = np.float32(source_height) / np.float32(height)
    output = np.empty((height, width, 3), dtype=np.uint8)
    one = np.float32(1)
    for y in range(height):
        source_y = (np.float32(y) + np.float32(0.5)) * scale_y - np.float32(0.5)
        clamped_y = np.clip(source_y, np.float32(0), np.float32(source_height - 1))
        y0 = int(np.floor(clamped_y))
        y1 = min(y0 + 1, source_height - 1)
        wy = np.float32(clamped_y - np.float32(y0))
        for x in range(width):
            source_x = (np.float32(x) + np.float32(0.5)) * scale_x - np.float32(0.5)
            clamped_x = np.clip(source_x, np.float32(0), np.float32(source_width - 1))
            x0 = int(np.floor(clamped_x))
            x1 = min(x0 + 1, source_width - 1)
            wx = np.float32(clamped_x - np.float32(x0))
            for channel in range(3):
                top = (np.float32(image[y0, x0, channel]) * (one - wx) +
                       np.float32(image[y0, x1, channel]) * wx)
                bottom = (np.float32(image[y1, x0, channel]) * (one - wx) +
                          np.float32(image[y1, x1, channel]) * wx)
                value = np.float32(top * (one - wy) + bottom * wy)
                output[y, x, channel] = np.uint8(np.floor(np.clip(value, 0, 255) + np.float32(0.5)))
    return output


def load_values(tensors: dict[str, object], name: str) -> np.ndarray:
    tensor = tensors[name]
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32)
    return np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    matrix = weight.reshape(weight.shape[0], -1)
    return np.einsum("ni,oi->no", x, matrix, dtype=np.float32, optimize=False) + bias.reshape(1, -1)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, epsilon: np.float32) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True, dtype=np.float32)
    variance = ((x - mean) ** np.float32(2)).mean(axis=1, keepdims=True, dtype=np.float32)
    return (x - mean) / np.sqrt(variance + epsilon) * weight.reshape(1, -1) + bias.reshape(1, -1)


def checksum(values: np.ndarray) -> float:
    flat = values.reshape(-1).astype(np.float64)
    weights = np.arange(flat.size, dtype=np.int64) % 97 + 1
    return float(np.sum(flat * weights, dtype=np.float64))


def run_block(current: np.ndarray,
              tensors: dict[str, object],
              block: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    patch_count = current.shape[0]
    prefix = f"v.blk.{block}."
    norm1 = layer_norm(
        current,
        load_values(tensors, prefix + "ln1.weight"),
        load_values(tensors, prefix + "ln1.bias"),
        np.float32(1e-6),
    )
    query = linear(norm1, load_values(tensors, prefix + "attn_q.weight"), load_values(tensors, prefix + "attn_q.bias"))
    key = linear(norm1, load_values(tensors, prefix + "attn_k.weight"), load_values(tensors, prefix + "attn_k.bias"))
    value = linear(norm1, load_values(tensors, prefix + "attn_v.weight"), load_values(tensors, prefix + "attn_v.bias"))
    query = query.reshape(patch_count, 12, 64).transpose(1, 0, 2)
    key = key.reshape(patch_count, 12, 64).transpose(1, 0, 2)
    value = value.reshape(patch_count, 12, 64).transpose(1, 0, 2)
    scores = (query @ key.transpose(0, 2, 1)) * np.float32(1 / np.sqrt(np.float32(64)))
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True)).astype(np.float32)
    probabilities = scores / scores.sum(axis=-1, keepdims=True, dtype=np.float32)
    context = (probabilities @ value).transpose(1, 0, 2).reshape(patch_count, 768)
    attention = linear(context, load_values(tensors, prefix + "attn_out.weight"), load_values(tensors, prefix + "attn_out.bias"))
    residual1 = current + attention
    norm2 = layer_norm(
        residual1,
        load_values(tensors, prefix + "ln2.weight"),
        load_values(tensors, prefix + "ln2.bias"),
        np.float32(1e-6),
    )

    # Legacy IDEFICS3 GGUF naming is swapped: ffn_down is input, ffn_up is output.
    hidden = linear(norm2, load_values(tensors, prefix + "ffn_down.weight"), load_values(tensors, prefix + "ffn_down.bias"))
    hidden = np.float32(0.5) * hidden * (np.float32(1) + np.tanh(
        np.float32(0.7978845608028654) *
        (hidden + np.float32(0.044715) * hidden * hidden * hidden)
    ))
    output = residual1 + linear(
        hidden,
        load_values(tensors, prefix + "ffn_up.weight"),
        load_values(tensors, prefix + "ffn_up.bias"),
    )
    return output, {
        "norm1": norm1,
        "attention": attention,
        "residual1": residual1,
        "norm2": norm2,
        "hidden": hidden,
    }


def main() -> int:
    if len(sys.argv) not in (3, 4, 5):
        print("usage: siglip_block_reference.py <mmproj.gguf> <image> [patch-count] [block-count]", file=sys.stderr)
        return 2
    patch_count = int(sys.argv[3]) if len(sys.argv) == 4 else 4
    if len(sys.argv) == 5:
        patch_count = int(sys.argv[3])
    block_count = int(sys.argv[4]) if len(sys.argv) == 5 else 1
    if patch_count < 1 or patch_count > 32:
        raise ValueError("patch-count must be in [1, 32]")
    if block_count < 1 or block_count > 12:
        raise ValueError("block-count must be in [1, 12]")

    reader = GGUFReader(sys.argv[1])
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    rgb = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.uint8)
    resized = resize_bilinear_half_pixel(rgb, 512)
    chw = ((resized.astype(np.float32) / np.float32(255) - np.float32(0.5)) /
           np.float32(0.5)).transpose(2, 0, 1)
    patches = chw.reshape(3, 32, 16, 32, 16).transpose(1, 3, 0, 2, 4).reshape(1024, 768)

    embedding = linear(
        patches[:patch_count],
        load_values(tensors, "v.patch_embd.weight"),
        load_values(tensors, "v.patch_embd.bias"),
    ) + load_values(tensors, "v.position_embd.weight")[:patch_count]

    print(f"patches={patch_count} hidden=768")
    print(f"patch_checksum={checksum(patches[:patch_count]):.9g}")
    print(f"embedding_checksum={checksum(embedding):.9g}")
    print(f"embedding_first={embedding.reshape(-1)[0]:.9g}")
    current = embedding
    trace: dict[str, np.ndarray] = {}
    for block in range(block_count):
        current, trace = run_block(current, tensors, block)
        if block_count > 1:
            print(f"layer={block} checksum={checksum(current):.9g} first={current.reshape(-1)[0]:.9g}")

    if block_count == 1:
        print(f"norm1_checksum={checksum(trace['norm1']):.9g}")
        print(f"attention_checksum={checksum(trace['attention']):.9g}")
        print(f"residual1_checksum={checksum(trace['residual1']):.9g}")
        print(f"norm2_checksum={checksum(trace['norm2']):.9g}")
        print(f"ffn_checksum={checksum(trace['hidden']):.9g}")
        print(f"output_checksum={checksum(current):.9g}")
        print(f"first={current.reshape(-1)[0]:.9g}")
        print("reference=numpy-valid")
    else:
        output = layer_norm(
            current,
            load_values(tensors, "v.post_ln.weight"),
            load_values(tensors, "v.post_ln.bias"),
            np.float32(1e-6),
        )
        print(f"blocks_executed={block_count}")
        print(f"post_norm_checksum={checksum(output):.9g}")
        print(f"first={output.reshape(-1)[0]:.9g}")
        print("reference=numpy-encoder-valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
