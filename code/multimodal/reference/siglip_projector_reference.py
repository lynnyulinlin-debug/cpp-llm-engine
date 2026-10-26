#!/usr/bin/env python3
"""SmolVLM 12 层 SigLIP 与 IDEFICS3 Projector 的可选 NumPy 参考实现。"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
from PIL import Image

from siglip_block_reference import (
    GGUFReader,
    checksum,
    dequantize,
    layer_norm,
    linear,
    load_values,
    resize_bilinear_half_pixel,
    run_block,
)


def run_llama_decoder(sequence: np.ndarray,
                      tensors: dict[str, object],
                      block_count: int) -> np.ndarray:
    epsilon = np.float32(1e-5)
    hidden = sequence
    token_count = sequence.shape[0]
    for block in range(block_count):
        prefix = f"blk.{block}."
        attn_norm_weight = load_values(
            tensors, prefix + "attn_norm.weight"
        ).reshape(1, 960)
        normalized = hidden * np.reciprocal(np.sqrt(
            np.mean(hidden * hidden, axis=1, keepdims=True, dtype=np.float32) + epsilon
        )) * attn_norm_weight
        query = linear(
            normalized, load_values(tensors, prefix + "attn_q.weight"),
            np.zeros(960, dtype=np.float32),
        ).reshape(token_count, 15, 64)
        key = linear(
            normalized, load_values(tensors, prefix + "attn_k.weight"),
            np.zeros(320, dtype=np.float32),
        ).reshape(token_count, 5, 64)
        value = linear(
            normalized, load_values(tensors, prefix + "attn_v.weight"),
            np.zeros(320, dtype=np.float32),
        ).reshape(token_count, 5, 64)
        for position in range(token_count):
            for pair in range(32):
                angle = np.float32(position) * np.float32(
                    np.power(np.float32(100000), np.float32(-2 * pair / 64))
                )
                cosine = np.float32(np.cos(angle))
                sine = np.float32(np.sin(angle))
                for head in range(15):
                    x, y = query[position, head, 2 * pair:2 * pair + 2].copy()
                    query[position, head, 2 * pair] = x * cosine - y * sine
                    query[position, head, 2 * pair + 1] = x * sine + y * cosine
                for head in range(5):
                    x, y = key[position, head, 2 * pair:2 * pair + 2].copy()
                    key[position, head, 2 * pair] = x * cosine - y * sine
                    key[position, head, 2 * pair + 1] = x * sine + y * cosine

        context = np.zeros((token_count, 15, 64), dtype=np.float32)
        scale = np.float32(1 / np.sqrt(np.float32(64)))
        for position in range(token_count):
            for query_head in range(15):
                kv_head = query_head // 3
                scores = np.asarray([
                    np.dot(query[position, query_head], key[key_position, kv_head]) * scale
                    for key_position in range(position + 1)
                ], dtype=np.float32)
                probabilities = np.exp(scores - scores.max()).astype(np.float32)
                probabilities /= probabilities.sum(dtype=np.float32)
                for key_position in range(position + 1):
                    context[position, query_head] += (
                        probabilities[key_position] * value[key_position, kv_head]
                    )
        attention = linear(
            context.reshape(token_count, 960),
            load_values(tensors, prefix + "attn_output.weight"),
            np.zeros(960, dtype=np.float32),
        )
        residual = hidden + attention
        ffn_norm_weight = load_values(
            tensors, prefix + "ffn_norm.weight"
        ).reshape(1, 960)
        ffn_input = residual * np.reciprocal(np.sqrt(
            np.mean(residual * residual, axis=1, keepdims=True, dtype=np.float32) + epsilon
        )) * ffn_norm_weight
        gate = linear(
            ffn_input, load_values(tensors, prefix + "ffn_gate.weight"),
            np.zeros(2560, dtype=np.float32),
        )
        up = linear(
            ffn_input, load_values(tensors, prefix + "ffn_up.weight"),
            np.zeros(2560, dtype=np.float32),
        )
        activated = gate / (np.float32(1) + np.exp(-gate).astype(np.float32)) * up
        hidden = residual + linear(
            activated, load_values(tensors, prefix + "ffn_down.weight"),
            np.zeros(960, dtype=np.float32),
        )

    final = hidden[-1:] * np.reciprocal(np.sqrt(
        np.mean(hidden[-1:] * hidden[-1:], axis=1, keepdims=True, dtype=np.float32) + epsilon
    )) * load_values(tensors, "output_norm.weight").reshape(1, 960)
    return linear(
        final, load_values(tensors, "output.weight"),
        np.zeros(49280, dtype=np.float32),
    ).reshape(-1)


def dynamic_visual_images(rgb: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    image_height, image_width, _ = rgb.shape
    tile_size = 512
    pixel_budget = 256 * 16 * 16 * 4 * 4
    refined_width = max(tile_size, round(image_width / tile_size) * tile_size)
    refined_height = max(tile_size, round(image_height / tile_size) * tile_size)
    aligned_pixels = refined_width * refined_height
    if aligned_pixels > pixel_budget:
        beta = math.sqrt(image_width * image_height / pixel_budget)
        refined_width = max(tile_size, math.floor(image_width / beta / tile_size) * tile_size)
        refined_height = max(tile_size, math.floor(image_height / beta / tile_size) * tile_size)
    elif aligned_pixels < pixel_budget:
        beta = math.sqrt(pixel_budget / (image_width * image_height))
        refined_width = math.ceil(image_width * beta / tile_size) * tile_size
        refined_height = math.ceil(image_height * beta / tile_size) * tile_size

    scale = min(refined_width / image_width, refined_height / image_height)
    content_width = min(math.ceil(image_width * scale), refined_width)
    content_height = min(math.ceil(image_height * scale), refined_height)
    offset_x = (refined_width - content_width) // 2
    offset_y = (refined_height - content_height) // 2
    refined = np.zeros((refined_height, refined_width, 3), dtype=np.uint8)
    refined[offset_y:offset_y + content_height, offset_x:offset_x + content_width] = (
        resize_bilinear_half_pixel(rgb, content_width, content_height)
    )
    rows = refined_height // tile_size
    cols = refined_width // tile_size
    visuals = [
        refined[row * tile_size:(row + 1) * tile_size,
                col * tile_size:(col + 1) * tile_size].copy()
        for row in range(rows)
        for col in range(cols)
    ]
    visuals.append(resize_bilinear_half_pixel(rgb, tile_size))
    return visuals, rows, cols


def encode_visual(rgb: np.ndarray,
                  tensors: dict[str, object],
                  full_tile: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized = resize_bilinear_half_pixel(rgb, 512)
    chw = ((resized.astype(np.float32) / np.float32(255) - np.float32(0.5)) /
           np.float32(0.5)).transpose(2, 0, 1)
    patches = chw.reshape(3, 32, 16, 32, 16).transpose(1, 3, 0, 2, 4).reshape(1024, 768)
    positions = np.arange(1024, dtype=np.int64) if full_tile else np.asarray(
        [y * 32 + x for y in range(4) for x in range(4)], dtype=np.int64
    )
    current = linear(
        patches[positions],
        load_values(tensors, "v.patch_embd.weight"),
        load_values(tensors, "v.patch_embd.bias"),
    ) + load_values(tensors, "v.position_embd.weight")[positions]
    for block in range(12):
        current, _ = run_block(current, tensors, block)
    encoded = layer_norm(
        current,
        load_values(tensors, "v.post_ln.weight"),
        load_values(tensors, "v.post_ln.bias"),
        np.float32(1e-6),
    )
    merged = (
        encoded.reshape(8, 4, 8, 4, 768).transpose(0, 2, 1, 3, 4).reshape(64, 12288)
        if full_tile else encoded.reshape(1, 4, 4, 768).reshape(1, 12288)
    )
    projected = linear(
        merged, load_values(tensors, "mm.model.fc.weight"),
        np.zeros(960, dtype=np.float32),
    )
    return encoded, merged, projected


def main() -> int:
    if len(sys.argv) not in (3, 4, 5, 6, 7):
        print("usage: siglip_projector_reference.py <mmproj.gguf> <image> "
              "[model.gguf] [decoder-layers] [prompt-token-spec] "
              "[--full-tile|--dynamic-tiles]",
              file=sys.stderr)
        return 2
    mode = sys.argv[6] if len(sys.argv) == 7 else ""
    if mode not in ("", "--full-tile", "--dynamic-tiles"):
        raise ValueError("mode must be --full-tile or --dynamic-tiles")
    dynamic_tiles = mode == "--dynamic-tiles"
    full_tile = mode in ("--full-tile", "--dynamic-tiles")
    decoder_layers = int(sys.argv[4]) if len(sys.argv) >= 5 else 1
    if decoder_layers < 1 or decoder_layers > 32:
        raise ValueError("decoder-layers must be in [1, 32]")

    reader = GGUFReader(sys.argv[1])
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    rgb = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.uint8)
    visual_images, tile_rows, tile_cols = (
        dynamic_visual_images(rgb) if dynamic_tiles else ([rgb], 0, 0)
    )
    positions = np.arange(1024, dtype=np.int64) if full_tile else np.asarray(
        [y * 32 + x for y in range(4) for x in range(4)], dtype=np.int64
    )
    encoded_visuals = [encode_visual(visual, tensors, full_tile) for visual in visual_images]
    encoded, merged, projected = encoded_visuals[0]
    projected_visuals = [values[2] for values in encoded_visuals]

    if dynamic_tiles:
        print(f"source_image={rgb.shape[1]}x{rgb.shape[0]}")
        print(f"tile_grid={tile_rows}x{tile_cols}")
        print(f"visual_count={len(projected_visuals)}")
    print("vision_mode=" + (
        "dynamic-tiles" if dynamic_tiles else ("full-tile" if full_tile else "top-left-window")
    ))
    print("vision_backend=numpy")
    print("selected_positions=" + (
        f"{positions[0]}..{positions[-1]}" if full_tile else
        ",".join(str(value) for value in positions)
    ))
    print(f"encoder_shape={encoded.shape[0]}x{encoded.shape[1]}")
    print(f"encoder_checksum={checksum(encoded):.9g}")
    print(f"merged_shape={merged.shape[0]}x{merged.shape[1]}")
    print(f"merged_checksum={checksum(merged):.9g}")
    print(f"merged_first={merged.reshape(-1)[0]:.9g}")
    print(f"output_shape={projected.shape[0]}x{projected.shape[1]}")
    print(f"output_checksum={checksum(projected):.9g}")
    print(f"first={projected.reshape(-1)[0]:.9g}")
    print("visual_output_checksums=" + ",".join(
        f"{checksum(values):.9g}" for values in projected_visuals
    ))
    print("reference=numpy-projector-valid")
    if len(sys.argv) >= 4:
        language_reader = GGUFReader(sys.argv[3])
        token_embedding = next(
            tensor for tensor in language_reader.tensors if tensor.name == "token_embd.weight"
        )
        prompt_spec = sys.argv[5].split(",") if len(sys.argv) >= 6 else []
        if prompt_spec:
            if prompt_spec.count("visual") != len(projected_visuals):
                raise ValueError("prompt-token-spec visual count does not match visual inputs")
            visual_positions = [
                index for index, value in enumerate(prompt_spec) if value == "visual"
            ]
            token_ids = [1 if value == "visual" else int(value) for value in prompt_spec]
            text_with_placeholder = np.asarray(
                dequantize(token_embedding.data[token_ids], token_embedding.tensor_type),
                dtype=np.float32,
            ).reshape(len(token_ids), 960)
            parts = []
            visual_index = 0
            for index in range(len(prompt_spec)):
                if prompt_spec[index] == "visual":
                    parts.append(projected_visuals[visual_index])
                    visual_index += 1
                else:
                    parts.append(text_with_placeholder[index:index + 1])
            sequence = np.concatenate(parts, axis=0)
        else:
            boundary = np.asarray(
                dequantize(token_embedding.data[[1, 49279]], token_embedding.tensor_type),
                dtype=np.float32,
            ).reshape(2, 960)
            text_with_placeholder = np.concatenate(
                (boundary[:1], np.zeros((1, 960), dtype=np.float32), boundary[1:]), axis=0
            )
            sequence = np.concatenate((boundary[:1], projected, boundary[1:]), axis=0)
        print("language_architecture=llama")
        print("language_hidden=960")
        print("text_token_ids=1,49279")
        print(f"text_embedding_checksum={checksum(text_with_placeholder):.9g}")
        print(f"sequence_shape={sequence.shape[0]}x{sequence.shape[1]}")
        print(f"sequence_checksum={checksum(sequence):.9g}")
        if prompt_spec:
            print("prompt_mode=" + (
                "smolvlm-dynamic-tiles" if dynamic_tiles else
                ("smolvlm-single-tile" if full_tile else "smolvlm-overview-only")
            ))
            print(f"prompt_prefix_tokens={visual_positions[0]}")
            print(f"prompt_suffix_tokens={len(prompt_spec) - visual_positions[-1] - 1}")
            print(f"prompt_visual_count={len(projected_visuals)}")
            print(f"prompt_visual_rows={sum(values.shape[0] for values in projected_visuals)}")
            print("prompt_token_ids=" + ",".join(prompt_spec))
        else:
            print("sequence_origins=text:0,visual:0:0,text:2")
        print("reference=numpy-multimodal-input-valid")

        language_tensors = {tensor.name: tensor for tensor in language_reader.tensors}
        logits = run_llama_decoder(sequence, language_tensors, decoder_layers)
        top1 = int(np.argmax(logits))
        print(f"decoder_layers={decoder_layers}")
        print(f"decoder_tokens={sequence.shape[0]}")
        print(f"logits_size={logits.size}")
        print(f"logits_checksum={checksum(logits):.9g}")
        print(f"top1_token_id={top1}")
        print(f"top1_logit={logits[top1]:.9g}")
        print("reference=numpy-multimodal-decoder-valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
