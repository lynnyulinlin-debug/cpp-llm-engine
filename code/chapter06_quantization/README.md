# Chapter 06: Quantization

This directory contains the minimal implementation for `chapter06_quantization`.

## Scope

- Quantization basics
- Tensor dtype extension
- Quantized weight loading
- Dequantize operator
- Quantized Linear path
- Accuracy validation

## How to run

- `bash scripts/run.sh`

## Current status

- Library, demo, and test targets are available
- INT8 tensor, dequantize operator, and quantized linear path are wired together
- This chapter is the deployment bridge between the generation runtime and benchmark closure
- The code is intentionally minimal so the tutorial can separate quantization gains, error sources, and backend tradeoffs before moving to benchmark evidence
- The chapter also keeps calibration and fusion decisions visible, so later extensions do not blur the deployment boundary
