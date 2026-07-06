# Chapter 03: MatMul

This directory contains the minimal implementation for `chapter03_matmul`.

## Scope

- Operator base class
- MatMulOp design
- CPU reference implementation
- cuBLAS implementation
- PyTorch comparison tests

## How to run

- `bash scripts/run.sh`

## Current status

- Library, demo, and test targets are available
- CPU reference path is verified
- CPU reference path is verified as the semantic baseline
- cuBLAS path is wired as the high-performance backend, but the operator interface remains the source of truth
- This chapter is the operator baseline for later attention / FFN work
- The current code keeps the interface explicit so backend swaps do not blur the operator contract
