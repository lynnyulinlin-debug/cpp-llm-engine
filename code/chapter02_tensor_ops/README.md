# Chapter 02: Tensor Ops

This directory contains the minimal `chapter02_tensor_ops` implementation.

## Scope

- Shared storage
- `reshape`
- `slice`
- View semantics
- Basic tests for storage sharing

## How to run

- `bash scripts/run.sh`

## Current status

- Library, demo, and test targets are available
- `reshape` and `slice` are verified
- Shared storage and view semantics are the base for later operator chapters
- This chapter defines the Tensor object as a view-aware wrapper over shared storage, not just a passive data container
