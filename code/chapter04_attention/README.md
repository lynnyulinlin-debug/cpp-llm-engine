# Chapter 04: Attention

This directory contains the minimal implementation for `chapter04_attention`.

## Scope

- Multi-head attention
- LayerNorm and activation
- Feed forward network
- DecoderLayer composition
- First-layer weight loading and validation

## How to run

- `bash scripts/run.sh`

## Current status

- Library, demo, and test targets are available
- Multi-head attention, LayerNorm, FFN, and DecoderLayer are wired together
- This chapter is the structural bridge from operators to generation runtime
- The chapter also defines the decoder block as a verifiable structural unit, so later runtime issues can be separated from model-shape issues
