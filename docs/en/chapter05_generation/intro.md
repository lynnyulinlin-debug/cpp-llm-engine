# Chapter 05 Generation Runtime and Serving Basics

Turn a single DecoderLayer into a generation runtime with cache and sampling.

## What you will do

- Add embeddings and positional information
- Build naive generation and KV cache
- Connect sampling and end-to-end tests

## What you will have at the end

- a minimal generation runtime with cache and sampling
- A clear bridge to the next chapter
- A visible checkpoint for the English route

## Status

This is the English landing page for the chapter. The detailed Chinese pages remain the source of truth for now.

## Next step

Continue to [summary](summary) or move to [the next chapter](/en/chapter06_quantization/intro).

## Related Reading

- Prerequisites: read the Chapter 04 Attention pages first, because this runtime is built on top of the closed DecoderLayer structure.
- Later use: `chapter06_quantization` keeps the same runtime organization while swapping the weight representation, and `chapter07_benchmark` measures that same path.
- Roadmap: if you want the full 03–06 path first, start from [the Chapter 03 intro](/en/chapter03_matmul/intro). It summarizes the reading order from operators to quantization.
