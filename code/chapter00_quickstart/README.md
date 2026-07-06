# Chapter 00: Quickstart

This directory contains the reference material for `chapter00_quickstart`.

## Scope

- Environment setup
- Building `llama.cpp`
- Downloading a small GGUF model
- Running the first inference
- Troubleshooting the quickstart path

## Local Smoke Check

This directory now also includes a small C++ smoke executable that verifies the
quickstart build path itself.

- `CMakeLists.txt` builds a minimal executable
- `main_quick.cpp` checks an optional model path and prints a readable status
- `run.sh` configures, builds, and runs the smoke executable

This is not a replacement for `llama.cpp`. It is a tiny verification entry
point that keeps the quickstart path executable inside this repository.
