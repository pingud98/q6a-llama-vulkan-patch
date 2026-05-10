# Q6A llama.cpp Vulkan Patch + Benchmarks

Everything needed to enable full GPU offload on Turnip Adreno 643.

## The Problem

llama.cpp's GPU backend crashes at context creation when offloading 3+ layers to Vulkan on Mesa Turnip:

```
pre-allocated tensor (cache_k_l3) in a buffer (Vulkan0)
that cannot run the operation (NONE)
```

## The Fix

A 12-line patch in `ggml/src/ggml-backend.cpp` that adds a fallback path: before aborting, it finds a backend matching the tensor's buffer type, ignoring op support. KV cache tensors don't need op-specific backends.

## Quick Start

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply /path/to/ggml-backend.cpp.patch
cmake -B build -DGGML_VULKAN=ON
cmake --build build --target llama-cli -j$(nproc)
./build/bin/llama-cli -m model.gguf -ngl 99
```

## Benchmarks

System: Q6A / Radxa Dragon Wing / Qualcomm QCS6490 / Adreno 643 / Mesa 25.0.7 / 12GB RAM

### Qwen3.5-0.8B Q4_K_M (521 MB)

| Config | Prefill (pp32) | Generation (tg128) |
|--------|---------------|-------------------|
| CPU | 15.2 tok/s | 12.2 tok/s |
| GPU (ngl=99) | 21.2 tok/s | 8.2 tok/s |
| *Delta* | *+40%* | *-33%* |

### Qwen3.5-0.8B Q8_K_XL (1.09 GB)

| Config | Prefill (pp32) | Generation (tg64) |
|--------|---------------|-------------------|
| CPU | 13.0 tok/s | 8.7 tok/s |
| GPU (ngl=99) | 21.0 tok/s | 7.5 tok/s |
| *Delta* | *+61%* | *-14%* |

### Summary

- GPU prefill is **40-61% faster** than CPU
- GPU generation is **14-33% slower** (Turnip lacks INT4 dot-product, dequantizes to fp16)
- **Q4_K_M on CPU is the sweet spot** for interactive use: 15/12 tok/s, zero setup
- Use GPU when you care about time-to-first-token (batch/search workloads)
