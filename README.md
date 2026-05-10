# Q6A llama.cpp Vulkan Patch + Benchmarks

Fixes llama.cpp's GPU backend on **Turnip Adreno 643** (Mesa Freedreno) — the GPU inside the **Radxa Dragon Wing Q6A** (Qualcomm QCS6490).

## The Problem

With `-ngl N` where N >= 3, llama.cpp crashes at context creation:

```
pre-allocated tensor (cache_k_l3) in a buffer (Vulkan0) that cannot run the operation (NONE)
```

**Root cause:** The backend scheduler in `ggml-backend.cpp` aborts when it finds a pre-allocated tensor (KV cache) whose backend supports the buffer type but doesn't register the NONE identity operation. On Turnip / Vulkan, NONE ops have no shader backend, so the scheduler gives up and aborts.

## The Fix

A 12-line patch in `ggml/src/ggml-backend.cpp`. Before aborting, it tries a fallback match: find a backend that supports the tensor's buffer type (`buft`) even if it doesn't support the particular op. For KV cache tensors (data-only), this is the correct behavior.

```patch
+        {
+            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
+            for (int i = 0; i < sched->n_backends; i++) {
+                if (ggml_backend_supports_buft(sched->backends[i], buf->buft)) {
+                    cur_backend_id = i;
+                    SET_CAUSE(tensor, "1.buft");
+                    return cur_backend_id;
+                }
+            }
+        }
```

## Apply the Patch

```bash
cd /path/to/llama.cpp
git apply /path/to/ggml-backend.cpp.patch
mkdir -p build && cd build
cmake -B build -DGGML_VULKAN=ON
cmake --build build --target llama-cli -j$(nproc)
```

Then run with full GPU offload:

```bash
./build/bin/llama-cli -m model.gguf -p "Hello" -n 128 -ngl 99
```

The patch works with or without `--no-warmup`. Does not require `-nkvo` or `-fa`.

## Benchmarks

Hardware: **Radxa Dragon Wing Q6A** — Qualcomm QCS6490, 12GB RAM, Adreno 643 (Turnip Mesa 25.0.7)

### Qwen3.5-0.8B — Q4_K_M (521 MB)

| Config | Prefill pp32 | Gen tg128 |
|--------|-------------|-----------|
| CPU (ngl=0) | 15.17 t/s | 12.24 t/s |
| **GPU (ngl=99)** | **21.18 t/s** | **8.24 t/s** |

### Qwen3.5-0.8B — Q8_K_XL (1.09 GB)

| Config | Prefill pp32 | Gen tg64 |
|--------|-------------|----------|
| CPU (ngl=0) | 13.01 t/s | 8.74 t/s |
| GPU (ngl=1) | 9.4 t/s | 3.3 t/s |
| **GPU (ngl=99)** | **21.01 t/s** | **7.49 t/s** |

### Observations

- **GPU prefill is 40-60% faster** than CPU in both quantizations
- **CPU generation is faster** for Q4_K_M (12 vs 8 t/s) — Turnip lacks INT4 dot-product instructions, so the GPU dequantizes to fp16 internally
- **Q4_K_M on CPU** is the overall sweet spot: 15/12 t/s, no GPU setup needed
- The patch enables `-ngl 99` for anyone who wants fast prefill or batch processing

## What Didn't Work

Claude Code suggested three alternatives that all failed:

- `--flash-attn` (-fa) — still hits the same scheduler abort
- `GGML_VK_DISABLE_F16=1` — no effect on the scheduler bug
- `-ub 64` — doesn't help when KV offload is on

The only workaround before the patch was `-ngl 99 -nkvo` (no KV offload), which limits performance to 8.9/4.0 t/s.

## Other Findings

- **NPU (Hexagon v68)**: Blocked — kernel lacks `CONFIG_QCOM_FASTRPC_UNSIGNED_MODULE=y`
- **LiteRT-LM**: Works on CPU/GPU, see [npu/litert-lm.md](https://github.com/pingud98/projects-wiki/blob/main/npu/litert-lm.md)
- **llvmpipe**: Software Vulkan renderer will be detected but ggml_vulkan intentionally skips CPU-type devices

## Files

- `ggml-backend.cpp.patch` — the actual patch
- `README.md` — this file
