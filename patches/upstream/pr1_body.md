## Goal

Make a MoE model whose **routed-expert weights do not fit in device memory** constructible and
servable in mainline vLLM, by keeping those weights on the host while everything else
(attention, router, shared experts, KV cache) stays on the GPU.

This is the enabling switch for a hybrid CPU/GPU MoE serving mode; it changes **no default
behaviour**.

## Problem

- DeepSeek-V4-Flash has 43 layers × 256 routed experts × **3.19 GiB** of raw MXFP4 weights per
  layer = **~137 GiB**. On a single 40 GB A100 the model OOMs *during construction*, before a
  single token is served.
- vLLM's existing offload machinery (`model_executor/offloader/prefetch.py`) moves weights that
  have already been constructed; it cannot prevent the construction-time allocation.
- Mainline has no way to say "build the routed-expert weights on the host". The MXFP4 oracle
  always prefers a GPU backend, which then fails with `b_q_weight is not on GPU` once the
  weights live on the host.
- ktransformers / lk_moe solve the same problem by keeping experts on the host and computing
  them on the CPU, which is what we do downstream — but they need a fork.

## Design

1. **`vllm/envs.py`** — new `VLLM_EXPERTS_LOAD_DEVICE` (`"gpu"` default, `"cpu"` opt-in).
   Orthogonal to `VLLM_TARGET_DEVICE`: the target device is still CUDA, only the routed-expert
   *parameters* live on the host.
2. **`routed_experts.py`** — when the switch is set, `create_weights()` runs inside
   `with torch.device("cpu")`. `create_weights()` allocates with no explicit device, so a scoped
   default device keeps the parameters on CPU from construction onward and the existing weight
   loaders copy into them in place. Everything else the layer owns (router tables, expert maps,
   shared experts) keeps its normal device.
3. **`fused_moe/oracle/mxfp4.py`** — with the switch set:
   - select the CPU backend unconditionally instead of preferring the GPU kernels, and
   - skip the AMX prepack, because the consumer is an out-of-tree CPU engine that reads the raw
     `[E, 2I, H//2]` / `[E, H, I//2]` uint8 weights and the raw e8m0 block scales directly.

   With `VLLM_EXPERTS_LOAD_DEVICE=gpu` (default) the oracle takes exactly the same path as before.

## Data layout the CPU backend consumes (context for reviewers)

| tensor | shape | meaning |
|---|---|---|
| `w13_weight` | `[E, 2I, H//2]` u8 | MXFP4 e2m1 nibbles, gate rows `[0,I)` then up rows `[I,2I)` |
| `w13_weight_scale` | `[E, 2I, H//32]` u8 | e8m0 block scales (`2^(byte-127)`), groupK = 32 |
| `w2_weight` | `[E, H, I//2]` u8 | MXFP4 e2m1 nibbles |
| `w2_weight_scale` | `[E, H, I//32]` u8 | e8m0 block scales |

This is the checkpoint's on-disk layout, so no repack is needed on load — which is the point of
skipping the AMX prepack.

## Why this is useful (measured on our box)

2× A100-PCIE-40GB (SM80), AMD EPYC 9654, DeepSeek-V4-Flash, CPU experts + GPU attention:

| prompt tokens | CPU prefill (hybrid) | GPU prefill (layerwise weight streaming) |
|---|---:|---:|
| 2 091 | 112.9 s | **5.6 s** |
| 4 137 | 439.4 s | **5.8 s** |
| 16 039 | – | **17.7 s** |

The CPU path is what this PR enables; the GPU path (stream one layer's weights H2D, compute,
free) is a separate, out-of-tree mechanism. Full report, code and raw data:
<https://github.com/yeungtuzi/vllm-xtu-moe>.

## Testing

- 2× A100-PCIE-40GB + EPYC 9654, DeepSeek-V4-Flash + out-of-tree CPU MoE engine: the model
  constructs and serves where it previously OOM'd at construction.
- `VLLM_EXPERTS_LOAD_DEVICE=gpu` (default): unchanged behaviour, existing GPU backends unaffected.
- No kernel changes; the diff is three files, +59/−2.

## Related

- #56119 fixes DeepSeek-V4 `o_proj` on SM8.x (needed for correct output on A100).
- #56120 adds the portable SM80 Triton path so DeepSeek-V4 builds on Ampere at all.
- An RFC for a generic "layerwise GPU prefill for CPU-offloaded experts" hook will follow.
