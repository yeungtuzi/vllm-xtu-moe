# RFC 草稿:CPU-offload MoE 的逐层 GPU prefill

> D4a 决策:先发 issue 征求方向,不直接提 PR。
> 目标仓库: `vllm-project/vllm`(issue,不是 PR)
> 建议标题: **[RFC] Layerwise GPU prefill for CPU-offloaded MoE (stream one layer's weights per step)**

---

## 正文草稿(可直接粘贴)

```markdown
### Motivation

Offloading routed-expert weights to host memory (`VLLM_EXPERTS_LOAD_DEVICE=cpu`,
see #<PR1>) makes MoE models larger than device memory *constructible*, but their
prefill then runs on the CPU and is one to two orders of magnitude slower than
the GPU path.

On our box (2x A100-PCIE-40GB, AMD EPYC 9654, DeepSeek-V4-Flash with 43 layers x
256 experts x 3.19 GiB of raw MXFP4 per layer = ~137 GiB total), CPU prefill
measures ~22 ms per token per... (full numbers below), while the GPU can do the
same work at ~11 us/token/layer once the weights are resident.

The weights do not fit in 40 GB, but **a single layer's weights do** (3.19 GiB),
and a prefill step streams them exactly once regardless of how many tokens are
in the batch. That makes a "stream one layer in, compute, free it" pipeline
attractive for prefill (not for decode, where the same bytes would move per
token).

### Proposal

Add an opt-in, model-agnostic hook for layerwise GPU prefill of CPU-offloaded
MoE experts:

- `VLLM_GPU_PREFILL_MIN_TOKENS` (int, 0 = off): when a layer's prefill batch has
  at least this many tokens, that layer's routed-expert computation runs on the
  GPU with the layer's weights streamed H2D for the duration of the layer.
- A small contract for the backend: `stream_layer_weights() -> handle` /
  `release(handle)`, with the engine responsible for the layout and the kernel.
- Double-buffered prefetch: while layer L computes, layer L+1's H2D is issued on
  a side stream (`cudaMemcpyAsync` from a pinned host copy), so the copy
  overlaps tensor-core work instead of serialising with it.

This is deliberately close to what `model_executor/offloader/prefetch.py`
(`PrefetchOffloader` + `StaticBufferPool`) already does, except that the weights
are **freed after each layer** instead of being pinned into a fixed set of
resident slots -- which is what makes it work when the whole model is larger than
VRAM. If maintainers prefer, the two could share the stream/event scaffolding.

### Measured behaviour (single A100 40GB, DeepSeek-V4-Flash)

| prompt tokens | CPU prefill | layerwise GPU prefill |
|---|---|---|
| 2 091 | 112.9 s | **5.6 s** |
| 4 137 | 439.4 s | **5.8 s** |
| 8 229 | - | **8.6 s** |
| 16 039 | - | **17.7 s** |

- The GPU path is flat at ~5.5 s up to 4K tokens: that is the H2D floor
  (137 GiB / ~25 GB/s). Cross-over with the CPU path is ~250 tokens.
- Pure MoE throughput (43-layer equivalent) crosses 1500 tok/s at ~8K tokens;
  32K tokens reaches ~2 000 tok/s.
- Numerics match a pure-torch reference to bf16 output precision
  (RMS relative error 5.6e-3).

Full report, code and raw measurements:
<https://github.com/yeungtuzi/vllm-xtu-moe>

### Questions for maintainers

1. Is a generic "layerwise GPU prefill for offloaded experts" hook something
   vLLM wants, or is the existing `PrefetchOffloader` (resident-slot) model the
   intended answer for this problem?
2. If yes, would you prefer the contract to live next to `PrefetchOffloader`, or
   as a `FusedMoEExperts` backend capability?
3. Any preference on the env-var name / gating semantics
   (`VLLM_GPU_PREFILL_MIN_TOKENS` vs a per-request scheduling hint)?
```

---

## 发布方式

```bash
gh issue create --repo vllm-project/vllm \
  --title "[RFC] Layerwise GPU prefill for CPU-offloaded MoE (stream one layer's weights per step)" \
  --body-file vllm-xiaotu-moe/patches/rfc_layerwise_gpu_prefill.md
```

> 建议先等 PR1 合入或至少有人回复,再发这个 RFC(否则链接不到 PR1)。

---
## 修订记录

- **2026-09-09(第 1 版)** — 按 D4a 决策产出 RFC 草稿(不直接提 PR)。
  依据:`docs/EXPERIMENT_REPORT.md` §5.3/§5.9 实测 + §4 的 PR 拆分。
