# [Perf] SM8x sparse-MLA prefill fallback re-reads the shared MLA latent once per head (~64x redundant KV traffic)

## Summary

On Ampere (`sm_80`), the portable Triton sparse-MLA fallback used for DSA-style models
(`_sparse_mla_fwd_with_sink_kernel`) dominates long-prompt prefill — 58.8% of CUDA time on a
single 16K-token request in my measurements. When I went looking for why, the kernel turns out
to re-read the MLA latent KV once **per query head** even though that latent is shared across
heads, and on top of that it only reaches about 6% of HBM bandwidth. I think there are two
separate fixable problems here, and I wanted to write them up with numbers rather than just
say "the fallback is slow".

I'm filing this as a performance issue, not a correctness one — the kernel produces correct
results as far as I can tell, and I understand the SM8x path exists so Ampere can run at all
(the comment in `glm5next/common/attention.py` says as much). This is about the cost of that
choice being much larger than it needs to be.

## Environment

- Model: `zai-org/GLM-5.3-Flash` (DSA / `is_v32` sparse-MLA path, `index_topk=2048`,
  `num_attention_heads=64`, `kv_lora_rank=512`, `qk_rope_head_dim=0` — the NoPE layout)
- Hardware: 2x A100-PCIE-40GB (compute capability 8.0), tensor parallel = 2
- vLLM: recent mainline checkout (the commit I'm on maps to the tree that contains
  `vllm/v1/attention/backends/mla/sparse_mla_kernels.py`)
- KV cache: bf16 (the SM8x path requires it)
- Workload: single request, 16384-token random prompt, concurrency 1

## What I measured

`torch.profiler` over one prefill request, CUDA time sorted:

```
_sparse_mla_fwd_with_sink_kernel     22.325 s   58.80%   (9 calls)
vllm::moe_forward_shared             11.628 s   30.28%
  Memcpy HtoD (Pageable -> Device)    6.157 s   16.22%
  down_kernel_fp8                     2.879 s    7.58%
  gate_up_kernel_fp8                  2.549 s    6.71%
  vllm::all_reduce                    2.084 s    5.49%
```

TTFT for that request was ~56 s. The attention kernel is the single biggest cost, and it is
the only thing on that list that is *not* doing useful work at a sane rate.

Scaling, measured at three prompt lengths (each with `MBT = L`, so one chunk per layer, and
the same profiler window throughout — so the comparison is apples to apples):

| prompt len | attention kernel | TTFT |
|---|---|---|
| 4,096 | 3.129 s | 28.7 s |
| 8,192 | 9.538 s | 38.5 s |
| 16,384 | 22.323 s | 56.0 s |

4x the prompt length costs 7.1x the kernel time, i.e. roughly `O(L^1.42)`. That's between
linear and quadratic, which is what I'd expect: the inner loop is capped by `topk`, so the
work per token is `min(position, topk)` and the total is `sum_t min(pos_t, topk)`. That sum
predicts 5.0x for the 4x increase, so the sparsity is doing its job — this is not a
complexity blowup. The per-unit cost is the problem: 22.3 s over 31.5M key-attends works out
to about 0.71 ns per key-attend.

## Problem 1: the KV tile is loaded once per head, but MLA shares it across heads

The kernel's grid is `(num_tokens, active_heads)`:

```python
# vllm/v1/attention/backends/mla/sparse_mla_kernels.py
_sparse_mla_fwd_with_sink_kernel[(num_tokens, active_heads)](
    ...
    BLOCK_K=16,
    BLOCK_D=block_d,
    num_warps=8,
)
```

and inside the kernel:

```python
token_idx = tl.program_id(0)
head_idx  = tl.program_id(1)

q = tl.load(q_ptr + token_idx*stride_q_t + head_idx*stride_q_h + dim_offsets*stride_q_d, ...)

# note: no head_idx anywhere in this address computation
kv = tl.load(kv_ptr + row_safe[:, None]*stride_kv_r + dim_offsets[None, :]*stride_kv_d, ...)
```

`head_idx` appears in the `q` and `output` addressing but **not** in the KV addressing. That's
correct in the sense that the latent really is shared — but it means every one of the
`num_heads` programs covering the same token loads the same KV rows again.

For GLM-5.3 at L=16384 that is 64x the necessary traffic:

```
sum_t min(pos_t, topk)                    = 31.5M key-attends
KV bytes per head                         = 31.5M * 512 dims * 2 B  = 32.2 GB
x 64 heads (same rows, re-read each time) = 2062 GB
what the algorithm actually needs         =   32.2 GB
```

The 64x factor doesn't depend on my assumption about the row width — it's just `num_heads`.
(If the row is narrower than the 512 I assumed, the absolute GB numbers shrink, but the
redundancy ratio and the effective-bandwidth conclusion don't change.)

## Problem 2: even the redundant traffic runs at ~6% of HBM bandwidth

Reading 2062 GB at A100's 1555 GB/s would take ~1.33 s. The kernel takes 22.3 s, so it's
another ~17x off even the *redundant* bound. Effective bandwidth is about 92 GB/s against a
1555 GB/s peak.

I suspect `BLOCK_K=16` is a big part of this. With `topk=2048` each program runs
`2048/16 = 128` loop iterations, and every iteration issues a masked 2-D gather
(`k_mask[:, None] & dim_mask[None, :]`) of a `16 x BLOCK_D` tile. That's a lot of small
masked loads, and the `dim_mask` suggests `BLOCK_D` is padded above the real row width too.
I haven't profiled the individual loads so this part is a hypothesis — the 64x from Problem 1
is the part I'm confident in.

## Suggested direction

Problem 1 looks mechanical to fix without touching the sparsity semantics at all: keep the
same per-token topk indices, but stop giving each head its own program. Either put a head
block (`BLOCK_H`) in the same program so the KV tile is loaded once and reused for all heads
in the block, or make the second grid dimension index KV blocks and loop over heads
internally. Since the latent is genuinely shared, this is purely a matter of loop
restructuring.

Problem 2 is more of a tuning question — larger `BLOCK_K`, and/or checking whether the
`dim_mask` is forcing an over-wide `BLOCK_D` and how that interacts with Triton's
pipelining (`num_stages` isn't set here, so the default applies).

I'd be happy to try the restructuring and post numbers if that's useful — I just wanted to
check first whether there's a reason the grid is shaped this way that I'm not seeing,
e.g. a constraint from the `topk_lens`/sink handling or from how the SM8x backend has to
match the SM90 `flash_mla_sparse_fwd` contract.

## What I might be getting wrong

- I did not benchmark the SM90 sparse path, so I can't say how much of this is specific to
  the SM8x fallback versus inherent to the sparse formulation. On SM90 this goes through
  `flash_mla_sparse_fwd` (CUDA) rather than this Triton kernel, so I'd expect it to be much
  better, but that's an expectation, not a measurement.
- The 512-dim row width is inferred from the model config (`kv_lora_rank=512`, rope 0). If
  the kernel is actually called with a different `head_dim`, the absolute bandwidth figure
  changes; the redundancy ratio does not.
- The `9 calls` in the profile table is an artifact of my profiler window (`CALLS=40` on the
  MoE kernel), not a real per-request call count, so please disregard the call count. The
  timings are all from the same window, so the comparisons hold.
- I'm running this through a plugin that keeps expert weights in host RAM, which is why
  `moe_forward_shared` and the HtoD copies show up so large in my profile. Those are my
  plugin's costs and not relevant to this report — the attention kernel numbers are the
  vLLM-side ones.
