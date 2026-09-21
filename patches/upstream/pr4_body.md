# pr4 — SM8x sparse-MLA prefill: head-blocked kernel (KV shared across heads, `tl.dot`)

**Status: design + prototype, NOT yet wired or validated.**
Written 2026-09-21 after filing upstream [vllm#57971](https://github.com/vllm-project/vllm/issues/57971).

## Why

`_sparse_mla_fwd_with_sink_kernel` (`vllm/v1/attention/backends/mla/sparse_mla_kernels.py`)
is the only sparse-MLA route on Ampere, and on a 16384-token GLM-5.3-Flash prefill it is
58.8% of CUDA time (22.3 s). Two independent problems, both fixable without touching the
sparsity semantics:

1. **The grid is `(num_tokens, active_heads)` and the KV address computation does not
   reference `head_idx`.** MLA's latent is shared across query heads, so each of the 64
   heads re-loads the same rows: ~2062 GB against the 32.2 GB the algorithm needs, a 64x
   redundancy (the ratio is independent of row width).
2. **No tensor cores.** `scores = tl.sum(kv * q[None, :], axis=1)` and the PV step are
   elementwise multiply + reduction. For the same reason as (1), there is no shared tile to
   hand to `tl.dot`.

This patch fixes both at once by blocking over heads: load the KV tile once per (token, KV
block) and reuse it for a block of heads, which simultaneously makes `tl.dot` legal because
`q` (BLOCK_H, D) and `kv` (BLOCK_K, D) are now both resident in the program.

## Feasibility (checked against the existing wrapper)

```python
block_d = 512 if head_dim <= 512 else next_power_of_2(head_dim)   # 512 for GLM-5.3
BLOCK_K = 16                                                       # existing value
```

- `tl.dot(q (BLOCK_H, 512), tl.trans(kv) (512, BLOCK_K))` — reduction dim 512, fine.
- `tl.dot(weights (BLOCK_H, BLOCK_K), kv (BLOCK_K, 512))` — reduction dim `BLOCK_K = 16`,
  which is exactly Triton's minimum for `tl.dot`. It fits, but **there is no headroom**: if
  `BLOCK_K` were ever lowered below 16 the second matmul would stop being legal. Worth a
  comment in the code.

`BLOCK_H` must be >= 16 for the first `tl.dot`, so 16 is the natural choice: the grid becomes
`(num_tokens, cdiv(H, 16))` = `(16384, 4)` = 65,536 programs instead of 1,048,576, and KV
traffic drops by up to 16x (64x is what perfect reuse would give; 16x is what one block of 16
heads gives, with the remaining factor available by raising BLOCK_H if registers allow).

## Prototype kernel

```python
@triton.jit
def _sparse_mla_fwd_with_sink_kernel_hb(
    q_ptr, kv_ptr, indices_ptr, lens_ptr, attn_sink_ptr, output_ptr,
    stride_q_t, stride_q_h, stride_q_d,
    stride_kv_r, stride_kv_d,
    stride_idx_t, stride_idx_k,
    stride_out_t, stride_out_h, stride_out_d,
    qk_scale,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must stay >= 16: it is the reduction dim of the PV tl.dot
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,   # must stay >= 16: it is the M dim of the QK^T tl.dot
):
    token_idx = tl.program_id(0)
    head_blk = tl.program_id(1)
    head_offs = head_blk * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offs < num_heads

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr + token_idx * stride_q_t
        + head_offs[:, None] * stride_q_h + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :], other=0.0,
    ).to(tl.float32)

    length = tl.load(lens_ptr + token_idx)
    # Same sink semantics as the original: the sink acts as a virtual key whose score is the
    # initial running max and whose contribution carries weight 1.
    running_max = tl.load(attn_sink_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    running_denom = tl.zeros((BLOCK_H,), tl.float32) + 1.0
    running_acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    for k_start in range(0, topk, BLOCK_K):
        candidates = k_start + tl.arange(0, BLOCK_K)
        k_mask = (candidates < length) & (candidates < topk)
        row = tl.load(
            indices_ptr + token_idx * stride_idx_t + candidates * stride_idx_k,
            mask=k_mask, other=0,
        ).to(tl.int64)
        row_safe = tl.where(k_mask, tl.maximum(row, 0), 0)

        # === the fix: loaded once, used by every head in the block ===
        kv = tl.load(
            kv_ptr + row_safe[:, None] * stride_kv_r + dim_offsets[None, :] * stride_kv_d,
            mask=k_mask[:, None] & dim_mask[None, :], other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * qk_scale          # (BLOCK_H, BLOCK_K)
        scores = tl.where(k_mask[None, :], scores, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        previous_weight = tl.exp(running_max - next_max)
        weights = tl.exp(scores - next_max[:, None])
        running_acc = running_acc * previous_weight[:, None] + tl.dot(weights.to(kv.dtype), kv)
        running_denom = running_denom * previous_weight + tl.sum(weights, axis=1)
        running_max = next_max

    tl.store(
        output_ptr + token_idx * stride_out_t
        + head_offs[:, None] * stride_out_h + dim_offsets[None, :] * stride_out_d,
        running_acc / running_denom[:, None],
        mask=head_mask[:, None] & dim_mask[None, :],
    )
```

Launch site replacement (same file, ~line 3639):

```python
BLOCK_H = 16
_sparse_mla_fwd_with_sink_kernel_hb[(num_tokens, triton.cdiv(active_heads, BLOCK_H))](
    q, kv, indices, topk_length, scale, attn_sink, output,
    q.stride(0), q.stride(1), q.stride(2),
    kv.stride(0), kv.stride(1),
    indices.stride(0), indices.stride(1),
    output.stride(0), output.stride(1), output.stride(2),
    scale,
    num_heads=active_heads,
    head_dim=head_dim,
    topk=topk,
    BLOCK_K=16,
    BLOCK_D=block_d,
    BLOCK_H=BLOCK_H,
    num_warps=8,
)
```

Note the wrapper zeroes heads past `num_heads`; with `head_mask` in the store that behaviour
is preserved, but **the zeroing must be re-verified** because the original relied on the grid
covering every head individually.

## Things that must be validated before this is a real patch

1. **Numerics.** Compare against the original kernel on the same inputs. The online-softmax
   bookkeeping changes shape (scalar denom -> per-head vector); the sink handling in
   particular needs a direct A/B, since the original seeds `running_max` from the sink and
   sets `running_denom = 1.0`, and that asymmetry is easy to get subtly wrong when the
   accumulator gains a head dimension.
2. **`weights.to(kv.dtype)`.** `tl.dot` wants both operands the same dtype. `kv` is bf16;
   `weights` is computed in fp32. Casting is the standard flash-attention move but it is a
   precision change relative to the original fp32 elementwise path, so it belongs in the
   numeric comparison above, not in a separate "it looks fine" step.
3. **Register pressure.** `running_acc` is now `(BLOCK_H, BLOCK_D)` = `(16, 512)` fp32 = 32 KB
   of registers per program, plus `q` (another 32 KB). This may spill or fail to compile at
   `BLOCK_H=16`; if so, try `BLOCK_H=16` with a smaller `BLOCK_D` split, or fall back to
   `BLOCK_H=8` and accept that `tl.dot` needs `M >= 16` (which would force a different
   structure — e.g. keep 16 heads but split D across two programs).
4. **The `-inf` + `tl.dot` interaction.** `tl.dot` on a tile containing rows that are entirely
   masked (`k_mask` false for all candidates in a block, which happens once
   `candidates >= length`) produces garbage that is then multiplied by `exp(-inf) = 0`. That
   is fine mathematically, but a NaN in the garbage would propagate; the original avoided this
   by construction. Needs an explicit test at a `topk`/`length` boundary (e.g. a prompt
   shorter than `topk`, which the scaling experiment showed is the common case for short
   prompts).
5. **Does it actually help?** Expected: KV traffic / 16 and `tl.dot` on tensor cores. But the
   kernel is also ~17x off the *redundant* bandwidth bound, so the small masked gathers
   (`BLOCK_K=16`, 128 iterations) may turn out to dominate. If so, raising `BLOCK_K` is the
   next lever and it composes with this change.

## Wiring and testing

Keep this out of the live tree until validated. The project convention is a `.patch` plus a
`prN_body.md` in `patches/upstream/`, following `pr2-fp8-sm80-o-proj.patch` /
`pr3-sm80-port.patch`. For a first look, edit
`vllm/v1/attention/backends/mla/sparse_mla_kernels.py` in the rebase tree directly, run one
16384-token request with `XIAOTU_TORCH_PROFILE` set, and check whether
`_sparse_mla_fwd_with_sink_kernel` — note the name — has been replaced in the profile table
and by how much `TTFT` moved. Then revert and regenerate as a patch.

The comparison baseline is already recorded: 22.3 s for that kernel at L=16384, which is
58.8% of CUDA time, with the scaling points 3.129 / 9.538 / 22.323 s at L = 4096 / 8192 /
16384.
