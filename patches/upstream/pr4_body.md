# pr4 — SM8x sparse-MLA prefill: 2-D tile shape (**validated, 2.88x**)

**Status: measured end-to-end, 2026-09-21. Tree reverted; patch is standalone.**
Supersedes the earlier head-blocking rationale in this file (kept at the bottom for the record).

Patch: `pr4-sm8x-sparse-mla-2d-tile.patch` (102 lines)

## What actually works

Make `q` and `running_acc` **two-dimensional** `(BLOCK_H, BLOCK_D)` tiles and put `BLOCK_H`
heads in one program, so the KV tile is loaded once per (token, K-block). Same math, same
per-token `topk` indices, no change to the sparsity semantics.

## Measured

Standalone kernel harness (`/tmp/kern_ab.py`, no model load), realistic shapes
N=1024 H=64 D=512 topk=2048 R=32768 (128 loop iterations):

| variant | time | speedup |
|---|---|---|
| original | 311.208 ms | 1.00x |
| **BLOCK_H=1** | 111.853 ms | **2.78x** |
| BLOCK_H=2 | 108.929 ms | 2.86x |
| BLOCK_H=4 | 135.885 ms | 2.29x |
| **BLOCK_H=8** | **108.185 ms** | **2.88x** |
| BLOCK_H=16 | 532.517 ms | 0.58x |

End-to-end, single 16384-token GLM-5.3-Flash prefill, same config as the baseline:

| metric | baseline | 2-D tile | change |
|---|---|---|---|
| TTFT | 55,954 ms | **36,894 ms** | **1.52x** |
| `_sparse_mla_..._kernel` | 22.325 s | **7.739 s** | **2.88x** |
| attention share | 58.80% | **33.15%** | −25.7pt |
| illegal access | 0 | 0 | — |

## The important correction

I originally argued this change was about reusing the shared MLA latent across heads, because
the KV address computation does not reference `head_idx` and the grid is
`(num_tokens, active_heads)` — so each of the 64 heads re-reads the same rows. That
observation is correct and is filed as vllm#57971.

**It is not what makes this fast.** `BLOCK_H=1` reuses nothing at all and is already 2.78x
faster; going to `BLOCK_H=8` adds about 3%. So the kernel is not KV-bandwidth bound, and the
win comes from the tile shape changing what Triton generates. Why the 2-D form is that much
better is **not explained** — I asked upstream in the issue follow-up
([comment](https://github.com/vllm-project/vllm/issues/57971#issuecomment-5762891125)).
`BLOCK_H=4` being worse than 1, 2 and 8 looks like an occupancy effect and should not be read
as a trend. `BLOCK_H=16` is a clear loss on both time and precision.

## Numerics

**Not bit-exact.** `max|diff|` = 1.965e-05 (small shapes) / 2.819e-05 (realistic), from
reduction order: `tl.sum(q[:, None, :] * kv[None, :, :], axis=2)` versus the original's
`tl.sum(kv * q[None, :], axis=1)`. That is fp32 accumulation over bf16 inputs, so it is
probably benign, but it is a difference and needs a decision rather than a shrug.
`BLOCK_H=16` measured 1.170e-03, sixty times worse.

## Why it is faster (bisected)

I bisected the two changes on the standalone harness, using four variants that share one loop
body, at the realistic shapes:

| variant | time | speedup |
|---|---|---|
| original | 311.200 ms | 1.00x |
| **2-D `q`, 1-D `acc`** | **110.899 ms** | **2.81x** |
| 1-D `q`, 2-D `acc` | compile error | — |
| 2-D `q`, 2-D `acc` (= this patch) | 112.572 ms | 2.76x |
| 2-D `q`, 2-D `acc`, `BLOCK_H=8` | 108.176 ms | 2.88x |

**The whole 2.8x comes from loading `q` as a 2-D `(1, BLOCK_D)` tile.** The 2-D accumulator
contributes nothing, and head blocking adds only about 3% on top.

That means two things. First, the mechanism is narrower than I described: in the original, `q`
is a 1-D `(BLOCK_D,)` vector and `tl.sum(kv * q[None, :], axis=1)` broadcasts it across the K
dimension of a 2-D tile; with the 2-D load the layouts line up. The plausible reason is that
the 1-D broadcast forces a layout conversion or costs vectorisation on the `(K, D)` product,
but **I have not verified that** -- the check would be diffing `convert_layout` and
`ld.global` widths in the ttgir/PTX of the two forms. Second, a smaller patch should get
2.81x of the 2.88x: changing only the `q` load. I have kept the head-blocked form because it
is the one measured end-to-end, but the minimal version is the one worth reviewing.

## Precedent in this codebase

Head-blocked prefill is not a new idea here. `sparse_mla_kernels.py` already contains a
head-blocked path for the same kind of attention:

```python
_PREFILL_INDEXED_HEAD_BLOCK = 8                      # line 1041
def accumulate_indexed_sparse_mla_attention_chunk(...):
    head_block = _PREFILL_INDEXED_HEAD_BLOCK
    if num_heads >= head_block:
        grid = (num_tokens, triton.cdiv(num_heads, head_block))
        _accumulate_indexed_attention_chunk_multihead_kernel[grid](..., HEAD_BLOCK=head_block, ...)
```

and it is already used in production by two models:

* `vllm/models/deepseek_v4/nvidia/flashmla.py:755`
* `vllm/models/deepseek_v41/nvidia/flashmla.py:569`

GLM does not use it. `flashmla_sparse_sm8x.py:165` calls `sparse_mla_fwd_with_sink`, the
per-token kernel with no head blocking, and that backend sets
`supports_dense_mha_prefill = False` with a comment saying every token, prefill included, is
forced through the sparse-MQA gather path.

Two things follow. The block size in this patch is not arbitrary -- `BLOCK_H = 8` matches
`_PREFILL_INDEXED_HEAD_BLOCK`, which is the value already in use. And the approach has
production precedent rather than being an unproven idea.

It does **not** follow that the fix should instead be to route GLM through
`accumulate_indexed_sparse_mla_attention_chunk`. I assumed that at first and had to withdraw
it: the two are different decompositions, not one function called differently. That path runs
a token-chunk loop around a topk-chunk loop plus a separate finish kernel, with
`max_score`/`denom`/`acc` allocated and threaded by the caller, so adopting it means
restructuring GLM's prefill into that pipeline. This patch is self-contained and keeps GLM's
single-call interface, which is probably the smaller change of the two.

## Open items

1. **The 1-D-broadcast explanation is still unverified.** The bisect says where the time goes
   (the 2-D `q` load), not why. A TTIR/PTX diff would settle it; I tried `TRITON_KERNEL_DUMP`
   and failed to capture both variants (switching the dump dir within one process caught only
   the second kernel; separate processes caught neither). Reading `asm` out of the compiled
   kernel via `device_caches` or `triton.compile` is the untried alternative. **Do not state
   the layout-conversion explanation as fact until then.**
2. **End-to-end numerics are unverified.** Kernel-level A/B is done: `max|diff| = 2.819e-05`
   at realistic shapes, from reduction order, not bit-exact. Confirming that this propagates
   to identical model output is still owed. Six attempts failed on my own scripting, not on
   the code — a one-word prompt, a `$tag_w` typo, a `pgrep` matching my own command line, a
   relative patch path that silently compared the unpatched kernel against itself, a
   comparison on `message.content` when `finish_reason=length` leaves it `None`, and a missing
   `chmod +x`. A seventh version is ready (arm B only, absolute patch path, compare
   `token_ids`, no `kill -PGID`) but has not been run.
3. `BLOCK_H` should be 2 or 8; 4 measured worse than 1, 2 and 8 (non-monotonic, looks like an
   occupancy effect) and 16 was a clear loss on both time and precision. It is 8 here, which
   also matches `_PREFILL_INDEXED_HEAD_BLOCK`.
4. The wrapper zeroes heads past `num_heads`; the new kernel writes them under `head_mask`
   instead. **That path was never exercised** — every test used `H` a multiple of `BLOCK_H`,
   and GLM's 64 heads divide evenly by 8, so the masking branch is untested.
5. The rebase tree was reverted after each measurement and verified by md5 and `diff -q`
   against a backup. One exception is recorded in the experiment ledger (B20): a run was
   interrupted after `git apply` but before the revert, leaving the tree modified; it was
   caught and reverted, and the three-way check is now mandatory rather than incidental.

## Appendix: the superseded head-blocking rationale

The original version of this file argued the change was about KV traffic (2062 GB read
against 32.2 GB needed, 64x). The arithmetic was right; the causal claim was wrong. Kept here
only so the record shows what was believed before the measurement.
