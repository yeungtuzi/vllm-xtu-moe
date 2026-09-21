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

## Open items

1. **Why is the 2-D form 2.8x faster?** Unexplained. Worth understanding before shipping.
2. **End-to-end numerics** were not checked — the standalone harness compared kernels, the
   end-to-end run only measured time. A quality check is still owed.
3. `BLOCK_H` should be 2 or 8; 4 is a local optimum to avoid and 16 is bad. `BLOCK_H=8` is
   what the patch uses.
4. The patch also depends on a detail worth re-checking: the wrapper zeroes heads past
   `num_heads`, and the new kernel writes them via `head_mask` instead. That path was not
   exercised in my tests (H was always a multiple of BLOCK_H).

## Appendix: the superseded head-blocking rationale

The original version of this file argued the change was about KV traffic (2062 GB read
against 32.2 GB needed, 64x). The arithmetic was right; the causal claim was wrong. Kept here
only so the record shows what was believed before the measurement.
