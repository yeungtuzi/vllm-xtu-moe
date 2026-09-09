## Goal

Let mainline DeepSeek-V4 **build and run on SM 8.x (A100 / A800)** without forking, by adding
portable Triton fallbacks for the parts that are SM90+/SM100-only today. Existing SM90 / SM100 /
SM110 paths are untouched — every change is capability-gated.

## Why this is needed

DeepSeek-V4 in mainline assumes SM90+: sparse MLA goes through FlashMLA/DeepGEMM, fp8 einsum
through DeepGEMM, MHC prenorm through tilelang/cutedsl, and several ops use the `float8e4nv`
Triton dtype. On SM 8.0 all of those are unavailable, so the model cannot even be constructed.
Ampere is still a very large installed base (A100/A800), so a portable fallback path is worth
having even at lower absolute performance.

## Design

Two layers, both selected at runtime by compute capability:

1. **Capability gates** (no behaviour change on SM90+):
   - `utils/import_utils.py`: `has_cutedsl()` returns False on SM8.x (cutedsl only emits SM90+ PTX).
   - `utils/deep_gemm.py`: `_use_sm12x_mqa_fallback()` extended to SM8.x; SM80 branches for
     `fp8_fp4_mqa_logits` / `paged` / `tf32_hc_prenorm_gemm`.
   - `compressor.py`, `sparse_mla.py`, `mhc/tilelang.py`, `flashinfer_sparse_mla_warmup.py`,
     `mla/{indexer,sparse_swa}.py`: skip cutedsl / DeepGEMM scheduler metadata / FlashInfer
     warmup where the arch cannot run them.
2. **Portable Triton implementations**:
   - `v1/attention/backends/mla/sparse_mla_env.py` (119 lines) + `sparse_mla_kernels.py`
     (3517 lines): the sparse-MLA decode/prefill kernels and the env switches
     (`VLLM_TRITON_MLA_SPARSE`, `..._TOPK_CHUNK_SIZE`, `..._QUERY_CHUNK_SIZE`,
     `..._HEAD_BLOCK_SIZE`, `..._MATMUL_DECODE`).
   - `models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` (711) and `sm12x_mqa.py` (756):
     DeepGEMM / MQA fallbacks for SM8.x and SM12x.
   - `common/ops/*`: uint8-based fp8 conversion instead of `tl.float8e4nv`
     (`_f32_to_e4m3_uint8` / `_e4m3_uint8_to_f32`), plus the cache helpers the portable
     decode/prefill path needs (`dequantize_global_slots_k_cache`,
     `dequantize_combined_sparse_mla_decode_kv`, `sparse_prefill_combined_topk_size`).
   - `models/deepseek_v4/nvidia/flashmla.py`: SM8.x dispatch to the Triton path.

## Algorithm notes

- **Sparse MLA**: per query, gather the top-k selected KV slots (window + compressed + top-k
  indexer selection), then a flash-style online-softmax attention over those slots, with the
  head dim (512) split for the Triton tile. The portable kernels keep the same math as the
  SM90 FlashMLA path; they differ only in tiling and in decoding fp8/uint8 scales in software.
- **Indexer**: the scoring pass (`_indexed_d512_split_score/value`) is kept as-is; only the
  fp8 encode/decode changes to uint8 on Ampere.
- **fp8 einsum**: block-FP8 with 128×128 weight blocks and 128-wide activation blocks; scales are
  expanded to element-wise factors and folded into the fp32 accumulator (see #56119).

## Measured on A100 (SM 8.0), DeepSeek-V4-Flash, 2× A100-PCIE-40GB

The point of the PR is that the model runs at all; these are the numbers we measured once it did
(full report + raw data: <https://github.com/yeungtuzi/vllm-xtu-moe>).

**Prefill, single A100** (CPU experts vs layerwise GPU weight streaming):

| prompt tokens | CPU prefill | GPU prefill | speed-up |
|---|---:|---:|---:|
| 2 091 | 112.9 s | **5.6 s** | 20× |
| 4 137 | 439.4 s | **5.8 s** | 76× |
| 8 229 | – | **8.6 s** | – |
| 16 039 | – | **17.7 s** | – |

**Prefill, 2× A100 (TP=2 + expert parallel)**: 16 039 tokens in **13.96 s** (1 149 tok/s).

**Kernel-time breakdown of a 16 K prefill** (in-process torch profiler, per pass):
MoE kernels 31 %, elementwise kernels 29 %, H2D 19 %, dense GEMM 18 %,
**attention + indexer 2.9 %**.

**Numerics**: the portable kernels match a pure-torch reference to bf16 output precision
(RMS relative error 5.6e-3).

## Files taken from a fork

All new files and the adapted `flashmla.py` / `o_proj.py` come from the vLLM fork
[Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x) (Apache-2.0, author guqiong96). They are
byte-for-byte identical except for mainline API adaptations (e.g. `combine_topk_swa_indices` now
returns `(idx, lens)`). SPDX headers are retained. Debug instrumentation from our development
branch has been removed for this PR.

## Testing

- A100-PCIE-40GB (SM 8.0) + DeepSeek-V4-Flash: model builds, generates coherent text; the
  measurements above come from this configuration.
- SM90/SM100/SM110 paths unchanged (capability gates only).
- 21 files, +6 350/−119.

## Rebase note (2026-09-09)

Rebased onto `main@1454b71`. Upstream refactored three files we also touch
(`common/ops/{cache_utils,fused_indexer_q,fused_inv_rope_fp8_quant}.py` — new
`VllmTritonJitKernel`/`LaunchSpec` warmup structure), so our changes there were re-applied on top
of the new structure rather than merged verbatim: fp8 encode/decode via
`_f32_to_e4m3_uint8` / `_e4m3_uint8_to_f32`, `has_cutedsl()` gated with
`and not is_ampere_or_ada()`, and the ported SM80 kernels appended at the end of
`cache_utils.py`.

Verification so far: every touched file parses; the patch applies to `main@1454b71` with no
conflict markers. **A full A100 runtime validation against this exact rebased tree is still
pending** — I will report the result here before marking the PR ready for review.

## Related

- #56118 (CPU-offloaded experts), #56119 (o_proj fp8 correctness — required for correct output
  on SM8.x), and an RFC for layerwise GPU prefill of CPU-offloaded experts.
