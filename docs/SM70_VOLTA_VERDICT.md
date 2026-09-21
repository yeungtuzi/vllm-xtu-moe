# SM70 / Volta compatibility verdict — GLM-5.3-Flash + sparse MLA

**Question:** can the profiled pipeline (GLM-5.3-Flash `Glm5NextForConditionalGeneration`,
sparse MLA + sink kernel `_sparse_mla_fwd_with_sink_kernel`) move from 2×A100-40GB (SM80)
to 8×V100-SXM2-32GB (SM70 / Volta)?

**Short answer: No.** vLLM in this tree does not route *this* model on SM70 at all. The
failure is by explicit source-level gating, not by a missing build flag. The single most
decisive piece of evidence is:

```python
# vllm/v1/attention/backends/mla/flashmla_sparse_sm8x.py:152-153
@classmethod
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    return capability.major == 8
```

with the caller binding that backend only when the device family is 8:

```python
# vllm/models/glm5next/nvidia/attention.py:538-543
if sm8x_sparse_mla_enabled() and sm8x_kv_ok:
    attn_backend = FlashMLASparseSM8XBackend
```

On SM70 the model therefore falls into the generic candidate pool, where **every** sparse-MLA
backend requires SM90+/SM100/SM120, so `cuda.py` raises
`ValueError: No valid attention backend found ...` (`vllm/platforms/cuda.py:499-503`).
Before that even happens, the bf16 dtype check aborts the worker on any card below cc 8.0.

Inspected trees (2026-09 vintage):
- upstream reference: `/home/user/lvllm/process_data/ref/repos/vllm-mainline`
  (git tag `v0.29.1rc0-105-gaf3e7c14d7`, 2026-09-18)
- project fork: `/home/user/lvllm/Lvllm` (tag `lvllm-v2.5.0`, base upstream commit `71888f507a`, 2026-09-14)

---

## 1. Does vLLM support SM70 / Volta at all?

**Verdict: not officially. The documented floor is SM75 (Turing).** SM70 survives only in a
legacy CMake fallback and, per external evidence, in community/self-built vLLM 0.x installs
running fp16 models. It does not help this model.

Verified from source:

| Evidence | File / line |
|---|---|
| Docs: **"GPU: compute capability 7.5 or higher (e.g., T4, RTX20xx, A100, ...)"** | `docs/getting_started/installation/gpu.cuda.inc.md:9` |
| Build arch lists for CUDA ≥ 12.8 / 13.0 / 13.4 start at **7.5** (`"7.5;8.0;8.6;..."`); only the `< 12.8` fallback still lists `7.0` | `CMakeLists.txt:118-131` |
| Requested archs are intersected with the supported list; an empty result is a hard build error | `CMakeLists.txt:226-240` (`FATAL_ERROR "No supported CUDA architectures"`) |
| Runtime dtype table treats Volta as fp16/fp32 only, comment **"Pascal, Volta and Turing NVIDIA GPUs, BF16 is not supported"** | `vllm/platforms/cuda.py:253-262` (identical in `Lvllm/vllm/platforms/cuda.py:253-262`) |
| bf16 is a hard error below cc 8.0 (see §3) | `vllm/platforms/cuda.py:666-684` |
| Docs still carry a per-feature **Volta** column, but that is feature-level (LoRA, CUDA graph, …), not a model-support matrix | `docs/features/README.md:62` |

- The default/prebuilt wheels in this tree are **CUDA 12.9** (and a `cu130` variant)
  (`docs/getting_started/installation/gpu.cuda.inc.md:3,54,132`). CUDA ≥ 12.8 arch lists do
  not contain `7.0`, so the shipped binaries have no SM70 cubins.
- The project's own release notes list the supported matrix as
  `SM80 | SM86 | SM89 | SM90 | SM100 | SM120` with **no SM70/SM75 columns at all**, and mark
  `GLM-5.3-Flash | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ fixed here`
  (`Lvllm/RELEASE_NOTES.md:5-17`). Reference hardware is RTX 3090 (SM86) and RTX 5060 Ti
  (SM120); the SM8x Triton path is described as what lets "SM86/SM89 reach a bf16 KV cache"
  (`Lvllm/RELEASE_NOTES.md:96-100`).

External (real-world, not this repo — cite as supporting, unverified here):
- CUDA 13.0 removed sm_70 offline compilation; community V100 builds pin CUDA 12.6.3
  ([nvidia-v100-ai-toolboxes](https://raw.githubusercontent.com/kyuz0/nvidia-v100-ai-toolboxes/main/README.md)).
- A community effort runs **vLLM 0.18.1 / 0.21.0 on 8×V100-SXM2-32GB with `--dtype half` +
  CUDA 12.6 + Triton 3.6**, and states upstream "declines sm_70-support PRs by policy"
  ([vllm-fp8-w8a16-sm70 / VOLTA_MOE_UPSTREAM.md](https://raw.githubusercontent.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/main/docs/VOLTA_MOE_UPSTREAM.md)).

**Inference (general knowledge, consistent with the above):** PyTorch `2.13.0+cu130`
(the pinned test/dep version, `requirements/test/cuda.txt:1228`) will not carry sm_70
kernels because CUDA 13 dropped that target. I could not inspect the wheel here.

---

## 2. Does the sparse MLA attention path work on SM70?

**Verdict: No — by explicit gating, and there is no SM70 fallback.**

### What the profiled kernel actually is

`_sparse_mla_fwd_with_sink_kernel` is a **Triton** JIT function, not CUDA/CUTLASS:

- Definition: `vllm/v1/attention/backends/mla/sparse_mla_kernels.py:3520-3597` (`@triton.jit`).
- Wrapper: `sparse_mla_fwd_with_sink` at `sparse_mla_kernels.py:3600-3665`; docstring says
  *"written in portable Triton so it runs on SM8x. Ported from the LvLLM SM8x work, itself
  ported from the SGLang SM80 effort."*
- The kernel body uses only `tl.load / tl.sum / tl.exp / tl.where / tl.maximum / tl.arange /
  tl.store` — it contains **no `tl.dot`, no explicit `cp.async`, no fp8 types**. The only
  `tl.dot` calls in that file are in *other*, DeepSeek-V4 fp8 kernels (lines 1740, 1869), not
  in this one.
- It is reached from `FlashMLASparseSM8XImpl._bf16_flash_mla_kernel`
  (`vllm/v1/attention/backends/mla/flashmla_sparse_sm8x.py:151-176`), which exists precisely to
  *replace* the SM90+ `flash_mla_sparse_fwd` CUDA kernel on Ampere/Ada.

So, at the source level, this particular kernel is not the blocker. The blockers are the
platform gates around it.

### The SM8x gate

`vllm/v1/attention/backends/mla/flashmla_sparse_sm8x.py`:

```python
# lines 51-62
def sm8x_sparse_mla_enabled() -> bool:
    if not current_platform.is_cuda():
        return False
    return current_platform.is_device_capability_family(80)   # major == 8 ONLY

# line 129
supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

# lines 152-153
@classmethod
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    return capability.major == 8

# line 223
if not sm8x_sparse_mla_enabled():
    return "FLASHMLA_SPARSE_SM8X requires an SM8x device"
```

`is_device_capability_family` compares `to_int() // 10` (`vllm/platforms/interface.py:480-493`),
so family 80 means major 8 exactly. Major 7 returns `False`.

The parallel helper for the DeepSeek-V4 Triton fallback excludes Volta for the same reason:

```python
# vllm/v1/attention/backends/mla/sparse_mla_env.py:22-26, 41-45
def is_ampere_or_ada() -> bool:
    return current_platform.is_cuda() and current_platform.is_device_capability_family(80)
def is_triton_sparse_mla_enabled_for_platform() -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120) or is_ampere_or_ada()
```

### What happens on a non-SM8x device

In `vllm/models/glm5next/nvidia/attention.py:522-547`, the sparse layer sets
`attn_backend = FlashMLASparseSM8XBackend` only under `sm8x_sparse_mla_enabled()`; otherwise
`attn_backend` stays `None`, and `MLAAttention` calls the generic selector with
`use_mla=True, use_sparse=True` (`vllm/model_executor/layers/attention/mla_attention.py:496-503`).

The generic candidate pool has no major-7 branch. For `use_mla` and major ∉ {10, 12} it is
(`vllm/platforms/cuda.py:95-153`):

```
FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA,
FLASH_ATTN_MLA_SPARSE, FLASHMLA_SPARSE, FLASHINFER_MLA_SPARSE_SM90
```

Every sparse entry rejects SM70:

| Backend | Guard | File:line |
|---|---|---|
| `FlashAttnMLASparseBackend` | `return capability.major == 9` | `mla/flashattn_mla_sparse.py:71-72` |
| `FlashMLASparseBackend` | `return capability.major in [9, 10]` | `mla/flashmla_sparse.py:156-157` |
| `FlashInferMLASparseSM90Backend` | `return capability.major == 9` | `mla/flashinfer_mla_sparse_sm90.py:106-107` |
| `FlashInferMLASparseTRTLLMBackend` | `return capability.major == 10` | `mla/flashinfer_mla_sparse.py:97-98` |
| `FlashInferMLASparseSM120Backend` | `return capability.major == 12` | `mla/flashinfer_mla_sparse.py:176-177` |

`TritonMLABackend.supports_compute_capability` does return `True`
(`mla/triton_mla.py:171-172`), **but it is not a sparse backend**: `is_sparse()` is the base
default `False` (`vllm/v1/attention/backend.py:177-178`), and
`validate_configuration` rejects on `use_sparse != cls.is_sparse()` with
`"sparse not supported"` (`vllm/v1/attention/backend.py:306-311`). So it cannot serve the DSA
(indexer) layers.

With zero valid candidates, the selector raises:

```python
# vllm/platforms/cuda.py:499-503
if len(valid_backends_priorities) == 0:
    raise ValueError(
        f"No valid attention backend found for {cls.device_name} "
        f"with {config_str}. Reasons: {reasons_str}."
    )
```

This is exactly the situation the SM8x file's docstring describes for sm8x:
*"whose FlashMLA/FlashInfer kernels require SM90+. On an sm8x card that leaves the platform
candidate pool empty (`cuda.py` raises)."* (lines 11-17) — and there is no equivalent rescue
backend for sm70.

### `TRITON_ATTN_DIFFKV` is *not* this path's fallback

The project note that SM80 needed "`TRITON_ATTN_DIFFKV`" is about **MiMo-V2.x**, a *non-MLA*
model with different K/V head dims:

```python
# vllm/model_executor/models/mimo_v2.py:296-314
# Use DiffKV backend when V has a different head dim than K.
# Auto-pick FA-DiffKV when FA3/4 is usable on this device, else fall
# back to TRITON_ATTN_DIFFKV.
```

`TritonAttentionDiffKVBackend` extends `TritonAttentionBackend`, which accepts any compute
capability (`triton_attn.py:377-378`), so it is *conceivably* SM70-capable for MiMo. But it
is a plain attention backend (`is_mla = False`), not an MLA backend, and it is never selected
for GLM-5.3. **Do not use it as evidence that the sparse-MLA path has a Volta fallback.**

### FlashMLA is not even built for SM70

`cmake/external_projects/flashmla.cmake` only compiles for `9.0a`, `10.0f`/`10.0a`, `10.7f`
(lines 55-71), and logs `"FlashMLA will not compile: unsupported CUDA architecture"`
(line 199). The Python import shim reports unavailability with the message *"likely was not
compiled ... or a supported arch was not in the list of target arches"*
(`vllm/v1/attention/ops/flashmla.py:33-39`).

---

## 3. Actual list of SM80+ features this path relies on

There are two layers here; be careful not to conflate them.

### (a) The profiled kernel itself: **no SM80-only instruction**

`_sparse_mla_fwd_with_sink_kernel` (Triton, `sparse_mla_kernels.py:3520-3597`) uses only
loads/reductions/elementwise math. No `cp.async` (source), no `tl.dot`, no `wgmma`, no TMA,
no fp8. Triton 3.6 still targets sm_70 (`triton/backends/nvidia/compiler.py:215` selects
`cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70`).
So **the 58.8 % kernel is not kept off Volta by an instruction that Volta lacks** — it is kept
off by vLLM's routing gate and by the rest of the stack.

### (b) The rest of the model path: SM80+ and above, verified from source

| Requirement | Where | Quote |
|---|---|---|
| **BF16** (Volta has none) | `vllm/platforms/cuda.py:666-684`, called from `vllm/v1/worker/gpu_worker.py:426` | `raise ValueError("Bfloat16 is only supported on GPUs with compute capability of at least 8.0. ...")` |
| BF16 type conversion stub below 800 | `csrc/libtorch_stable/type_convert.cuh:72-73` | `#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) ...` `// CUDA_ARCH < 800 does not have BF16 support` |
| Fused QK-norm/RoPE MLA kernel compiled out below 800 | `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:117,314,330,557` | `#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)` |
| That same kernel uses **`cp.async`** (SM80+) | `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:400-402,683-685` | comments: "`cp.async` reads", "uses `cp.async` to load cos/sin in 16-byte chunks" |
| DeepSeek-V4/GLM fused qnorm-rope-KV kernel: no-op stub for sm_70/sm_75 **and** a runtime `sm_80+` refusal | `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:425-428, 608-614` | `// no-op stub for sm_70/sm_75 to keep multi-arch builds happy.` … `STD_TORCH_CHECK(sm_version >= 80, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert requires sm_80+ (Ampere or newer); got sm_", sm_version);` |
| **`cp.async`** in KDA decode | `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu:86-100` | inline `asm volatile("cp.async.cg.shared.global ...")`, `cp.async.commit_group/wait_group` |
| **`cp.async.bulk` (SM90+ TMA-style)** | `csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu:226,242` | `"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes ..."`, guarded by `__CUDA_ARCH__ >= 1000` |
| **FlashMLA / FlashInfer sparse / FlashAttn-MLA** | see §2 table; `flashmla_sparse_sm8x.py:11-17` docstring | *"FlashMLA/FlashInfer kernels require SM90+"* |
| **Marlin WNA16 for MXFP4/NVFP4** — only SM75+ | `vllm/model_executor/layers/quantization/modelopt.py:1562-1564`; `.../linear/scaled_mm/marlin.py:43` | "Turing and up (SM75+): NVFP4 routed experts run via Marlin W4A16 (SM75+)"; "FP8 Marlin requires compute capability 7.5 or higher" |
| **FP8 tensor cores** (SM89+; used elsewhere in the tree) | `docs/features/quantization/llm_compressor/fp8.md:16` | "FP8 computation is supported on NVIDIA GPUs with compute capability >= 8.9" |
| **DeepGEMM** (Hopper/Blackwell only) | `vllm/platforms/cuda.py:718-725` | `support_deep_gemm()` → `is_device_capability(90) or family(100) or family(120)` |

`wgmma` (SM90+) and `__CUDA_ARCH__ >= 900`/`>= 1000` guards appear throughout the CUDA/CUTLASS
kernels the path would otherwise use (`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:452,482`,
`kimi_k3/fused_kda_decode_kernel.cu:299,324`, `kimi_k3/attn_res_kernel.cu:242`), but not in the
Triton sparse-MLA kernel itself.

**Summary:** the *single* profiled kernel is portable Triton; the *pipeline* is SM80+ at
minimum because of bf16, the fused CUDA MLA helper kernels (with explicit sm_70 stubs), and —
for the sparse MLA specifically — SM8x-or-SM90+ backend selection. There is no `cp.async`
dependency *inside* `_sparse_mla_fwd_with_sink_kernel`; the "Volta lacks `cp.async`" argument
in the existing handoff note is true as a hardware fact but does not by itself explain why
this kernel cannot run on Volta.

---

## 4. Practical alternative stack for Volta

**Verdict: a Volta route exists for *some* model, but not for GLM-5.3-Flash on vLLM in this
tree.** Options, in decreasing practicality:

1. **llama.cpp / ggml (CUDA backend).** External evidence: llama.cpp builds for
   `-DCMAKE_CUDA_ARCHITECTURES=70`; its custom Flash-Attention kernels "DO support Volta sm_70";
   use CUDA 12.6.3 and `--dtype half`/fp16 (no BF16)
   ([nvidia-v100-ai-toolboxes](https://raw.githubusercontent.com/kyuz0/nvidia-v100-ai-toolboxes/main/README.md)).
   This is the only *documented* stack that runs on V100 in this era. **Caveat (inference):**
   it requires a GGUF conversion of GLM-5.3-Flash (320B MoE / ~18B active) to exist; none was
   found in the workspace. Quantized CPU/GPU-offload inference of a 320B MoE on 8×32 GB V100
   is a low-throughput regime; the toolbox benchmarks show single-digit-to-low-tens tok/s for
   large MoEs on V100.
2. **Older vLLM (0.x) on Volta.** External evidence: vLLM 0.18.1 and 0.21.0 run on
   8×V100-SXM2-32GB (CUDA 12.6, Triton 3.6, fp16) — but upstream declines sm_70 PRs
   ([vllm-fp8-w8a16-sm70](https://raw.githubusercontent.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/main/docs/VOLTA_MOE_UPSTREAM.md)).
   **Does not help here:** those releases predate `Glm5NextForConditionalGeneration`; the
   model/support code only exists in the newer tree that dropped the arch. Even in an old
   vLLM, a Triton sm_70 MoE GEMM is reported "~40× off the memory-bandwidth floor (no
   tensor-core `tl.dot` path on Volta)".
3. **Hand-written / ported kernels on top of this tree.** Theoretically possible, but see §5.

---

## 5. If SM70 support were to be pursued, what would have to change

This is *not* a supported configuration and would be a fork, not a flag:

1. Relax `sm8x_sparse_mla_enabled()` / `FlashMLASparseSM8XBackend.supports_compute_capability`
   to admit major 7, and make the backend accept `torch.float16` + an fp16 KV cache
   (currently `[torch.bfloat16]` / `["auto","bfloat16"]`,
   `flashmla_sparse_sm8x.py:129-130`). The kernel hardcodes bf16 KV rows
   (`_bf16_flash_mla_kernel`) and `_sink_buf` is fp32 — the arithmetic would still work in
   fp16, but the kernel binding/type plumbing would need a real change.
2. Bypass the bf16 dtype gate (`cuda.py:666-684`) by loading the model as fp16 — but the
   checkpoint is bf16/NVFP4, and the MoE expert GEMM fallback (Marlin WNA16) needs SM75+, not
   SM70. This is a second independent wall.
3. Recompile/port the fused CUDA helpers that currently compile to no-op stubs below
   `__CUDA_ARCH__ 800` (`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`,
   `fused_qknorm_rope_kernel.cu`, `type_convert.cuh`) — or route around them with torch ops.
4. The KDA / linear-attention half of the hybrid model (`kda.py:_resolve_kda_prefill_backend`
   accepts only `capability.major in (9,10,12)` for FlashKDA, `kda.py:136-144`) would fall to
   the Triton path; that path uses `tl.dot` and may need sm_70 validation.
5. There is no evidence anywhere in the tree of an SM70 MLA attention implementation to port
   from. The `TRITON_ATTN_DIFFKV` backend is a non-MLA MiMo path and cannot be repurposed as
   one.

---

## Verified vs inferred

**Verified from source in this workspace:**
- Documented floor SM75 and the CUDA≥12.8 CMake arch lists; empty-intersection build error.
- `sm8x_sparse_mla_enabled` / `capability.major == 8` gates; bf16-only dtype/KV for the SM8x
  sparse backend; GLM5Next binding it only on SM8x.
- The full sparse-MLA candidate pool and every backend's capability guard; the
  `"No valid attention backend found"` raise.
- `TritonMLABackend` is non-sparse and therefore rejected for DSA layers.
- `_sparse_mla_fwd_with_sink_kernel` is Triton with no `tl.dot`/fp8/cp.async in source.
- The bf16 `check_if_supports_dtype` raise and its call site.
- All quoted `__CUDA_ARCH__` guards and `STD_TORCH_CHECK(sm_version >= 80)` in `csrc/`.
- FlashMLA CMake only builds for 9.0a/10.0a/10.0f/10.7f.
- The project release-note matrix omits SM70/SM75 and marks GLM-5.3 as SM80+.

**Inferred (general knowledge / external sources, not executed here):**
- Volta has no `cp.async`, no BF16 tensor cores, no FP8 tensor cores, no `wgmma`, no TMA —
  consistent with the in-tree comments but not proven by this repo.
- CUDA 13.0 removed sm_70; PyTorch `2.13.0+cu130` therefore has no Volta kernels
  (the pinned Triton 3.7.1 was not present locally; I inspected Triton 3.6.0, which still
  targets sm_70).
- Actual runtime failure on a V100: derived from control flow, not observed — no Volta
  hardware was used.
- llama.cpp's Volta viability and the absence of a GLM-5.3 GGUF: external / workspace search,
  not a build test.

**Decisive single fact:** `FlashMLASparseSM8XBackend.supports_compute_capability` returns
`capability.major == 8`, and the GLM-5.3 sparse layers are only bound to that backend when
`sm8x_sparse_mla_enabled()` is true; on SM70 the generic pool then has zero valid sparse-MLA
candidates and vLLM raises.
