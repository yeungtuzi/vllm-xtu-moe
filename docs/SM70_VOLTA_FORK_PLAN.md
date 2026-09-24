# Fork plan: GLM-5.3-Flash + DeepSeek-V4.1-Flash on 10×V100-16GB (SM70) with CPU decode

**Target hardware (user):** Intel Xeon Gold 6254 box + 10× Tesla V100 16GB **PCIe (no NVLink)**;
plus many free CPU-only servers for decode, all on 100 Gbps InfiniBand RDMA.
**Goal:** run `Glm5NextForConditionalGeneration` (GLM-5.3-Flash) and DeepSeek-V4.1-Flash,
forking vLLM as needed.

This document is the engineering assessment and staged plan. It builds on
`docs/SM70_VOLTA_VERDICT.md` (which established that stock vLLM refuses SM70 on these models).

---

## 0. Verdict up front

**It can be made to run, but not by flipping a flag, and the two models are very different
amounts of work.** The good news is large: the hardest part of the problem — serving a
300–750B MoE on small GPUs — is already solved by this project's plugin, and that part is
CPU-ISA based (AVX2/AVX-512), so it is *architecturally blind to Volta*. The work is almost
entirely in the **GPU non-expert path** (attention / indexer / dense + shared experts) and in
the **toolchain / dtype stack**.

| Workstream | Size | Blocking? |
|---|---|---|
| **W0. Toolchain with SM70** (torch/CUDA pin) | Medium — dependency archaeology | **Yes, gates everything** |
| **W1. Attention + indexer + linear-attn on SM70** (both models) | Large | Yes |
| **W2. FP8 → FP16 for non-expert GPU layers** | Medium–Large | Yes |
| **W3. GPU prefill MoE on Volta** | Large once you want tensor cores (Triton gives FMA only) | Only for long-prompt prefill |
| **W4. 16GB / PCIe parallelism & KV budget** | Design + config | Yes for context length |
| **W5. Pure-CPU decode (your stated PD shape)** | Very large for both models | Only if decode is GPU-less |

**Single most important performance finding (reproduced):** on sm_70, **Triton emits no
tensor-core MMA at all** — every `tl.dot` lowers to scalar `fma.rn.f32`, even for pure
fp16×fp16. I compiled a minimal Triton dot for `GPUTarget("cuda", 70/75/80, 32)` with the
project's pinned env (Triton 3.7.1) and disassembled: cap70/cap75 → `mma_layout=False`,
`ptx_mma=False`, `ptx_fma=True`; cap80 → `mma_layout=True`, `ptx_mma=True`. (Triton 3.6
behaves the same in practice — the community V100 work reports "no tensor-core `tl.dot` path on
Volta".) So **all of vLLM's Triton compute kernels — MoE prefill, the sparse indexer score
kernel, and GLM's KDA linear-attention kernels — run on CUDA cores on a V100.** Casting bf16 to
fp16 does not fix this; only a hand-written/CUTLASS fp16 HMMA GEMM does. V100's 125 TFLOPS fp16
tensor-core number is unreachable through this code path.

**Second most important conclusion:** *pure-CPU decode is not available for either target
model in this tree.* DeepSeek-**V4** has a CPU implementation (`vllm/models/deepseek_v4/cpu/`,
~1900 lines + `current_platform.is_cpu()` dispatch in `vllm/models/deepseek_v4/__init__.py:33`),
but **V4.1 does not** (`vllm/models/deepseek_v41/__init__.py` only branches on ROCm vs NVIDIA),
and **GLM-5.3 has no CPU branch at all** (`vllm/models/glm5next/__init__.py` unconditionally
imports `.nvidia.model`). Worse, the CPU platform itself rejects sparse attention outright:
`vllm/platforms/cpu.py:144-145` → `raise NotImplementedError("Sparse Attention is not supported
on CPU.")`, and GLM-5.3 sets `is_sparse = config.index_topk is not None`
(`vllm/models/glm5next/nvidia/attention.py:519,590`). A GPU-less decode fleet is therefore a
from-scratch CPU attention port for GLM (KDA + sparse MLA + indexer) plus a port of the V4 CPU
path to V4.1 — and the V4 CPU sparse kernels are bf16/AMX-only (`csrc/cpu/sgl-kernels/*`, built
only into the AMX `_C` target and loaded only when `_is_avx512_bf16_supported()`,
`vllm/platforms/cpu.py:603-611`) — a Xeon Gold 6254 has AVX-512 but **no AVX512-BF16**, so those
kernels would not even load.

**Recommended pragmatic shape:** keep **one V100 per decode instance** for attention, and put
only the MoE experts on the CPU servers. That is exactly the architecture the plugin already
implements, needs no new CPU kernels, and still lets the free CPU fleet absorb the MoE cost.
PD separation then becomes GPU↔GPU KV transfer with CPU MoE on both sides.

---

## 1. W0 — the toolchain is the first wall (do this before any kernel work)

The 2026 mainline pins `torch == 2.13.0` (`pyproject.toml:10`, `requirements/cuda.txt:7`) and
ships cu129/cu130 wheels. **PyTorch deleted sm_70 for CUDA 12.8/12.9 builds**
([pytorch#157517](https://github.com/pytorch/pytorch/issues/157517)) and CUDA 13 removed Volta
entirely; a separate RFC deprecates sm70 for CUDA 12.8 in 2.11
([pytorch#172352](https://github.com/pytorch/pytorch/issues/172352)). So **the pinned torch has
no Volta kernels**, and nothing you do in vLLM changes that.

Verified in-tree facts that follow:
- `CMakeLists.txt:118-131`: for CUDA ≥ 12.8 the supported arch list starts at **7.5**; only the
  `< 12.8` fallback still contains `7.0`. The requested archs are intersected with this list and
  an empty result is `FATAL_ERROR` (`CMakeLists.txt:226-240`).
- The community V100 stack proves the working combination: **CUDA 12.6.3 + Triton 3.6 +
  fp16** (no bf16), where CUDA 12.6.3 is "the last CUDA release with full, non-deprecated sm_70
  support" ([nvidia-v100-ai-toolboxes](https://raw.githubusercontent.com/kyuz0/nvidia-v100-ai-toolboxes/main/README.md)).

**Action:** pin CUDA 12.6/12.7 and a torch build that carries sm_70, re-add `7.0` to
`CUDA_SUPPORTED_ARCHS`, and build with `TORCH_CUDA_ARCH_LIST=7.0`. Expect dependency
friction because vLLM 2.x code may assume newer torch APIs. If a torch 2.13+cu126 wheel with
sm_70 does not exist, the fallback is a source build of torch for sm_70 — budget for that.

Also required in CMake: FlashMLA (`cmake/external_projects/flashmla.cmake:55-71`) only builds
for `9.0a/10.x`, which is fine — it will simply be absent, and the SM70 path must not depend
on it.

---

## 2. What already exists (the foundation you are extending)

The plugin's design already separates the problem correctly:
- **CPU**: expert weights + expert compute, AVX2/AVX-512 (no AMX needed; your Xeon 6254 has
  AVX-512). `xiaotu_moe/csrc/` is pure CPU C++/headers — **no SM70 work at all**.
- **GPU**: attention / KV / router / dense / shared experts, plus a *layerwise GPU prefill*
  path that streams one layer of expert weights H2D and computes MoE in Triton
  (`vllm_xiaotu_moe/gpu_prefill.py`, `gpu_prefill_fp8.py`).

The upstream **SM80 port already exists as a patch** and is the template for SM70:
`patches/upstream/pr3-sm80-port.patch` (21 files) adds portable Triton fallbacks for sparse MLA,
MQA/DeepGEMM, fp8 einsum and MHC, and gates them by compute capability. Its own description:
*"Let mainline DeepSeek-V4 build and run on SM 8.x (A100/A800) without forking, by adding
portable Triton fallbacks for the parts that are SM90+/SM100-only today. … every change is
capability-gated."* The same pattern extends to SM70, but the *contents* differ (see §3).

---

## 3. W1 — the GPU attention/indexer port (the real project)

### 3.1 Extend the capability gates (easy)

The SM8x sparse-MLA path is explicitly major-8-only:

```python
# vllm/v1/attention/backends/mla/flashmla_sparse_sm8x.py:51-62
def sm8x_sparse_mla_enabled() -> bool:
    ...
    return current_platform.is_device_capability_family(80)   # major == 8 ONLY
# :152-153
def supports_compute_capability(cls, capability) -> bool:
    return capability.major == 8
```

and the DeepSeek path:

```python
# vllm/v1/attention/backends/mla/sparse_mla_env.py:41-45
return current_platform.is_device_capability_family(120) or is_ampere_or_ada()
# vllm/utils/deep_gemm.py:817-825  _use_sm12x_mqa_fallback(): family(120) or family(80)
```

GLM binds the backend at `vllm/models/glm5next/nvidia/attention.py:538-543`; V4.1 dispatches in
`vllm/models/deepseek_v41/nvidia/flashmla.py` via `is_triton_sparse_mla_enabled(q.device)`.
All of these predicates need a `7.x` arm. Mechanically small, but it only gets you to the next
blocker.

### 3.2 bf16 → fp16 is unavoidable and cross-cutting

- vLLM hard-fails bf16 below cc 8.0: `vllm/platforms/cuda.py:666-684`, called from
  `vllm/v1/worker/gpu_worker.py:426`.
- The SM8x sparse backend is bf16-only:
  `flashmla_sparse_sm8x.py:129-130` → `supported_dtypes = [torch.bfloat16]`,
  `supported_kv_cache_dtypes = ["auto", "bfloat16"]`, and the kernel is literally named
  `_bf16_flash_mla_kernel`.
- V100 has **no bf16 tensor cores**; only FP16 (125 TFLOPS).

So the fork must introduce an **fp16 mode**: model dtype fp16, fp16 KV cache, and fp16 kernels.
The good news: the profiled Triton kernel `_sparse_mla_fwd_with_sink_kernel`
(`sparse_mla_kernels.py:3520-3597`) is dtype-generic — it loads and casts to fp32, and uses no
`tl.dot`/`cp.async`/fp8. Porting *that* kernel is mostly the naming and the surrounding
dtype plumbing.

### 3.3 The SM80 portable path uses TF32 — an SM80-only instruction

This is the concrete difference between "SM80 port" and "SM70 port". The portable MQA/indexer
fallback that the SM80 PR introduced is written for Ampere's TF32 tensor cores:

```
vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py:119  scores += tl.dot(q, tl.trans(k), input_precision="tf32")
vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py:429  scores += tl.dot(q, k, input_precision="tf32")
vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py:688  acc += tl.dot(x, fn, input_precision="tf32", out_dtype=tl.float32)
```

**Volta has no TF32.** These must become `input_precision="ieee"` or an explicit fp16 dot.
The GLM KDA Triton kernels are safer — `kernels.py:693` passes `DOT_PRECISION="ieee"` — but the
other `tl.dot` sites (`kernels.py:294,298,574,647,788,813,952`) need a Volta review (Triton on
Volta lowers fp16 `tl.dot` to WMMA, but has no bf16/fp8/tf32 path).

### 3.4 Fused CUDA helpers compile to no-op stubs below `__CUDA_ARCH__ 800`

These must be replaced by torch fallbacks (or reimplemented without cp.async):

| File | Guard |
|---|---|
| `csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:117,314,330` | `#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)` plus `cp.async` at `:400-402,683-685` |
| `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:425,608-614` | `// no-op stub for sm_70/sm_75` and `STD_TORCH_CHECK(sm_version >= 80, "... requires sm_80+ ...")` |
| `csrc/libtorch_stable/type_convert.cuh:72-73` | `#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)` ... `// CUDA_ARCH < 800 does not have BF16 support` |

GLM-5.3 is NoPE (`qk_rope_head_dim=0`), so it may dodge the RoPE kernels, but it still does
QK-norm and KV-insert. V4.1 has `qk_rope 64` and definitely hits them.

---

## 4. W2 — FP8 is a hard gate: vLLM's fp8 method requires SM75+

Both models are natively FP8/FP4, and vLLM enforces a minimum capability per quant method:

```python
# vllm/model_executor/layers/quantization/fp8.py:148-150
@classmethod
def get_min_capability(cls) -> int:
    return 75
```

```python
# vllm/config/vllm.py:836-841
if capability < quant_config.get_min_capability():
    raise ValueError(
        f"The quantization method {model_config.quantization} "
        "is not supported for the current GPU. Minimum "
        f"capability: {quant_config.get_min_capability()}. ...")
```

So loading an FP8 checkpoint on SM70 fails at config time. The CPU expert engine already
decodes FP8 correctly, but the **GPU non-expert layers** (attention q/kv/o projections, dense
MLP, shared experts, router) are FP8 linears that normally go through Marlin FP8 (SM75+) —
there is no SM70 fallback.

Two options:
1. **Offline dequantize** the non-expert (and shared/dense) FP8 weights to FP16 before load, and
   run the GPU as a plain fp16 model with `--dtype half`; keep the expert weights FP8 for the
   CPU engine. This is the lowest-risk route and avoids writing new fp8 kernels.
2. Lower `get_min_capability` and add an fp16 W8A16 dequant-in-kernel path. More work.

⚠️ A subtlety if you take option 1: the CPU expert engine's format registry is
**BF16 / FP8 / MXFP4 / INT4(WNA16)** (
expert backend**. So you cannot simply convert the whole checkpoint to fp16: the experts would
lose their format (and a BF16 expert copy would be ~567 GiB for GLM). The workable split is
**FP8 experts for the CPU engine + FP16 (or W8A16) non-expert weights on GPU**, which means a
selective per-tensor dequant of the non-expert weights rather than a whole-model conversion.

Note that the MXFP4/NVFP4 variants are worse: no FP4 tensor cores below SM100/120, and vLLM's
Marlin fallback needs SM75+. Stick with the FP8 checkpoint.

---

## 5. W3 — GPU prefill MoE kernels: the tensor-core problem, not just dtype

All of the plugin's GPU prefill MoE kernels are hardcoded to **bf16**. Confirmed sites:

```
vllm_xiaotu_moe/gpu_prefill_fp8.py:119,123,175,176   .to(tl.bfloat16) / tl.dot
vllm_xiaotu_moe/gpu_prefill.py:190,191,192,193,196,256,257,260,297,323,324,370,395,396,397,398
```

They already decode FP8 via pure uint8 arithmetic (`_e4m3_uint8_to_f32`, added for Ampere
precisely because "Triton has no fp8e4nv type on Ampere"), so **the FP8 decode is portable to
Volta** and `patches/upstream/pr2-fp8-sm80-o-proj.patch` is arch-neutral — usable as-is.

**But dtype is not the blocker.** I compiled all six real kernels (`_gate_up_kernel`,
`_gate_up_kernel_split`, `_down_kernel`, `_down_kernel_split`, `gate_up_kernel_fp8`,
`down_kernel_fp8`) offline with the pinned env for `GPUTarget("cuda", 70/75/80, 32)`:

- **They all compile to a valid sm_70 cubin** — no inline PTX, no `cp.async` intrinsic, no
  `tl.float8e4nv` anywhere in the plugin. So there is no source-level compile blocker.
- **At cap70/cap75 every `tl.dot` lowers to scalar `fma.rn.f32` — 0 `mma.sync` — including a
  pure fp16×fp16→fp32 dot.** Triton 3.7.1 only emits MMA at cap ≥ 80. bf16 converts are also
  emulated at sm70 (no `cvt.rn.bf16.f32`).
- **`num_stages` is inert on SM70**: `triton/backends/nvidia/compiler.py:273-293` runs
  `add_pipeline` only for `capability//10 ∈ {8,9}` or ≥10; cap 7 hits the `else`. The
  `tl.range(..., num_stages=NS)` calls therefore do nothing and there is no cp.async; measured
  shared memory at cap70 for `_gate_up` is ~16 KB, so the 96 KB limit is not binding.

Consequence: **the GPU MoE prefill path will run on the V100 but on CUDA cores, with no tensor
cores.** The A100 prefill numbers in the docs do not transfer. A fork that wants real V100
prefill must supply its own fp16 HMMA path (`mma.sync.aligned.m8n8k4` / CUTLASS sm70), or route
MoE to the CPU engine during prefill (correct but ~113 s for a 2K prompt per the project's own
measurements), or dequant to fp16 and use a cuBLAS/batched-GEMM grouped MoE instead of Triton.
Merely casting `tl.bfloat16` → `tl.float16` buys correctness, not speed.

Also note: `_GP_BACKENDS` only registers `MOE_MXFP4`/`MOE_FP8` (`mixed_experts.py:195-198`);
`MOE_BF16`/`MOE_WNA16` have no GPU prefill path (`:201-208`), and the kernels hardcode bf16
intermediates. There is **no fp16 option today**, and no SM/arch gate preventing selection on a
V100 (`gpu_prefill_min_tokens` `:60-84`, `_supports_mixed_device` `:442-443`, `_expect_dtype`
`:1052`).

Finally, a trap in the SM80 patch itself: `has_cutedsl()` is defined as
`not is_device_capability_family(80)` (`pr3-sm80-port.patch` / `mainline_sm80_mixed_mode.patch`),
which is **True on SM70** and would select SM90-only cutedsl/PTX. It must be re-gated to
`major >= 9`.

---

## 6. W4 — 10×16GB PCIe changes the parallelism and context limits

Verified KV-unit numbers from `docs/KNOWN_LIMITATIONS.md`:
- **GLM-5.3-Flash ≈ 11.9–12.3 KB/token** (only the 11 NoPE sparse-MLA layers carry per-token
  KV; each is a 512-dim latent × 2 B). 256K ≈ **3.0 GB** per rank; 1M ≈ **12 GB**.
- **DeepSeek-V4.1-Flash ≈ 40 KB/token**. 128K ≈ 5.1 GB; 256K ≈ **10.2 GB**; 512K ≈ 20 GB.

Constraints specific to your box:
- **MLA KV is replicated across TP ranks** (the latent is shared), so tensor parallelism does
  *not* buy KV memory. The SM8x sparse backend sets `supports_dcp = False`
  (`flashmla_sparse_sm8x.py`), so decode context parallelism is also unavailable without new
  work. On a 16GB card, after fp16 non-expert weights, **DeepSeek-V4.1 lands around 128–256K
  and GLM around 256–512K** — far below the 1M model limit.
- **No NVLink**: TP all-reduce crosses PCIe. Attention-only TP has small messages, but MoE is
  already on CPU, so prefer modest TP (2–4) plus expert/CPU sharding rather than TP=10.
- **GPU prefill staging** borrows VRAM: the plugin reports **7.59 GiB/rank** extra staging for
  GLM-5.3/TP=2 (process-persistent), which is why `util` must be lowered. On 16GB that is a
  serious squeeze — reduce slots or expert parallelism per rank.
- 10×16GB = 160 GB aggregate, but the model weights do not need to fit on GPU at all (MoE is
  CPU-resident); only non-expert + KV + staging do.

---

## 7. W5 — PD separation and "CPU decode"

### 7.1 What vLLM gives you

`vllm/distributed/kv_transfer/kv_connector/v1/` contains production-grade connectors
including `nixl/` (RDMA-capable), `lmcache_connector.py`, `mooncake/`, `moriio/`, `hf3fs/`,
`flexkv_connector.py`, plus CPU offloading connectors; `KVTransferConfig` supports
`kv_buffer_device`, `kv_role` (producer/consumer) and `kv_connector_module_path`
(`vllm/config/kv_transfer.py:60-96`). So **GPU-prefill → GPU-decode PV/PD over your 100 Gb IB
is supported** and is the path of least resistance.

### 7.2 What is *not* supported (corrected after re-verification)

A CUDA-prefill → **CPU-decode** transport *does* exist in vLLM, but it is narrow and does not
reach these models:

- **NIXL supports a CPU device**: `nixl/utils.py:18-29`
  `_NIXL_SUPPORTED_DEVICE = {"cuda": ("cuda","cpu"), ..., "cpu": ("cpu",)}`, with a CPU branch
  at `nixl/base_worker.py:566-583` (`use_host_buffer = False`).
- **The CPU worker initializes KV transfer** — `CPUWorker(Worker)` inherits
  `Worker.initialize_from_config`, which calls `ensure_kv_transfer_initialized(...)`
  (`vllm/v1/worker/gpu_worker.py:752`). (The call site looks GPU-only but lives in the shared
  base class.)
- **There is an explicit heterogeneous receive path**, `nixl/base_worker.py:2315-2321`:
  `if (nixl_agent_meta.attn_backend_name != self.backend_name and self.backend_name in ["CPU_ATTN"]): ... enable_heterogeneous_attn_post_process = True`,
  applied via `current_platform.pack_kv_cache(...)`; commits `ef5a226819` and `4938d44a3b` are
  literally "CPU_ATTN as Decoder and Flash_ATTN as prefiller".

**But it is hardcoded to the dense `CPU_ATTN` backend** — it never fires for `CPU_MLA`,
`AMX_MLA`, or `DeepseekV4CPUSparseBackend`. And the MLA case is explicitly stubbed out:
`vllm/platforms/cpu.py:644-653` returns early with *"MLA uses a single latent cache … there is
nothing to pack here"* for 3-D caches. FP8 KV is also refused on CPU with KV transfer:
`cpu.py:673-675` `raise NotImplementedError("FP8 KV cache is not yet supported with KV transfer
on CPU")` — relevant because DeepSeek uses `fp8_ds_mla`. The `attn_backend_name` is part of the
NIXL compatibility hash (`nixl/metadata.py:188`) and `enforce_handshake_compat` defaults True
(`base_worker.py:779-780`), so the mixed pair is rejected before the heterogeneous branch unless
you opt out.

So: **for a sparse-MLA GPU-prefill / CPU-decode split there is still no supported option**, but
the transport and a CPU KV post-process already exist and are the right scaffolding. Extending
them means (a) teaching the heterogeneous branch about an MLA/sparse-MLA CPU backend, (b)
implementing `pack_kv_cache` for the 3-D MLA latent, (c) an fp8-capable CPU cache path. The
other blockers stand:
- **A CPU sparse-MLA backend for GLM / V4.1 does not exist.** `vllm/platforms/cpu.py:144-145`
  raises `NotImplementedError("Sparse Attention is not supported on CPU.")` whenever
  `use_sparse` is set. The generic CPU MLA backend is dense/reference-quality, head_dim-576 and
  block_size-16 only (`cpu_mla.py:1-30`). The only CPU sparse implementation is base
  DeepSeek-V4's, and its kernels need **AVX512-BF16 + AMX** (`csrc/cpu/sgl-kernels/*`, loaded
  only when `_is_avx512_bf16_supported()`, `cpu.py:603-611`) — a Xeon Gold 6254 has AVX-512 but
  no AVX512-BF16, so they would not even load.
- **Cross-machine expert distribution is a draft.**
  "初步设计(v0.1 草案)"; the functions to change are `run_moe_and_ep` +
  `EpShmState` (`xiaotu_moe/csrc/python_binding/binding.cpp:395-477,184-200`) and
  `_ep_shm_attach` / `ep_shm_enabled` (`vllm_xiaotu_moe/hybrid_model.py:274-295,247-253`).
  There is no NIXL/RDMA code in the plugin today; the reduction is single-host `/dev/shm`.

### 7.3 The CPU-bandwidth reality check

The plugin's published CPU numbers come from **2× EPYC 9654, 24-channel DDR5-4800
(692–768 GB/s marginal)**. A **Xeon Gold 6254 is 6-channel DDR4-2666 ≈ 128 GB/s per socket**
(general spec knowledge; verify your actual machine), roughly **5–6× less**. Per-token expert
bytes:
- DeepSeek-V4.1: **3.246 GB/token** (measured, `CLUSTER_SCALE_DESIGN.md` R8) → ≥25 ms/token at
  perfect bandwidth utilization, realistically 35–50 ms.
- GLM-5.3: derived from the FP8 expert size — 283.5 GiB over 42 layers × 288 experts ≈ 25.2 MB
  per expert; 8 active experts × 42 layers ≈ **8.5 GB/token** (2.6× V4.1). On one Xeon 6254
  that is **≥66 ms/token**, i.e. ~10–15 tok/s single-stream *if* the engine saturates DRAM.

So "many free CPU servers" buys **aggregate throughput** (MoE weights are reused across a
batch, so throughput scales with concurrency and with machines), **not single-stream latency**.
Also check RAM: V4.1 needs ~630 GiB peak host RSS with Engram resident
(`MEM_FOOTPRINT_V41.md`), GLM-5.3 experts ~285 GiB + non-expert; a decode server must fit the
shard it is responsible for.

---

## 8. Model-by-model

### GLM-5.3-Flash (321B/18B)
- 45 layers: 3 dense + 42 sparse MoE; 288 experts top-8 + 1 shared; hidden 4096/inter 2048.
- Attention: **34 KDA linear-attention layers + 11 NoPE sparse-MLA**; `index_topk=2048`;
  KDA state is constant-size (context does not grow that half).
- FP8 e4m3 block-128 experts ≈ 283.5 GiB; non-expert ≈ 22 GiB.
- On SM70: hardest model because of the **KDA tl.dot kernels** (need Volta validation) and the
  NoPE sparse MLA (the SM8x backend exists but is bf16-gated). No CPU branch.

### DeepSeek-V4.1-Flash (552B + 196B Engram)
- 40 layers (20 causal-encoder + 20 decoder), 384 experts top-6, hidden 5120/inter 2304,
  FP4 experts (269 GiB) + FP8 Engram (183 GiB) — Engram is a lookup table and is already
  designed to live in host RAM with RDMA prefetch.
- KV ≈ 40 KB/token → the 16GB card is the binding constraint on context.
- On SM70: attention port is closer to the existing SM80 PR (sparse MLA + indexer + MQA
  fallback), but the SM80 fallback uses **tf32 tl.dot** (SM80-only), and there is no CPU branch.

---

## 9. Recommended phased plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **P0** | Toolchain: torch with sm_70 + CUDA 12.6/12.7, CMake `7.0` re-added, vLLM fork builds and imports on one V100 | `vllm serve` a small fp16 dense model on V100 |
| **P1** | FP16 + FP8-offline-dequant stack: load GLM-5.3/DSv4.1 as fp16, all non-expert layers fp16, no quant method | Model constructs on V100; weight load succeeds |
| **P2** | Attention port for **one** model (start with DeepSeek-V4.1, closest to the SM80 PR): extend gates, fix tf32 dots, torch fallbacks for the SM80-stubbed cuda helpers, fp16 KV | One prompt produces coherent output on 1 V100 (CPU experts) |
| **P3** | GPU prefill MoE on Volta: either (a) supply an fp16 HMMA grouped GEMM (CUTLASS sm70 / cuBLAS batched), or (b) accept CPU prefill. Casting to fp16 alone is not enough (Triton emits FMA on sm70) | Long-prompt TTFT beats the pure-CPU baseline |
| **P4** | GLM-5.3 attention port (KDA + NoPE sparse MLA), 16GB KV budget, multi-GPU TP/EP over PCIe | Coherent output at 128–256K context |
| **P5** | PD: GPU prefill + GPU decode instances (NIXL over IB), CPU MoE on both. Extend the existing heterogeneous path if a CPU-decode MLA backend is wanted | End-to-end disaggregated serving |
| **P6 (optional, very large)** | Pure-CPU decode: promote NIXL's `CPU_ATTN`-only heterogeneous path to an MLA/sparse-MLA CPU backend, implement `pack_kv_cache` for 3-D MLA + an fp8 CPU cache, port V4 CPU path to V4.1, write GLM CPU attention (KDA + sparse MLA + indexer) | CPU-only decode instance |

P2/P3 can be run in parallel with P0/P1 prep. Do **not** start P6 before P5 works. P3's choice
matters: without an HMMA kernel, "GPU prefill" on a V100 is largely a bandwidth/staging play,
and may not beat the CPU engine by much.

---

## 10. Verified vs inferred

**Verified from source (this workspace):**
- All capability gates, dtype gates, `get_min_capability` enforcement, backend tables and line
  numbers cited above.
- The plugin's GPU prefill kernels hardcode `tl.bfloat16`; the FP8 decode is uint8 arithmetic
  and `pr2` is arch-neutral.
- **Triton 3.7.1 emits `mma.sync` at cap80 but emits only `fma.rn.f32` at cap70/cap75** — I
  reproduced this by compiling minimal fp16/bf16 `tl.dot` kernels for cap 70/75/80 and
  inspecting the generated PTX/TTGIR (`mma_layout`/`ptx_mma`/`ptx_fma`). `num_stages`/pipelining
  is skipped for cap<80.
- The six plugin MoE kernels compile to valid sm_70 cubins (no inline PTX/cp.async/fp8 type).
- DeepSeek-V4 has a CPU implementation and platform branch; V4.1 and GLM-5.3 do not; CPU sparse
  attention is hard-rejected at `vllm/platforms/cpu.py:144-145`; the V4 CPU sparse kernels need
  AVX512-BF16/AMX.
- NIXL supports a CPU device and there is a heterogeneous CUDA-prefill→CPU-decode path, but it
  is hardcoded to dense `CPU_ATTN`, `pack_kv_cache` no-ops for MLA, and FP8 KV is refused on CPU
  with KV transfer.
- KV sizes per token and the plugin's per-model memory measurements from the project docs.
- `sm12x_mqa.py` uses `input_precision="tf32"`; KDA uses `DOT_PRECISION="ieee"`;
  `has_cutedsl()` is `not family(80)` and therefore True on SM70.

**Inferred / external (not proven here):**
- PyTorch sm_70 removal versions and CUDA 13 Volta removal (web sources cited).
- Xeon Gold 6254 memory bandwidth (~128 GB/s/socket) and absence of AVX512-BF16 — general spec
  knowledge; the actual "free CPU servers" specs are unknown and matter more than this one.
- GLM-5.3 per-token expert bytes (~8.5 GB/token) — derived arithmetic from the documented
  expert size, not a measurement.
- Feasibility/performance of a CUTLASS fp16 HMMA MoE on V100, and the effort estimates.
- No build or run on Volta hardware was performed; runtime failure modes are derived from source
  and from offline compilation, not from execution.

**Biggest unknowns to retire first:** (1) does a torch with sm_70 exist that vLLM 2.x can
actually run on; (2) how much of the end-to-end time is in `tl.dot` Triton kernels (indexer,
KDA, MoE) versus the dot-free sparse-MLA sink kernel — this decides whether a custom HMMA GEMM
is mandatory; (3) what the free CPU servers actually are (cores, channels, RAM).
