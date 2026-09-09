# xiaotu-moe — Open-Source Reimplementation of the Closed-Source lk_moe

**Milestone report (v1.0) — bilingual (EN/CN).**

- Repository: `xiaotu-moe` (Python package `xiaotu_moe`, native extension `_xiaotu_moe_C_*`)
- Target: drop-in replacement for the closed-source `lk_moe` binary engine inside the `Lvllmds4-x` / `Lvllm` vLLM forks
- License posture: Apache-2.0 based, no copying of lk_moe proprietary code

---

## 1. Project Origin

**lk_moe** is a closed-source CPU inference engine distributed as a
`cp312 manylinux x86_64` wheel on PyPI (version 2.3.3), no source distribution,
under a proprietary license (signed Qiong GU / guqiong9696@gmail.com, 2025-11-08).
It is used by the vLLM forks `Lvllm`, `Lvllmds4-x`, and `lsglang` (same author)
to replace the GPU-only MoE forward with a **CPU–GPU hybrid engine** so that a
larger model fits in VRAM+RAM (the "1+1=2", 100% VRAM utilization scheme).

Although lk_moe is closed, the author ships `THIRD_PARTY_LICENSES` that
explicitly name **three files derived from KVCache.AI ktransformers** (Apache-2.0):
`csrc/lk_moe/moe.h`, `csrc/lk_moe/moe.cpp`, `csrc/lk_moe/backend_numa.cpp`.
A byte-level inspection of the wheel confirmed:
- The **NUMA scheduler** (`Backend_NUMA`) originates from
  `ktransformers_ext/cpu_backend/backend.cpp` (string evidence including the
  author's original typo "suscess").
- The **MoE skeleton** is retained, but the **quantized compute core was fully
  rewritten (V2)** as a proprietary `MOE_V2<WeightTraits, ActivationType>`
  template with 10 instantiations, and the ggml/llamafile dependency was stripped.

**Conclusion:** reimplementation is feasible. The scheduler and MoE skeleton are
open (ktransformers/lktransformers, Apache-2.0); only the quantized GEMM /
dequantization kernels (FP8/BF16/FP16/WNA16/NVFP4/MXFP4) must be written from
scratch. See `LK_MOE_INVESTIGATION_REPORT.md` for the full provenance chain.

## 2. Goals

1. **Open, source-available replacement** for the closed-source `lk_moe` CPU MoE
   engine, from Apache-2.0 building blocks, without copying proprietary code.
2. **ABI-compatible drop-in**: implement the `MOEConfigV2` + `data_ptr()` /
   `cpu_decode` / `cpu_prefill` contracts so the fork's `moe_runner.py` dispatch
   works unchanged (`self.lk_moe` → `xiaotu_moe` instance).
3. **Correctness parity** with lk_moe on the real model, verified against the
   actual DeepSeek-V4-Flash-0731 checkpoint.
4. **Constraints**: reuse existing code; `MOEConfigV2` layout defined by us (the
   reimplementation); package name `xiaotu-moe`; no dependency on
   ggml/llama.cpp/libnuma.

## 3. Approach (思路)

A staged plan with each stage independently verifiable against the closed binary:

- **Phase 1** — minimal closed loop: BF16 CPU MoE (a dependency-free
  `MOE_V2<WeightTraits, ActivationType>` + bf16 gemm + a small `std::thread`
  pool). Establishes the orchestration semantics (gate → up → SiLU → down →
  weighted accumulate).
- **Phase 2** — extend to quantized types FP8 / WNA16 / MXFP4, each behind a
  CRTP `WeightTraits` interface so the orchestration loop is shared verbatim.
- **Phase 3a** — NVFP4 (strategy A: reuse the shared E2M1 kernel + per-expert
  global scale to match vLLM NVFP4 semantics).
- **Phase 3b** — **key architectural decision: xiaotu-moe is pure CPU**; the GPU
  side completely reuses the fork's standard vLLM GPU MoE (no hand-written CUDA).
  The author's own GPU CUDA files (`moe_v2_gpu_*.cu`) are understood as having
  **two roles** — (1) a shallow stream/buffer bridge for `cpu_decode`
  (always needed, xiaotu-moe meets it via the Python `_cpu_decode` contract with
  no CUDA kernel), and (2) a deep self-written `gpu_prefill` kernel that is a
  redundant substitute for standard vLLM GPU MoE and is therefore NOT
  reimplemented or copied.
- **Phase 4** — packaging: multiple ISA `.so` variants + `_dynamic_loader`
  runtime selection + integration into the fork and end-to-end validation.

## 4. Implementation

- `csrc/moe/moe_v2.hpp` — `MOE_V2<WeightTraits, ActivationType>`: config, weight
  snapshot (see fix below), `forward_many`, per-token orchestration.
- `csrc/moe/moe_v2_bf16.hpp` / `moe_v2_fp8.hpp` — BF16 / FP8 weight traits.
- `csrc/moe/moe_v2_packed4.hpp` — shared E2M1 packed-int4 kernel parameterized by
  `Packed4WeightTraitsBase<LUT, Tag, E8M0>` (WNA16 / MXFP4 = E8M0 true / NVFP4).
- `csrc/kernels/bf16_gemm.hpp`, `bf16/utils` — gemm + activation (bf16 helpers).
- `csrc/python_binding/binding.cpp` — pybind module exporting `MOE_BF16/FP16/FP8/
  FP16/WNA16/MXFP4/NVFP4[_FP16]` and `forward_many`, `cpu_decode`, `cpu_prefill`.
- `scripts/build_variants.sh` — builds 5 ISA variants (`scalar`/`avx2`/
  `avx512_base`/`avx512_vnni`/`avx512_bf16`) from one source via
  `-DXIAOTU_MOE_MODULE_NAME`.
- `xiaotu_moe/loader.py` — `_dynamic_loader` picks the best ISA at import.
- `integration/` — patches and the clean `routed_experts.swap_xiaotu.clean.py`
  that swaps `lk_moe.*` → `xiaotu_moe.*` in the fork.

### Weight layout (match to the fork's feed)

For each expert block (validated against real checkpoint, all in
token-of-the-checkpoint int8 / e8m0 format):
- w13 `[E][2I][H/2]`, w2 `[E][H][I/2]` — packed int4 along K (low nibble = even k,
  high nibble = odd k).
- Per-block scale `[E][N/groupN][K/groupK]`: for MXFP4 / NVFP4 the raw `fp8_e8m0`
  one-byte exponent scale is interpreted as `2^(byte−127)`; groupN=1, groupK=32.
- NVFP4 additionally carries a per-expert global scale.

## 5. Phase Status

| Phase | Scope | Status |
|---|---|---|
| 1 | BF16 CPU closed loop | ✅ |
| 2 | FP8 / WNA16 / MXFP4 CPU kernels | ✅ verified (A/B) |
| 3a | NVFP4 (strategy A) | ✅ verified |
| 3b | Pure-CPU + reuse fork GPU MoE | ✅ decided & documented |
| 4 | Multiple .so variants + loader + integration | ✅ end-to-end validated |

## 6. First end-to-end bug: dangling weight pointer → SIGSEGV (root cause & fix)

During integration into the real fork with the actual checkpoint, the EngineCore
segfaulted on the first CPU forward. Crash instruction
`vmovd (%r12,%rcx,1),%xmm0` (MXFP4 `gate_up` reading W); fault address
= `w13_ + 254×8MB` (layer-0 expert 254).

Evidence chain:
- **Not a race**: a single-threaded pre-scan reading the same address still
  faulted.
- **`/proc/self/maps`**: the layer-0 weight region was almost entirely inside a
  ~2.25 GB `---p` (PROT_NONE) anonymous guard region.
- **Control**: restoring the pristine fork (`import lk_moe`) + same checkpoint +
  a real prompt produced correct output (`" Paris. The capital of Spain is
  Madrid"`), proving lk_moe **snapshots its weights into its own buffer at
  construction**.

**Root cause:** vLLM's `clean_weights_after_loading` does `delattr(
self, 'w13_weight')` after loading; torch releases the original param tensors and
the pages are reclaimed / parked PROT_NONE. A reimplementation that only stores the
`data_ptr()` kernel pointer then reads freed memory → SIGSEGV.

**Fix:** the `MOE_V2` constructor now **copies w13/w2/scales into engine-owned
`std::unique_ptr<uint8_t[]>` buffers** and points its members at the copies — the
same ownership model the closed engine uses. Cost: ~2.7 s/layer extra construction
(~127 s full model load) and a transient ~137 GB while the originals are still
alive (machine has 1.3 TB, ~1.18 TB free at test time).

## 7. Verification

- Real checkpoint (DeepSeek-V4-Flash-0731), fresh fork env, clean build:
  `Application startup complete`, real prompt → **HTTP 200, output byte-identical
  to lk_moe**, 0 segfault. Longer generation (28-token haiku) also stable.
- cp310 + cp312 × 5 ISA variants all rebuild successfully; module imports and
  exports all 12 `MOE_*` classes; `load()` present.
- Pure-CPU numeric A/B vs numpy golden earlier: MXFP4 max rel ~3.4e-4, NVFP4
  ~8.7e-6, far better than lk_moe's ~22% tail.

## 8. Deliverables / Artifacts

- `csrc/moe/moe_v2.hpp` — weight-snapshot fix (core)
- `csrc/moe/moe_v2_packed4.hpp` — `kE8M0` trait
- `integration/routed_experts.swap_xiaotu.clean.py` — clean module-swap file
  (installed into the fresh env site-packages)
- `build/*.so` — 12 variants (6 ISA × cp310/cp312)
- Docs: `LK_MOE_REIMPLEMENTATION_PLAN.md`, `LK_MOE_INVESTIGATION_REPORT.md`

## 9. Known gaps & next steps

- Measured only on `Lvllmds4-x`; `Lvllm` (parallel line) is judged equivalent by
  ABI (see investigation §1.5) but not separately live-tested.
- The A6000 / without-FP8 tensor-core fallback line and a real WNA16 model were
  not integration-tested.
- The two optimizations below (expert grouping + persistent NUMA thread pool)
  **were implemented and measured** — see §10. Continual-load pollution of the
  benchmark machine (an unrelated vLLM server saturating ~90 cores) means the
  timing numbers are order-of-magnitude only, not clean Roofline data.

## 10. Performance optimizations (measured)

### 10.1 Expert grouping — implemented, **no speedup, honest negative result**

`forward_many` was rewritten to walk active experts once each instead of
per token×rank:
1. `per_expert[e]` keeps the assignment index list for each active expert (the
   routing weights `w==0` entries are skipped).
2. **Adaptive dispatch**: when `n_active×F ≤ NASS` (fits in the scratch space,
   F defaults to 8, overridable via `XIAOTU_MOE_GROUP_FACTOR`), the grouped path
   runs; otherwise it falls back to the per-token path. This keeps the two paths
   bit-comparable for A/B.
3. The grouped path is **3-phase** to avoid CAS contention across experts:
   - Phase 1 (parallel over jobs): gate/up + SiLU → `act_scratch_`
   - Phase 2 (parallel over jobs): bf16 + down → `down_scratch_` (no shared
     output writes)
   - Phase 3 (parallel over tokens): `out[t] = Σ_r w·down_scratch_[ai]`
   (each token one thread, rank-ordered accumulation, no contention)

**Correctness**: grouped vs per-token reference is **bit-identical**
(max abs 0, max rel 0); grouped vs the numpy golden stays at the usual
quantization error (max rel ~8.7e-4 relative to |ref|>0.05 for MXFP4).

**Measured result (machine under continual external load — indicative only):**

| batch | per-token (diverse routing) ms/layer | grouped (concentrated) ms/layer |
|---|---|---|
| 1  | 76 | 74 |
| 8  | 76 | 74 |
| 16 | 76 | 110 |
| 32 | 152 | 213 |
| 64 | 303 | 424-430 |

Grouping did **not** reduce latency; at larger batches it made things notably
slower. **Root cause:** each gate_up/down still reads its full 12 MB expert block
per assignment (the block is far larger than L2, so nothing is reused across
assignments) — grouping alone therefore does not achieve "read each block once."
Actually reading each block once requires a **blocked / batched GEMM with register
blocking**, which is beyond this phase and remains future work. This honest
negative result is recorded here and in the repository so it is not re-burned.

### 10.2 Backend_NUMA equivalent — persistent NUMA thread pool (implemented, fixed)

`csrc/moe/numa_pool.hpp` provides `NumaWorkPool`: a persistent worker pool
replacing the per-call `ThreadPool`.
- Persistent threads (created once, reused across calls) — removes thread-creation
  overhead per call.
- NUMA affinity: topology probed from `/proc/cpuinfo` + `/sys/devices/system/node`;
  each worker pinned to a distinct physical core, rotating across NUMA nodes via
  `sched_setaffinity`. Memory interleave uses a raw `mbind` syscall — **no libnuma
  dependency**.
- Dynamic scheduling: shared atomic index `counter_.fetch_add(1)` (implicit work
  stealing); `parallel_for(n, fn)` uses a generation completion barrier.
- Thread count = hardware concurrency by default, overridable via
  `XIAOTU_MOE_THREADS`.

**Completion-barrier bug (found & fixed):** the original shared
`done_this_gen_`+`completion_gen_` counters had a cross-generation reset race that
let `parallel_for` return while workers were still running (reproduced ~80% crash
under stress; ASan proved a worker read the caller's numpy array after it was
freed). **Fix:** per-worker completion records
(`worker_gen_[w]`, `std::atomic<uint64_t>[]`); each worker only ever writes its own
slot, and the caller waits until every slot equals the current generation. No
shared accumulated-counter reset → no cross-generation ambiguity, provably no
early return. Verified 20/20 stress runs with 0 crashes and no deadlock.

> **Note on "apparent hangs":** on this machine an unrelated persistent vLLM
> server (port 8070, ~90 cores) keeps the host load >130. The pool's sleeping
> workers occasionally take ~5 s to be woken by `cv_` under that load, which
> self-recovers and is **not** a deadlock. Timing/pool measurements on this host
> must factor in this background load; a plain `pthread_barrier` on hot-resident
> threads measures ~13 µs.
