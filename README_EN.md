# vllm-xtu-moe

> **XTU = X Transformers Unity** (pronounced *"xiao tu"*, Chinese for "little rabbit").
> A **vLLM plugin** that keeps **MoE expert weights in CPU memory** while
> **attention and KV cache stay on the GPU** — so large MoE models can be served on
> machines where the experts do not fit in VRAM. **No fork, no upstream patching.**

[中文](README.md) · English (default)

> **📌 Current release: v0.2.3** (2026-09-20) — **upstream tracking + MTP for GLM and MiMo**:
> the patch stack is rebased onto upstream `133b71e0b` (11 patches / 40 files);
> **GLM-5.3-Flash MTP is wired up and ON by default** (`SPEC_K=1`; accept 1.46,
> TPOT 45.7 → 44.2 ms, at the cost of a 27% smaller KV pool);
> **MiMo-V2.5 (310B/15B) now runs end-to-end on a single A100-40GB**, where
> **MTP k=1 measures −9.4% TPOT**. GLM production `GPU_UTIL` drops to **0.82**.
>
> Previous, **v0.2.2** (2026-09-19) — **GLM-5.3-Flash support**: FP8 GPU prefill wired up
> (4K-prompt TTFT **29.3 s → 22.8 s**), delivered as **256K context × 2 concurrent
> sequences**; also fixed an **e4m3 subnormal-decode defect** and added an all-codeword gate.
>
> **v0.2.1** (2026-09-18) — **major CPU-engine performance work**:
> the CPU MoE engine now **beats the reference `lk_moe` implementation on every real shape**;
> DeepSeek-V4-Flash benefits as well.
> Release notes: [`RELEASE_NOTES_v0.2.3.md`](RELEASE_NOTES_v0.2.3.md) · [`RELEASE_NOTES_v0.2.2.md`](RELEASE_NOTES_v0.2.2.md) · [`RELEASE_NOTES_v0.2.1.md`](RELEASE_NOTES_v0.2.1.md)

---

## What problem it solves

Large MoE models (DeepSeek-V4.1-Flash 748B, V4-Flash, GLM-5.3-Flash 321B, …) carry
150-300 GiB of expert weights — far beyond any single- or dual-GPU card. But the experts
are only a small part of the per-token compute; the rest is attention and shared layers,
which run fast on the GPU. So we split the model:

* **non-expert** weights (attention, KV cache, embeddings) → GPU, using vLLM's own kernels;
* **experts** → CPU memory, computed by the `xiaotu_moe` engine over NUMA shards;
* **long prefill** (token count ≥ threshold) → experts are streamed layer by layer onto the GPU.

## Goals

1. **Any MoE model.** Not tied to one architecture generation. DeepSeek-V4 / V4.1 work
   end to end; GLM-5.3-Flash runs end to end too, with an FP8 GPU prefill path and a
   delivered **256K × 2-concurrent** service config (see the support matrix).
2. **Any x86 ISA.** `scalar → AVX2 → AVX-512 (base/VNNI/BF16/VBMI)`, selected at import
   time from `/proc/cpuinfo`.
3. **A fixed VRAM priority order.** `1M context → GPU prefill → speculative decoding → residency`.
   No feature may push that order back.
4. **Exactly one copy of the weights in host memory** (not "source tensors + engine copy").
5. **Engine efficiency benchmarked against the best available.** `xiaotu_moe` is compared with
   `lk_moe` **on the same machine, same weights, same thread count**; the target is ≥90% of it.

---

## Support matrix

| Model | Size | Expert format | Status |
|---|---|---|---|
| **DeepSeek-V4.1-Flash** | 748B | MXFP4 (E8M0 block-32) | ✅ end to end (TP=2) |
| **DeepSeek-V4-Flash** (0731) | 256 experts / top-6 | MXFP4 | ✅ end to end |
| **GLM-5.3-Flash** | 321B / 18B active, 288 experts / top-8 | FP8 block-128 | ✅ **end to end (TP=2, A100/SM80)**, delivered as **256K context × 2 concurrent sequences** (KV pool 988,081 tokens), needs `--kv-cache-dtype bfloat16`; C=1 decode ~22 tok/s (TPOT 46 ms); 4K-prompt TTFT **29.3 s (CPU) → 22.8 s (FP8 GPU prefill, 1.29×)** |
| Other plug-and-play MoE | — | BF16 / FP8 | ✅ generic path, no per-model calibration |

**Measurement host (every number below was taken here)**

| Item | Configuration |
|---|---|
| CPU | 2× AMD EPYC 9654 (192 physical cores / 384 threads, 8 NUMA nodes) |
| RAM | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| OS | Ubuntu 22.04 · conda env `lvllm` (vLLM 2.5.0 base) |

---

## Performance (latest release **v0.2.3**)

> **How to read these tables.** Every number is **elapsed time in ms/layer — lower is better**,
> i.e. "the same work done faster". The baseline is `lk_moe` (Lvllm's CPU MoE engine), measured
> on the same machine, same real layer weights and same thread count, with the **`lk_moe`
> denominator re-measured in the same session** to avoid cross-session drift.
> **A ratio below 1.0 means we are faster.**

### GLM-5.3-Flash (2×A100-40GB, TP=2, SM80; delivered config: 256K × 2 concurrent)

Official `vllm bench serve`, random dataset + `--ignore-eos`, a distinct seed per cell.

| Concurrency | prompt / output | out tok/s (incl. TTFT) | mean TTFT | mean TPOT | completed |
|---|---|---|---|---|---|
| C=1 | 256 / 128 | **16.18** | 2022 ms | **46.37 ms** | 8/8 |
| C=2 | 256 / 128 | 19.36 | 3287 ms | 78.13 ms | 8/8 |
| **C=1** | **4096 / 64** | 2.49 | **22764 ms** | 46.95 ms | 2/2 |

* **Long-prompt prefill**: 4096-in TTFT drops from **29.3 s (v0.2.1, CPU prefill) to 22.8 s
  (1.29×)** — the FP8 GPU prefill streams each layer's 3.62 GB/rank of expert weights onto the
  GPU and **overlaps** that transfer with attention;
* **Decode is unchanged**: C=1 TPOT 46 ms (~22 tok/s), as in v0.2.1 (the GPU path only affects
  prefill);
* **Context**: 256K × 2 concurrent (KV pool 988,081 tokens) is the **delivered target for this
  hardware**; 512K/704K merely start (704K with only 1.06× concurrency, ceiling ≈733K) and are
  recorded as capability, not targets; **1M is out of scope** (needs an fp8 KV cache, decided
  against — see [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md));
* **Correctness**: engine determinism gate 11/11; layer gate rms_rel 4.4e-3; a 29,746-token
  needle retrieval is exact; three concurrent ~8K requests (two-way admission) all retrieve
  their own secret; 0 OOM.

> The FP8 CPU inner loop has a **default-off** acceleration switch, `XIAOTU_MOE_FP8_BF16_MMA=1`
> (AVX512-BF16 `vdpbf16ps`; 1.17-1.20× for M≥6, at the cost of rounding weights to bf16:
> rms_rel 3.6e-3 between the two paths).

### DeepSeek-V4.1-Flash (real routing shape `na≈226`, 60 threads, ms/layer)

| Shape | `xiaotu_moe` | `lk_moe` | Ratio |
|---|---|---|---|
| BS=227 (~6 rows/expert) | **28.56** | 30.24 | **0.944×** |
| BS=1893 | **213.36** | 217.34 | **0.982×** |
| BS=8192 | **877.55** | 933.94 | **0.940×** |
| decode BS=1 | **0.37** | 0.43 | **0.861×** |

### DeepSeek-V4-Flash (0731, same protocol)

| Shape | `xiaotu_moe` | `lk_moe` | Ratio |
|---|---|---|---|
| BS=227 | **20.89** | 22.18 | **0.942×** |
| BS=1893 | 162.05 | **155.86** | 1.040× |
| BS=8192 | **646.60** | 668.10 | **0.968×** |
| decode BS=1 | 0.29 | 0.29 | 1.000× |

### DeepSeek-V4.1-Flash, service level (same-parameter A/B on one host, TP=2, official `vllm bench serve`)

Both arms run in the **same conda env**; the only variable is the CPU MoE engine. Prompts are
byte-identical and both arms use 60 threads. **The three metrics are independent and are never
converted into one another** — `output tok/s` is the average decode rate for the whole run
(**TTFT included**), while TTFT and TPOT (**TTFT excluded**) are reported separately.

| prompt / output | `lk_moe` out tok/s | ours out tok/s | ratio | `lk_moe` TTFT | ours TTFT | ratio | `lk_moe` TPOT | ours TPOT | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 256 / 32 | 11.24 | 9.44 | **0.840×** | 2153 ms | 2391 ms | 1.111× | 22.37 ms | 32.19 ms | 1.439× |
| 256 / 1024 | 41.08 | 31.07 | **0.756×** | 2159 ms | 2436 ms | 1.128× | 22.26 ms | 29.83 ms | 1.340× |
| 8192 / 32 | 0.49 | 0.44 | **0.898×** | 64259 ms | 71242 ms | 1.109× | 22.03 ms | 29.13 ms | 1.322× |
| 8192 / 1024 | 11.76 | 10.53 | **0.895×** | 64208 ms | 67734 ms | 1.055× | 22.33 ms | 28.82 ms | 1.291× |

⇒ At the service level we are **still 10-24% slower** (TTFT 5-13% higher, pure-decode TPOT
29-44% higher), but the gap narrowed by **+8% to +15%** versus the previous release.

### GPU prefill (long prefill handed to the GPU)

Streaming expert compute layer by layer onto the GPU makes client-side TTFT **2.0-2.8× faster**
on DeepSeek-V4.1-Flash (worth it above a 4096-token threshold) and **1.29×** on GLM-5.3-Flash
(FP8, 4096-in: 29.3 → 22.8 s). Configuration and the VRAM
recipe are in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

**Full service-level comparison, methodology and reproduction commands:**
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) · [`RELEASE_NOTES_v0.2.3.md`](RELEASE_NOTES_v0.2.3.md) · [`RELEASE_NOTES_v0.2.2.md`](RELEASE_NOTES_v0.2.2.md) · [`RELEASE_NOTES_v0.2.1.md`](RELEASE_NOTES_v0.2.1.md).

---

## Quick start

```bash
# 1) upstream vLLM (this project is a plugin, no fork needed)
pip install vllm==2.5.0

# 2) this plugin (distribution name vllm-xtu-moe, NOT on PyPI) -- pick one
# (a) install the wheel attached to the GitHub Release (all 6 ISA variants included):
pip install ./vllm_xtu_moe-0.2.2-cp312-cp312-manylinux_2_34_x86_64.whl
# (b) or install from source (needs a local compiler):
# CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh && pip install -e .

# 3) serve a MoE model whose experts do not fit in VRAM
VLLM_EXPERTS_LOAD_DEVICE=cpu \
vllm serve <MODEL_DIR> --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 65536 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95
```

Per-model recipes, memory budgeting, self-checks and troubleshooting →
**[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** and **[`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md)**.

---

## Changelog (brief)

| Version | Theme |
|---|---|
| **v0.2.3** | **Upstream tracking + GLM/MiMo MTP** — patch stack rebased onto upstream `133b71e0b` (11 patches / 40 files); **GLM-5.3-Flash MTP wired up, ON by default** (`SPEC_K=1`, TPOT 45.7 → 44.2 ms at the cost of a 27% smaller KV pool); **MiMo-V2.5 (310B/15B) runs end-to-end on one A100-40GB** with MTP k=1 at −9.4% TPOT; GLM memory contract re-calibrated (`GPU_UTIL` 0.85 → **0.82**) |
| **v0.2.2** | **GLM-5.3-Flash support** — FP8 GPU prefill wired up (4K-prompt TTFT 29.3 → 22.8 s), delivered as 256K × 2 concurrent; fixes an e4m3 subnormal-decode defect and adds an all-codeword gate |
| **v0.2.1** | **Major CPU-engine performance work** — the CPU MoE engine now **beats `lk_moe` on every real shape**; DeepSeek-V4-Flash benefits too |
| v0.2 | DeepSeek-V4.1-Flash end-to-end (1M context + GPU prefill + speculative decoding) plus CPU-prefill path optimisation |
| v0.1.0 | First public release: hybrid mode (CPU experts + GPU rest), AVX2 / AVX-512 multi-ISA, DeepSeek-V4 family |

Details, performance comparisons and parameter changes:
[**v0.2.3**](RELEASE_NOTES_v0.2.3.md) · [**v0.2.2**](RELEASE_NOTES_v0.2.2.md) · [**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
(archived: [v0.2pre](RELEASE_NOTES_v0.2pre.md))

---

## Documentation

**For users — [`docs/`](docs/)**

| Document | Contents |
|---|---|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | **Runbook**: install, launch flags, VRAM budgeting, GPU-prefill recipe, JIT cache, release process |
| [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) | Per-model resource needs, launch commands and performance |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Measured data and how to reproduce it |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | Known limitations and unsupported combinations |
| [`docs/INSTALL_MAINLINE.md`](docs/INSTALL_MAINLINE.md) | Preparing an upstream vLLM environment |

**For developers** — architecture and upstream integration, the GPU-prefill implementation,
upstream drift, tuning logs and all internal reports are **internal development documents and are
not published with this repository** (they exist only in the local working copy).

---

## Acknowledgements

This project was inspired by **KTransformers** and **Lvllm**. In particular the compute engine
**`xiaotu-moe`** borrows heavily from ideas in **lk-moe**; **all `xiaotu-moe` code is independently written**.
Our sincere thanks.

* KTransformers — <https://github.com/kvcache-ai/ktransformers>
* Lvllm (and its CPU MoE engine **lk-moe**; the `lk_moe` 2.4.2 build used as our baseline comes from this repo) — <https://github.com/guqiong96/Lvllmds4-x>
* Thanks also to **vLLM** (<https://github.com/vllm-project/vllm>) for the plugin extension points
  that let this project integrate without a fork.

---

## License

Apache-2.0. Third-party components and notices: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`NOTICE`](NOTICE).
