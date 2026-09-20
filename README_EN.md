# vllm-xtu-moe

> **XTU = X Transformers Unity** (pronounced *"xiao tu"*, Chinese for "little rabbit") —
> a **high-performance MoE inference acceleration layer**.
>
> It is a **vLLM plugin for very large MoE models**. The core idea is to split the work:
>
> * the **GPU** handles the **non-expert** parts — attention, KV cache, embeddings;
> * the **CPU** holds the **expert weights** and does the **expert compute**;
> * for **long prompts**, part of the expert compute is **streamed onto the GPU**.
>
> In other words: even when the **expert weights do not fit in VRAM**, a large MoE model
> still runs — while **changing mainline vLLM as little as possible**.
>
> **Focus**
>
> * very large MoE models such as **DeepSeek / GLM / MiMo**
> * **CPU + GPU hybrid inference** · **low VRAM footprint**
> * **high-performance long-context inference**
> * **mainline-vLLM compatible, no fork required**
> * aimed at **large-model deployment and engineering optimisation**

[中文](README.md) · English (default)

> **📌 Current release: v0.2.3** (2026-09-20) — **upstream tracking + MTP for GLM and MiMo**:
> the patch stack is rebased onto upstream `133b71e0b` (11 patches / 40 files);
> **GLM-5.3-Flash MTP is wired up and ON by default** (`SPEC_K=1`; accept 1.46,
> decode 21.9 → 22.6 tok/s, at the cost of a 27% smaller KV pool);
> **MiMo-V2.5 (310B/15B) now runs end-to-end on a single A100-40GB**, where
> **MTP k=1 measures +10% decode**. GLM production `GPU_UTIL` drops to **0.82**.
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
| **GLM-5.3-Flash** | 321B / 18B active, 288 experts / top-8 | FP8 block-128 | ✅ **end to end (TP=2, A100/SM80)**, delivered as **256K context × 2 concurrent sequences** (KV pool 988,081 tokens), needs `--kv-cache-dtype bfloat16`; C=1 decode ~21.5 tok/s; 4K-prompt prefill **140 → 180 tok/s (FP8 GPU prefill, 1.29×)** |
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

| Concurrency | prompt / output | prefill (tok/s) | decode (tok/s) | completed |
|---|---|---|---|---|
| C=1 | 256 / 128 | **127** | **21.6** | 8/8 |
| C=2 | 256 / 128 | 78 | 12.8 | 8/8 |
| **C=1** | **4096 / 64** | **180** | **21.3** | 2/2 |

> **Metric definitions (uniform across this README)**: `prefill (tok/s) = prompt_tokens / TTFT`
> and `decode (tok/s) = 1000 / TPOT`; both are converted from the TTFT/TPOT of the same
> `vllm bench serve` run. **`output tok/s` (which mixes in TTFT), TPOT and ITL are no longer
> reported.** For C=2 the two rates are per-stream, not aggregate throughput.

* **Long-prompt prefill**: 4096-in goes from **140 to 180 tok/s (1.29×)** — the FP8 GPU prefill
  streams each layer's 3.62 GB/rank of expert weights onto the
  GPU and **overlaps** that transfer with attention;
* **Decode is unchanged**: C=1 **21.3–21.6 tok/s**, as in v0.2.1 (the GPU path only affects
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

> **Engine-level microbenchmarks** (`xiaotu_moe` vs `lk_moe`, **milliseconds per layer** across
> BS shapes) are a **development metric** and are not published in the README — see the internal
> dev doc `dev-docs/TUNING_LK_MOE_VS_XIAOTU.md`. The README keeps only **service rates**
> (prefill / decode below).

### DeepSeek-V4.1-Flash, service level (same-parameter A/B on one host, TP=2, official `vllm bench serve`)

Both arms run in the **same conda env**; the only variable is the CPU MoE engine. Prompts are
byte-identical and both arms use 60 threads. **Only two rates are reported, with no crosstalk**:
`prefill (tok/s)` (before the first token) and `decode (tok/s)` (after it). Ratios are
**ours / `lk_moe`**, so **> 1.0 means we are faster**.

| prompt / output | `lk_moe` prefill (tok/s) | ours prefill (tok/s) | ratio | `lk_moe` decode (tok/s) | ours decode (tok/s) | ratio |
|---|---|---|---|---|---|---|
| 256 / 32 | 118.9 | 107.1 | **0.900×** | 44.7 | 31.1 | **0.695×** |
| 256 / 1024 | 118.6 | 105.1 | **0.886×** | 44.9 | 33.5 | **0.746×** |
| 8192 / 32 | 127.5 | 115.0 | **0.902×** | 45.4 | 34.3 | **0.756×** |
| 8192 / 1024 | 127.6 | 120.9 | **0.948×** | 44.8 | 34.7 | **0.775×** |

⇒ At the service level we are **still slower**: prefill by **5-11%** and **decode by 29-44%**
(decode is 0.70-0.78× of `lk_moe`), but the gap narrowed by **+8% to +15%** versus the previous
release.

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
| **v0.2.3** | **Upstream tracking + GLM/MiMo MTP** — patch stack rebased onto upstream `133b71e0b` (11 patches / 40 files); **GLM-5.3-Flash MTP wired up, ON by default** (`SPEC_K=1`, decode 21.9 → 22.6 tok/s at the cost of a 27% smaller KV pool); **MiMo-V2.5 (310B/15B) runs end-to-end on one A100-40GB** with MTP k=1 at +10% decode; GLM memory contract re-calibrated (`GPU_UTIL` 0.85 → **0.82**) |
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
