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
   end to end; GLM-5.3-Flash runs end to end too with an FP8 GPU prefill path; MiMo-V2.5
   runs end to end on a single card.
2. **Any x86 ISA.** `scalar → AVX2 → AVX-512 (base/VNNI/BF16/VBMI)`, selected at import
   time from `/proc/cpuinfo`.
3. **A fixed VRAM priority order.** `1M context → GPU prefill → speculative decoding → residency`.
   No feature may push that order back.
4. **Exactly one copy of the weights in host memory** (not "source tensors + engine copy").
5. **Engine efficiency benchmarked against the best available.** `xiaotu_moe` is compared with
   `lk_moe` **on the same machine, same weights, same thread count**; the target is ≥90% of it.

---

## Measurement host

Every performance number below was taken on this machine:

| Item | Configuration |
|---|---|
| CPU | 2× AMD EPYC 9654 (192 physical cores / 384 threads, 8 NUMA nodes) |
| RAM | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| OS | Ubuntu 22.04 · conda env `lvllm` |

---

## Performance (latest release **v0.2.3**)

> **Metric definitions (uniform across this README)**: `prefill (tok/s) = prompt_tokens / TTFT`
> (the prefill rate before the first token) and `decode (tok/s) = 1000 / TPOT` (the decode rate
> after it, **prefill excluded**); both are converted from the TTFT/TPOT of the same
> `vllm bench serve` run. **`output tok/s` (which mixes in TTFT), TPOT and ITL are no longer
> reported.** ⚠️ **A short prompt depresses prefill (tok/s)** because of fixed overhead, so read
> that column on long prompts (see the notes below). C=1 and C=2 rates are **per-stream**, not
> aggregate throughput.

| Model (hardware) | Spec | prompt | conc. | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>2×A100-40GB · TP=2 | off | 256 | 1 | 110 | 21.9 |
| | off | 4096 | 1 | 143 | 21.2 |
| | **on k=1** (default) | 256 | 1 | 83 | **22.6** |
| | on k=1 | 4096 | 1 | 138 | 16.7 |
| | on k=4 | 256 | 1 | 111 | 12.4 |
| **MiMo-V2.5**<br>1×A100-40GB · TP=1 | off | 256 | 1 | 59 | 15.6 |
| | **on k=1** (recommended) | 256 | 1 | 53 | **17.2** |
| | on k=1 | 1024 | 1 | 81 | 17.5 |
| | on k=1 | 2048 | 1 | 85 | 17.1 |
| | on k=1 | 4096 | 1 | 86 | 17.0 |
| | on k=1 **+ GPU streaming prefill** | 4096 | 1 | **194** | 13.8 |
| **DeepSeek-V4.1-Flash**<br>2×A100-40GB · TP=2 | off | 32 | 1 / 2 | 68 / 36 | 27.6 / 22.2 |
| | off | 256 | 1 / 2 | 129 / 93 | 22.9 / 17.2 |
| | off | 1024 | 1 / 2 | 102 / 68 | 15.5 / 8.7 |
| | off | 4096 | 1 / 2 | 319 / 212 | 15.1 / 7.8 |
| | off | 16384 | 1 / 2 | **474** / 272 | 18.4 / 7.6 |
| | off | 32768 | 1 / 2 | 462 / 286 | 17.1 / 3.6 |

**Per-row configuration and sample size**

| Model | Serving config | Sample / notes |
|---|---|---|
| GLM-5.3-Flash | `GPU_UTIL=0.82` (mandatory since 0.2.3), `SPEC_K` controls speculation, GPU prefill ON (threshold 1500) | decode is **N=8×2** repeated; prefill comes from the same config at N=1–2. KV pool: spec off **915,487** / on **666,366** (−27%); delivered as **256K × 2 concurrent** |
| MiMo-V2.5 | single card TP=1, Hybrid SWA-128 + DiffKV (`TRITON_ATTN_DIFFKV`), experts on CPU | decode **N=8×2**; prefill N=2. **MTP k=3 is unusable** (accept 1.016 ⇒ 2.4× slower); greedy output is **byte-identical** to spec-off; load takes ~25–30 min |
| DeepSeek-V4.1-Flash | TP=2 · MBT=8192 · GPU prefill ON · **spec off** · prefix caching off (to measure real prefill) · random out=128 | full 6 lengths × 4 concurrencies in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md); aggregate ceiling ≈420–455 tok/s; TTFT is linear in length and strongly concurrency-dependent |

**How to read prefill** — all three models get **faster prefill as the prompt grows**, because part
of the cost is a **fixed per-chunk cost** (streaming expert weights to the GPU) that a bigger chunk
amortises:

* GLM 110 → 143 (short → 4K); MiMo 59 → **194** (with GPU streaming prefill on);
  DeepSeek-V4.1 68 → **474** (32 → 16K, 6×);
* conversely DeepSeek-V4.1 on a **32-token prompt** gets only 68 tok/s — all fixed overhead;
* **decode is the opposite**: it falls as concurrency rises (clearly at C=2), because prefill and
  decode contend for the same CPU expert compute.

**Other points**

* **GLM delivery = 256K context × 2 concurrent**; since 0.2.3 `GPU_UTIL` **must be `0.82`**
  (the upstream KV sizing changed, and 0.85 OOMs at two-way concurrency): KV pool 915,487
  (spec off) / 666,366 (spec on), i.e. **MTP's real cost is KV −27%**
  (256K concurrency 3.49× → 2.54×) plus **pulsing inter-token spacing** (1–2 tokens per step,
  which hurts streaming). 1M is out of scope (needs an fp8 KV cache, decided against).
* **Correctness**: engine determinism gate 11/11; layer gate rms_rel 4.4e-3; a 29,746-token
  needle retrieval is exact; three concurrent ~8K requests (two-way admission) all retrieve their
  own secret; 0 OOM.
* **MiMo with GPU streaming prefill** needs a lower `GPU_UTIL` (measured 0.65) because staging is
  **12.75 GiB/rank** (larger than GLM's 7.59 — `E=256×I=2048` is wider); at 8K context 5.8×
  concurrency still fits.
* The FP8 CPU inner loop has a **default-off** switch, `XIAOTU_MOE_FP8_BF16_MMA=1`
  (AVX512-BF16 `vdpbf16ps`; 1.17–1.20× for M≥6, at the cost of rounding weights to bf16:
  rms_rel 3.6e-3 between the two paths).

### DeepSeek-V4.1-Flash, service level: same-parameter A/B against `lk_moe`

Both arms run in the **same conda env**; the only variable is the CPU MoE engine. Prompts are
byte-identical and both arms use 60 threads. Ratios are **ours / `lk_moe`**, so
**> 1.0 means we are faster**.

| prompt / output | `lk_moe` prefill (tok/s) | ours prefill (tok/s) | ratio | `lk_moe` decode (tok/s) | ours decode (tok/s) | ratio |
|---|---|---|---|---|---|---|
| 256 / 32 | 118.9 | 107.1 | **0.900×** | 44.7 | 31.1 | **0.695×** |
| 256 / 1024 | 118.6 | 105.1 | **0.886×** | 44.9 | 33.5 | **0.746×** |
| 8192 / 32 | 127.5 | 115.0 | **0.902×** | 45.4 | 34.3 | **0.756×** |
| 8192 / 1024 | 127.6 | 120.9 | **0.948×** | 44.8 | 34.7 | **0.775×** |

⇒ At the service level we are **still slower**: prefill by **5-11%** and **decode by 29-44%**
(decode is 0.70-0.78× of `lk_moe`), but the gap narrowed by **+8% to +15%** versus the previous
release. Engine-level microbenchmarks (ms/layer) are a development metric and are not published
here.

### GPU prefill (long prefill handed to the GPU)

Streaming expert compute layer by layer onto the GPU. Two expert formats are supported:
**MXFP4** (DeepSeek-V4.1-Flash et al.) and **FP8 e4m3 block-128** (GLM-5.3-Flash).

**⚠️ The payoff is decided entirely by the chunk size.** Every chunk must stream **one full copy
of all layers' expert weights** over H2D — independent of how many tokens the chunk holds — so the
**per-chunk cost is near fixed** and a bigger chunk amortises it better. That is why the win is
1.2× for one model and 4.4× for another: it depends on whether the model can use large chunks.

| GLM-5.3-Flash (same prompt, different chunk size) | pure CPU | GPU streaming | win |
|---|---|---|---|
| **2176** (GLM's **hard cap**, pinned by the KDA `block_size`) | 11.3 s | 9.7 s | 1.2× |
| 4096 (counterfactual: without the KDA constraint) | 21.3 s | 9.7 s | **2.2×** |
| 8192 (counterfactual) | 42.6 s | 9.7 s | **4.4×** |

⇒ the **GPU column is 9.7 s at every chunk size** — direct evidence of the fixed cost — while the
CPU column grows linearly. **GLM's 1.2× is not the approach failing; it is the 2176 cap.**

**DeepSeek-V4.1-Flash is not capped and can use large chunks, so it lands in a different league**
(same **13.8K** prompt):

| Version | TTFT | prefill |
|---|---|---|
| before the fix | 76.5 s | 181 tok/s |
| **after the fix** (ring-slot reuse + device-level gating + fast transpose) | **8.008 s** | **1725 tok/s** |

⇒ **9.6×**, and **6.7×** against CPU prefill (13.8K × 3.9 ms ≈ 54 s).

**Per-layer cost breakdown (GLM, measured)**: `assembly 207 ms` (weight H2D, 3.62 GB/rank) +
`GPU MoE 23 ms` — **assembly is 90%**, so the lever is **overlapping assembly with attention**
(the side stream, on by default), not "moving bytes faster". In isolation assembly measures
**134.9 ms = 26.86 GB/s**, this host's H2D ceiling (1-D and pitched 2-D are equally fast; pinned
memory, `numactl --interleave` and two-rank concurrency change nothing). The 207 ms in service is
contention with vLLM's own PCIe traffic (TP=2 all-reduce; there is no NVLink between GPU 0 and 1).

**GLM-specific caveat**: its chunk is pinned by `block_size=2176`, so the plugin's 4096 default
threshold **never fires** for GLM; `scripts/serve_glm53_mainline.sh` lowers it to
`GPU_PREFILL_MIN=1500` (measured break-even ~1300). On the MXFP4 side the default ≥4096 is
worth it.

Configuration and the VRAM
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
