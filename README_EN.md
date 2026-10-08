# vllm-xtu-moe

> ⚠️ **命名变更(2026-10-07,用户定案)**:本文件里出现的旧环境变量名**已修改为**下面的规范名 ——
> `XIAOTU_MOE_W4A8` **已修改为** `XIAOTU_MOE_INT8`(int8 激活路径**总开关**,默认由权重格式决定);
> `XIAOTU_MOE_INT8_ALIGN` **已修改为** `XIAOTU_MOE_INT8_ALIGN`(实现选择:ALIGN tile,默认档);
> `XIAOTU_MOE_INT8_VNNI` **已修改为** `XIAOTU_MOE_INT8_VNNI`(实现选择:旧 VNNI tile,默认 0);
> `XIAOTU_MOE_I8_MIN_M` **已修改为** `XIAOTU_MOE_INT8_VNNI_MIN_TOKENS`(⭐ M 阈值,**默认 160**,唯一设定处)。
> 旧名仍被识别:会**告警一次**并映射到新名(**仍按旧值生效**),绝不静默退回 fp32 ✓


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

> **📌 Current release: v0.2.6** (2026-10-06) — **bootstrapped development: the whole session ran on the engine it was building, and FP4 expert weights gained an INT8 activation compute path**:
> **MXFP4 expert weights now have an INT8 activation path** (the fp4 code set `{0, ±1…±12}` fits int8 exactly,
> so the **weight side is free**; the entire numerical cost comes from quantising the activations) —
> kernel level **24.9 -> 15.1-15.7 ms/layer (1.54-1.62x)**, `max_abs ≈ 3.163e-03`;
> end to end (qfn, MXFP4-FP8): GSM8K 200 questions **193/200 on both arms**, 194/200 per-question agreement
> (net change 0), 256K four-pin **4/4 with `finish_reason=stop` on both arms**, Vision 23 agrees 18/23;
> also fixed the **fallback-path regression** (was 16-22% slower -> **+1.49% instructions**),
> brought up **MTP on qfn** (**2.44x** decode) and set the qfn service default thread count to
> **120 (= 5 cores/CCD)**. The path is **off by default**; enable with `XIAOTU_MOE_INT8=1`.
> Release notes: [`RELEASE_NOTES_v0.2.6.md`](RELEASE_NOTES_v0.2.6.md)

---

> [!WARNING]
> ## ⚠️ Do NOT enable LMCache (known critical issue)
>
> **This project disables LMCache by default** (`LMCACHE=0`, and the `lmcache_server` is not started), because:
>
> * 🔴 **A long prompt can take the whole engine down**: one large request can trigger
>   `KV connector reported block-level load failures (invalid_block_ids)` -> `EngineDeadError`,
>   killing **every** session. Measured here: the *same* 780k-token request crashed the engine
>   **twice** with LMCache on, while with it off a 200k-token request returned 200 and GPU prefill
>   went from **0 to 686** `device=cuda` launches.
> * 🔴 **Strongly correlated with output degeneration** (repetition, intent drift, corrupted paths),
>   matching several unfixed upstream issues.
> * ✅ **Disabling it also helps**: ~**0.85 GiB** less GPU memory and ~**65 GiB** less host RSS,
>   and GPU prefill starts working again.
>
> **Full evidence, upstream issue list, and how to disable / roll back:
> [`docs/KNOWN_ISSUES_LMCACHE.md`](docs/KNOWN_ISSUES_LMCACHE.md).**
> **Do not enable it in production until upstream fixes it** — any LMCache reference below is historical.

## Project highlights

1. Any MoE model. Not tied to one architecture generation. DeepSeek-V4 / V4.1 work
   end to end; GLM-5.3-Flash runs end to end too with an FP8 GPU prefill path; MiMo-V2.5
   runs end to end on a single card.
   **MiMo-V2.6-Flash-RL also runs on A100/SM80**: the skeleton, a 1M context and real-weight
   long-context generation have all been measured (multimodal input and throughput pending).
   It takes the same **MXFP4** engine path as V4.1 and is 161 GiB -- see `docs/MODEL_GUIDES.md`
   section 3b and `docs/EXPERIMENTS.md` B92-B100.
2. Any x86 ISA. `scalar → AVX2 → AVX-512 (base/VNNI/BF16/VBMI)`, selected at import
   time from `/proc/cpuinfo`.
3. A fixed VRAM priority order: `KV pool → GPU prefill staging → speculative draft → activation workspace (∝ MBT)`.
   No feature may push that order back.
   > Revised 2026-09-20: (a) expert-layer residency is dropped — it measured poorly
   > (3.36 GiB per layer, and the gain does not justify the squeeze it puts on 32K prefill, so
   > the user retired it from the priority list); (b) the activation workspace is added — it
   > scales with the chunk (i.e. MBT) and was the real cause of two consecutive long-prompt OOMs,
   > yet it had never been listed.
4. One copy of the weights even across the multiple NUMA nodes of a single server:
   each NUMA node only reads and writes its local memory, maximising performance while
   saving memory.
5. **Bootstrapped development.** The 2026-10-06 session was developed and tested entirely inside the
   local "self-service" environment: the agent acted as the developer while the engine under
   development **was its own inference backend**. Every step, regression and A/B measurement was
   served by the very engine being changed -- which also exercised its stability under real live load
   (that service ran **16+ hours without a fault** during the session).

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

## Performance

* Metric: **aggregate throughput** -- `prefill (tok/s) = concurrency x prompt_tokens / TTFT`, `decode (tok/s) = concurrency x 1000 / median(TPOT)`
  (at C=1 this is just the per-stream rate; equivalently "total tokens of that phase / that phase's wall clock"), both from the same `vllm bench serve` run; `decode` uses the **median** TPOT.
  > **Correction (2026-10-04): the long/C=2 decode figure is largely a metric artefact.** `median TPOT`
  > counts **prefill steps in its denominator**, so a handful of mixed-in chunks wrecks it. On the clean
  > metric (**median ITL**) the real per-stream decode contention is only **1.13-1.77x** (see **B126**).
  > The MiMo 203.6 ms and GLM 700.8 ms figures are wrong for the same reason.
  > **What is real** is that mixing a long prefill with a decoding request on one engine starves the
  > decoder (measured, **B277**/**B279**): its inter-token latency is bounded below by the prefill
  > scheduled between its steps divided by prefill throughput. Injecting a 32768-token prompt into a
  > running decode took that request's ITL from a **64.8 ms median to about 8.4 s, sustained for the
  > whole 57.6 s prefill window** -- it gets exactly one token per step, and each step is filled by one
  > `MBT`-sized prefill chunk. Shrinking the chunk (`--long-prefill-token-threshold`) also destroys
  > prefill efficiency, because every chunk re-streams all forty layers of weights (a 2048-token chunk
  > runs at about 410 tok/s against roughly 1000 for 8192), so it makes the throughput **worse**
  > (1.25 -> 0.57 tok/s). There is no scheduler knob that fixes this.
  > * **Single-stream serving** (the 1M x `seqs=1` preset): `--max-num-seqs 1` serialises requests so
  >   nothing is mixed; per-stream ITL stays a steady ~65 ms. Pick by scenario: **1M x 1 / 512K x 2 /
  >   256K x 4**, plus the **`512K` x 4 combination production uses** (high ceiling + many short tasks;
  >   only about **2 full-length lanes** can run at once, the rest **queue** rather than fail).
  >   `seqs x maxlen <= pool` is a **saturation threshold, not a hard invariant**. See
  >   `docs/MODEL_GUIDES.md` section 0.1b.
  > * **Multi-user concurrency**: use **prefill/decode disaggregation**, not `max-num-seqs`,
  >   `max-num-batched-tokens` or `long-prefill-token-threshold`.
  > * **Root cause**: prefill throughput is **DMA / assembly-bound** -- every chunk re-streams about
  >   **3.84 GB** of weights per card per layer; the device-side **strided transpose** measures only
  >   **84 GB/s** against **1361 GB/s** contiguous, and assembly takes **250 ms/layer = 85%**, while that
  >   layer's MoE compute needs only **5-50 ms**. Faster prefill needs the strided transpose removed or the
  >   assembly fused, not scheduling.
  > * The old note that "`--max-num-seqs` must be >=2" holds only for **benchmarking**: with `seqs=1` a
  >   `C=2` run degrades to serial and depresses short-C=2 aggregate prefill to 54 (a **configuration**
  >   problem, **B124**). Use `seqs>=2` for concurrency benchmarks and `seqs=1` for single-stream serving.
* Dataset: **random tokens**, with `--random-input-len` pinned to short 128 / long 16384 and output 128; prefix caching on; 8 requests per cell with **a distinct seed per cell**.
* "actual tokens" = `total_input_tokens / completed` (includes a small chat-template overhead).
  Configuration details, criteria and the `MBT` trade-off live in `docs/TUNING_GUIDE.md` section 8 and `docs/EXPERIMENTS.md`.

| Model (optimal config) | prompt | actual tokens | conc. | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>TP=2 · util 0.85 · **MAXLEN 262144**<br>MBT 12288 · MTP off | short | 140 | 1 | **115.2** | **22.0** |
| | short | 140 | 2 | **140.8** | **27.8** |
| | long | 16,396 | 1 | **266.3** | **21.4** |
| | long | 16,396 | 2 | pending | pending |
| **DeepSeek-V4.1-Flash**<br>TP=2 · util 0.85 · **MAXLEN 262144**<br>MBT 8192 · **CED on** (long rows) · spec decode off | short | 128 | 1 | **254.3** | **19.8** |
| | short | 128 | 2 | **350.4** | **28.0** |
| | long | 16,384 | 1 | **903.9** | **19.3** |
| | long | 16,384 | 2 | **1204.2** | **15.4** |

Neither model uses speculative decoding (random tokens are its worst case, and the draft layer competes
with long context for VRAM).

## Quick start

```bash
# 1) upstream vLLM (this project is a plugin, no fork needed)
pip install vllm==2.5.0

# 2) this plugin (distribution name vllm-xtu-moe, NOT on PyPI) -- pick one
# (a) install the wheel attached to the GitHub Release (all 6 ISA variants included):
pip install ./vllm_xtu_moe-0.2.7-cp312-cp312-manylinux_2_34_x86_64.whl
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
| **v0.2.6** | **Bootstrapped development: INT8 activation x FP4 expert weights** — MXFP4 experts gain an **INT8 activation** path (the fp4 code set fits int8 exactly, so the **weight side is free**): kernel level **24.9 -> 15.1-15.7 ms/layer (1.54-1.62x)**; end to end (qfn, MXFP4-FP8) GSM8K **193/200 on both arms**, 194/200 per-question (net 0), 256K four-pin 4/4 + `stop` on both arms, Vision 23 agrees 18/23; **fallback-path regression fixed** (16-22% slower -> +1.49% instructions); **MTP on qfn (2.44x)**; qfn default threads **120 (5 cores/CCD)**; path off by default (`XIAOTU_MOE_INT8=1`) |
| **v0.2.5** | **GPU prefill memory rework** — the GPU operator consumes NUMA-sharded experts directly, dropping one layer of weight staging (**6.33 GiB saved for DeepSeek-V4.1-Flash, 3.16 GiB per card at TP=2**; per-layer staging 10.73 → **7.56 GiB/rank**; byte-identical results); **1M context + GPU prefill** (CED: KV 5437 → 2106 B/token, so 1M needs ~2.2 GiB); **Engram pinned over-allocation fixed (−75 GiB host)**; rebasing onto upstream restores **CED** (16k prefill 527.8 → **982.4** tok/s); **TP=2 uses GPU1+GPU2**; service logs named by PID |
| **v0.2.4** | **MiMo-V2.6-Flash-RL support + performance work** — MXFP4 experts over V4.1's engine path (TP=2 / 1M context / multimodal / MTP k=1 all measured); the **GPU prefill "zero means off" trap fixed** (long prefill 313 → **811** tok/s); **`--max-num-seqs` now defaults to 4**; decode metric switched to median ITL |
| **v0.2.3** | **Upstream tracking + GLM/MiMo MTP** — patch stack rebased onto upstream `133b71e0b` (11 patches / 40 files); **GLM-5.3-Flash MTP wired up, ON by default** (`SPEC_K=1`, decode 21.9 → 22.6 tok/s at the cost of a 27% smaller KV pool); **MiMo-V2.5 (310B/15B) runs end-to-end on one A100-40GB** with MTP k=1 at +10% decode; GLM memory contract re-calibrated (`GPU_UTIL` 0.85 → **0.82**) |
| **v0.2.2** | **GLM-5.3-Flash support** — FP8 GPU prefill wired up (4K-prompt TTFT 29.3 → 22.8 s), delivered as 256K × 2 concurrent; fixes an e4m3 subnormal-decode defect and adds an all-codeword gate |
| **v0.2.1** | **Major CPU-engine performance work** — the CPU MoE engine now **beats `lk_moe` on every real shape**; DeepSeek-V4-Flash benefits too |
| v0.2 | DeepSeek-V4.1-Flash end-to-end (1M context + GPU prefill + speculative decoding) plus CPU-prefill path optimisation |
| v0.1.0 | First public release: hybrid mode (CPU experts + GPU rest), AVX2 / AVX-512 multi-ISA, DeepSeek-V4 family |

Details, performance comparisons and parameter changes:
[**v0.2.6**](RELEASE_NOTES_v0.2.6.md) · [**v0.2.5**](RELEASE_NOTES_v0.2.5.md) · [**v0.2.4**](RELEASE_NOTES_v0.2.4.md) · [**v0.2.3**](RELEASE_NOTES_v0.2.3.md) · [**v0.2.2**](RELEASE_NOTES_v0.2.2.md) · [**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
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
| [`docs/FP4_INT8_HARDWARE_OUTLOOK.md`](docs/FP4_INT8_HARDWARE_OUTLOOK.md) | **Public hardware outlook on 4-bit inference formats** (FP4 vs INT4, with sources) and what it implies for **CPU-first MoE serving** |
| [`docs/INT8_ACTIVATION_FOR_MXFP4_EXPERTS.md`](docs/INT8_ACTIVATION_FOR_MXFP4_EXPERTS.md) | **INT8 activation x MXFP4 expert weights** — what the path solves, kernel-level gains, the end-to-end semantic acceptance protocol, how to enable it, and known limitations |

**For developers** — architecture and upstream integration, the GPU-prefill implementation,
upstream drift, tuning logs and all internal reports are **internal development documents and are
not published with this repository** (they exist only in the local working copy).

---

## About the author

Independent researcher. Chief designer and operator of a **10-PFLOP/s (double-precision) supercomputer**, with a background in high-performance computing.

That is where this project's priorities come from: **NUMA topology and memory layout, bandwidth-bound kernels, and host-side data movement** - which is exactly where running giant MoE models on ordinary servers is decided today.

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
