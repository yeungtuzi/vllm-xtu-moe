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

> **📌 Current release: v0.2.7** (2026-10-09) — Optimized engine: CPU→GPU DMA buffers cut, 11 GB of VRAM saved in total, no performance loss
> [Release notes](RELEASE_NOTES_v0.2.7.md) · [Work report](docs/WORK_REPORT_2026-10-09.md)

---

## Project highlights

1. **Runs very large MoE models on a single GPU (>40 GB) with enough system RAM** (verified on DeepSeek-V4.1-Flash, GLM-5.3-Flash, MiMo-V2.6-Flash-RL and Qwen3.8-Flash-Next); other MoE models are presumed to work.
2. **Automatic CPU/GPU prefill split**: for long prompts (threshold configurable) the weights are streamed into VRAM with DMA double buffering, **saturating the PCIe link** (verified on 2x Tesla A100 40GB: both PCIe 4.0 x16 lanes reach the **25 GB/s** ceiling), keeping GPU-side prefill performance as high as possible.
3. **Supports older GPUs such as SM80**, and **NVFP4 / MXFP4** quantized weights; where the hardware has no native support, the weights are **dequantized and computed on the best available path**.
4. Supports **AVX-512 VNNI and other ISAs**, picking the best CPU compute backend automatically.
5. **Hardware-aware NUMA slicing**: the process is spread evenly across cores according to the NUMA topology, and the MoE weights are sharded to match, so **every node's compute only touches local memory** (no cross-socket traffic) while **total memory stays at a single copy of the weights**.

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

* Metric: **aggregate throughput** -- `prefill (tok/s) = concurrency x prompt_tokens / TTFT`, `decode (tok/s) = (output tokens - concurrency) / (last-token time - first-token time)`
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
pip install ./vllm_xtu_moe-0.2.3-cp312-cp312-manylinux_2_34_x86_64.whl
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
