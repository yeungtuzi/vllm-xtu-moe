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

## Project highlights

1. Any MoE model. Not tied to one architecture generation. DeepSeek-V4 / V4.1 work
   end to end; GLM-5.3-Flash runs end to end too with an FP8 GPU prefill path; MiMo-V2.5
   runs end to end on a single card.
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

* Metric: `prefill (tok/s) = prompt_tokens / TTFT`, `decode (tok/s) = 1000 / TPOT` (both converted from the same `vllm bench serve` run).
* Dataset: **random tokens**, with `--random-input-len` pinned to short 128 / long 16384 and output 128; **prefix caching on**; **a distinct seed per cell** (otherwise C=2 reads the cache C=1 left behind); 8 requests per cell.
  > Two differences from the previous revision: (1) ShareGPT was replaced by the random dataset -- the goal is prefill/decode throughput at a **controlled input length**, and ShareGPT's length distribution cannot produce a "long = 16384" column; (2) the concurrency bands changed from C=1/C=4 to **C=1/C=2**.
  > Note the random dataset is **random tokens**, the worst case for speculative decoding (predictability ~0), so this table does **not** represent the real gain from enabling MTP.
* "actual tokens" = `total_input_tokens / completed` (includes a small chat-template overhead).

| Model (optimal config) | prompt | actual tokens | conc. | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>TP=2 · util 0.82 · GPU prefill<br>KV capped 2 GiB · MBT 4096 · MTP off | short | 140 | 1 | **115.2** | **22.0** |
| | short | 140 | 2 | **70.4** | 13.9 |
| | long | 16,396 | 1 | **155.3** | **21.5** |
| | long | 16,396 | 2 | **119.5** | 1.7 |
| **DeepSeek-V4.1-Flash**<br>TP=2 · GPU prefill<br>KV capped 1.0 GiB · MBT 16384 · spec decode off | short | 128 | 1 | **254.3** | **19.8** |
| | short | 128 | 2 | **175.2** | 14.0 |

> **MiMo-V2.5 is still supported** (see "Supported models"), but since its successor MiMo-2.6
> is about to be released, this table **no longer lists its performance data**.

## Long context (256K) long prefill

The table above uses a 32K budget. The one below is measured **with a required 256K context**
(`MAXLEN=262144`), giving everything else to GPU prefill. **Single 16384-token random prompt,
C=1, prefix caching on.**

| Model | prefill (tok/s) | decode (tok/s) | TTFT | Key config | Criteria |
|---|---|---|---|---|---|
| **GLM-5.3-Flash** | **233.6** | **21.4** | 70,203 ms | TP=2 · util 0.85 · **MBT 8192** · `KV_CACHE_BYTES=5113807360` (4.76 GiB) | `[fp8-asm]=252`, `DISABLED=0`, `illegal=0` |
| **DeepSeek-V4.1-Flash** | **115.8** | **18.7** | 141,618 ms | TP=2 · util 0.85 · **MBT 4096** · `KV_CACHE_BYTES=8031830016` (7.48 GiB) | `DISABLED=0`, `aten::new_empty=0`, `illegal=0` |

* The KV cap is set to the **engine-derived requirement** (GLM 19,505 B/token, V4.1 30,639
  B/token) with **no multiplicative margin** -- at 256K a 10% margin is 0.48 GiB and was measured
  to OOM.
* **MBT is the critical knob at 256K**: GLM OOMs at `MBT=16384` by 120 MiB and succeeds at
  `MBT=8192`; V4.1 hits an `aten::new_empty` allocation failure at `MBT=16384` and succeeds at
  `MBT=4096`. Mechanism in `docs/PREFILL_KNOWN_ISSUES.md`.
* Neither model uses speculative decoding here (random tokens are its worst case, and the draft
  layer competes with long prefill for VRAM).
* **MiMo-V2.5 cannot run at 256K**: its KV is about 248 KiB/token (GLM's is 19 KiB), so 256K
  needs **61.9 GiB/rank** against a 35.5 GiB budget -- a factor of 1.74, determined by
  architecture rather than configuration.

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
