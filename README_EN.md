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
3. **A fixed VRAM priority order.** `KV pool → GPU prefill staging → speculative draft → activation workspace (∝ MBT)`.
   > **Revised 2026-09-20**: (a) **expert-layer residency is dropped** — it measured poorly
   > (3.36 GiB per layer, and the gain does not justify the squeeze it puts on 32K prefill, so
   > the user retired it from the priority list); (b) **the activation workspace is added** — it
   > scales with the chunk (i.e. MBT) and was the real cause of two consecutive long-prompt OOMs,
   > yet it had never been listed.
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

## Performance

* Metric: `prefill (tok/s) = prompt_tokens / TTFT`, `decode (tok/s) = 1000 / TPOT` (both converted from the same `vllm bench serve` run).
* Dataset: real ShareGPT conversations, filtered into a "short" and a "long" band; **prefix caching on**; C=1 and C=4 use **disjoint prompt slices** (otherwise C=4 reads the cache the C=1 run left behind); every cell is warmed up first.

| Model (optimal config) | prompt | actual tokens | conc. | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>TP=2 · util 0.82 · GPU prefill<br>KV capped 2 GiB · MBT 4096 | short | 151 | 1 | **114.8** | **21.7** |
| | short | 126 | 4 | **36.5** | 8.2 |
| | long | 4,538 | 1 | **178.3** | 21.3 |
| | long | 4,518 | 4 | **92.9** | 2.2 |
| **MiMo-V2.5**<br>1×A100 · maxlen 16K · KV capped 4 GiB<br>MTP k=1 · GPU prefill | short | 153 | 1 | 70.9 | **18.0** |
| | short | 153 | 4 | measuring | measuring |
| | long | 4,661 | 1 | **156.2** | 15.6 |
| | long | 4,819 | 4 | **83.2** | 1.9 |
| **DeepSeek-V4.1-Flash**<br>TP=2 · dspark k=5 · GPU prefill<br>KV capped 0.5 GiB · MBT 8192 | short | 84 | 1 | **217.7** | **21.3** |
| | short | 84 | 4 | measuring | measuring |
| | **long** | 4,796 | 1 | **330.3** | **24.7** |
| | **long** | 4,796 | 4 | measuring | measuring |

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
