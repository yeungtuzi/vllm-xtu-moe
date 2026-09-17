# vllm-xtu-moe

> **XTU = X Transformers Unity** (pronounced *"xiao tu"*, Chinese for "little rabbit").
> An **upstream-vLLM plugin** that keeps **MoE expert weights on the CPU** while
> **attention and long prefill stay on the GPU** — so models whose experts do not
> fit in VRAM can still be served.

[中文](README.md) · English (default)

> **📌 Current release: v0.21.0** (2026-09-17) — **DeepSeek-V4.1-Flash (748B) support**:
> the full chain (**1M context + GPU prefill + speculative decoding**) is verified
> end to end; GPU prefill is **2.0-2.8x faster** than the CPU path, plus a full official
> `vllm bench serve` acceptance sweep (6 prompt lengths × C=1/2/4/8).
> [Release notes](RELEASE_NOTES_v0.21.0.md) ·
> [Releases](https://github.com/yeungtuzi/vllm-xtu-moe/releases/tag/v0.21.0)

---

## Verified models

| Model | Status | Key numbers (measured on our machine) |
|---|---|---|
| **DeepSeek-V4.1-Flash** (748B) | ✅ **v0.21.0** | full chain (**1M context + GPU prefill + speculative**); host peak **629.4 GiB (-42%)**; KV **6.72M tokens (6.41x 1M concurrency)**; single stream **16.64 tok/s / TPOT 36.75 ms**; greedy ×2 **5/5 byte-identical** |
| **DeepSeek-V4-Flash** | ✅ v0.2.0 | end to end on the mainline (serve / `bench_lat` C=1/2/4 / numeric gate `OK=7 BAD=1` / startup self-check) |

* **Why V4.1-Flash works**: 38.5% of its weights are a **pure lookup table**
  (Engram, 183 GiB, only ~12 KB of host traffic per token) and the official production
  stack keeps that table in **host memory** too — the same idea as our `XIAOTU_PLE_CPU=1`.
  The rest (CED / CSA2 / FP4 KV / DSpark speculation) comes from upstream vLLM's
  `deepseek_v41`; this plugin supplies the **MoE layer**. The day-of-release feasibility
  analysis is in [`docs/V41_FLASH_ANALYSIS.md`](docs/V41_FLASH_ANALYSIS.md); the landed
  measurements and config are in
  [`RELEASE_NOTES_v0.21.0.md`](RELEASE_NOTES_v0.21.0.md) and `docs/RUNBOOK.md` §5.9.

---

## The problem it solves

Modern MoE models carry tens to hundreds of GB of **expert weights**
(DeepSeek-V4 ≈ 137 GiB), far beyond a single GPU's
VRAM. Meanwhile attention, KV cache, the router and shared experts are touched
by *every* token and belong on the GPU. Splitting the two is the only practical
deployment path.

Upstream vLLM already provides the *interface* for CPU-resident experts
(`FusedMoEFactory` + quantized backend slots), but the CPU kernels it ships
(MXFP4 / FP8 / INT4 / INT8) **all require Intel AMX**. On x86 hosts without AMX
(e.g. AMD EPYC) there is therefore **no usable CPU kernel** for those formats.

This plugin plugs the **xiaotu CPU engine** (AVX-512 VNNI/BF16, no AMX needed)
into those slots, so:

- any MoE model built on `FusedMoEFactory` runs attention on the GPU and experts
  on the CPU, as long as its weight format is supported;
- no vLLM fork and no model-code changes are required;
- different routing styles (GLM / DeepSeek / Mixtral) are all supported.

## Support matrix

| Weight format | Engine kernel | Status |
|---|---|---|
| **BF16 / FP16** (unquantized) | `MOE_BF16` / `MOE_FP16` | ✅ |
| **FP8 e4m3 + 128×128 blocks** (vLLM `kFp8Static128BlockSym`) | `MOE_FP8` | ✅ layer-level numerics verified (48-layer self-check ≤1.4e-4) |
| **MXFP4** (e2m1 + e8m0 block 32) | `MOE_MXFP4` | ✅ |
| **NVFP4** | `MOE_NVFP4` | ✅ engine side |
| **INT4 / WNA16** (GPTQ / compressed-tensors group quantization) | `MOE_WNA16` | ✅ symmetric (zero point 8) works: the checkpoint layout is repacked once at engine construction; asymmetric zero points and AWQ's N-packed layout raise an explicit error, see `docs/KNOWN_LIMITATIONS.md` |
| INT8 W8A8 | — | ❌ not implemented in the engine |

**Routing**: the plugin reuses upstream vLLM's router objects (softmax,
sigmoid + `noaux_tc`, `sqrtsoftplus`, grouped top-k, custom routing functions),
so GLM and DeepSeek style routing all select experts correctly.

**Activations**: packed-layout gated activations (SILU, `SWIGLUOAI_UNINTERLEAVE`)
plus `swiglu_limit` / `alpha` / `beta` (the clamped SwiGLU used by GLM-5.x,
DeepSeek-V4 and MiniMax). Interleaved gate/up layouts (`SWIGLUOAI`, e.g.
gpt-oss) are **rejected** rather than silently mis-computed.

**Instruction sets**: the bundled engine picks the best variant at runtime from
`/proc/cpuinfo`: `scalar → avx2 → avx512_base → avx512_vnni → avx512_bf16`.
No AMX is required; on AMX-capable hosts upstream's own CPU kernels remain
available.

## Install and quick start

```bash
# 1) upstream vLLM (this is a plugin, not a fork)
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130

# 2) this plugin (build the engine extension first when installing from source)
git clone https://github.com/yeungtuzi/vllm-xtu-moe.git
cd vllm-xtu-moe
PYTHON=$(which python) bash scripts/build_engine_variants.sh   # ~90 s
pip install -e .

# 3) serve a MoE model whose experts do not fit in VRAM
export VLLM_EXPERTS_LOAD_DEVICE=cpu     # keep expert weights on the CPU
vllm serve <MODEL_DIR> \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \

# 4) (optional, recommended) also put **long prefill on the GPU** — measured
#    2.0-2.8x faster on DeepSeek-V4.1-Flash. Two prerequisites:
#    ① threshold >= 3072 tokens (below that the CPU path is cheaper);
#    ② you MUST cap the KV pool explicitly, otherwise vLLM fills VRAM and GPU
#       prefill is silently rejected layer-by-layer (falls back to CPU).
vllm serve <MODEL_DIR> \
  --tensor-parallel-size 2 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 8192 \
  --kv-cache-memory 4294967296 \
  --gpu-memory-utilization 0.55 \
  --speculative-config dspark
# VRAM recipe (TP=2 / MBT=8192): non-KV 10 + KV 4 + staging 7.2 + first-request
# growth 7.7 + activations ~= 33 GiB. Full table: docs/RUNBOOK.md §5.9.
# To check it is really on GPU: the log must contain `GPU prefill ACTIVE`.
  --enforce-eager
```

Self-check (prints the engine ISA variant, the active integration shims and the
registered backends):

```bash
VLLM_EXPERTS_LOAD_DEVICE=cpu python -m vllm_xiaotu_moe.mainline_shims
VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py   # backend selection probe
```

> ⏸️ **Support not claimed at this time**: this model's measurements predate the v0.2 changes (execution model / small-batch path / EP storage sharding) and have **not been re-verified on the current code**. See `docs/HANDOFF_v0.2.md` §5.1 for the re-verification plan.

Recommended parameters and measurements for DeepSeek-V4-Flash:
**[`docs/TUNING_REPORT.md`](docs/TUNING_REPORT.md)**.

Full install notes and environment variables:
**[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**.
Step-by-step guides plus preliminary measurements for **DeepSeek-V4-Flash**:
**[`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md)**.

## Measured performance (v0.21.0)

> Machine: 2x AMD EPYC 9654 (192 cores) / 3x A100-40GB / DDR5-4800 24 channels.
> Protocol: **TP=2**, fully warmed up, unique prompts (no prefix-cache hits),
> official `vllm bench serve`. Raw data:
> [`report/tuning/logs/bench_serve_acc2/`](report/tuning/logs/bench_serve_acc2/).

### GPU prefill vs CPU prefill (DeepSeek-V4.1-Flash, client-side TTFT)

| prompt | CPU prefill | **GPU prefill** | speedup |
|---|---|---|---|
| ~3.7K | 24.17 s (151 tok/s) | **12.11 s (302 tok/s)** | **2.0x** |
| ~7.0K | 42.17 s (165 tok/s) | **15.19 s (460 tok/s)** | **2.8x** |

Cost model (measured): **~8.9 s fixed per chunk + 0.79 ms/token**. The fixed term is
**143.6 GiB/rank** of expert weights moved per chunk (TP=2, 40 layers x 3.589 GiB),
so **the larger `--max-num-batched-tokens`, the better**.

### End to end (official `vllm bench serve`; TP=2 / MBT=8192 / GPU prefill / KV capped at 4 GiB)

| prompt | TTFT C=1 | TTFT C=8 | total tok/s C=1 | total tok/s C=8 | TPOT C=1 |
|---|---|---|---|---|---|
| 32 | 0.47 s | 3.5 s | 37 | 95 | 36.2 ms |
| 256 | 1.98 s | 10.9 s | 55 | 134 | 43.6 ms |
| 1024 | 10.0 s | 23.6 s | 65 | 242 | 64.6 ms |
| 4096 | 12.8 s | 46.4 s | 200 | 379 | 66.3 ms |
| 16384 | 34.5 s | 170 s | 399 | 455 | 54.4 ms |
| 32768 | 70.9 s | 333 s | 420 | 454 | 58.4 ms |

**Reading it**: ① total throughput for long prompts **saturates at ~420-455 tok/s**
(the ceiling is the fixed per-chunk GPU-prefill cost); ② **TTFT is linear in length and
strongly coupled to concurrency** — prefill is a serial shared resource; ③ short prompts
(32/256) stay on the CPU and have sub-second TTFT.

---

## How it works (one paragraph)

```
GPU:  attention · KV cache · router · shared experts · experts during long prefill (optional)
CPU:  routed-expert weights and compute (xiaotu engine, AVX-512)
```

Expert weights are constructed on the CPU so the model can be built at all; the
oracle is steered to this plugin's CPU backend in mixed mode; long prefill can
switch per layer to streaming that layer's weights to the GPU
([`docs/GPU_PREFILL.md`](docs/GPU_PREFILL.md)). Because the required upstream
integration points are not merged yet, the plugin supplies them via
`vllm_xiaotu_moe/mainline_shims.py`, so it works on stock vLLM.
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

## Documentation

| Document | Contents |
|---|---|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | install, env vars / CLI flags, per-model commands, troubleshooting |
| [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) | **guide and preliminary measurements for DeepSeek-V4-Flash** |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | mixed-mode and mainline integration design |
| [`docs/GPU_PREFILL.md`](docs/GPU_PREFILL.md) | layerwise GPU prefill |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | measured hardware, throughput, latency, ablations |
| [`docs/UPSTREAM.md`](docs/UPSTREAM.md) | upstream changes needed and the corresponding PRs |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | format / ISA / backend generality plan |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | known limits and unsupported cases |
| [`patches/`](patches/) | upstream PR descriptions, patch snapshots and an RFC draft |
| [`report/`](report/) | benchmark data and figures |

## Repository layout

```
vllm-xtu-moe/
├── vllm_xiaotu_moe/     # plugin: mainline integration layer
│   ├── mixed_experts.py #   generic CPU experts backend (BF16/FP8/MXFP4/INT4)
│   ├── mainline_shims.py#   integration shims for stock vLLM
│   ├── hybrid_model.py  #   DeepSeek-V4 model-level override (optional path)
│   └── gpu_prefill.py   #   layerwise GPU prefill
├── xiaotu_moe/          # bundled CPU engine (Python bindings + C++ kernels + runtime ISA selection)
├── scripts/             # build, benchmarks, numeric tests, end-to-end smoke tests
├── patches/             # upstream PRs / patches / RFC
├── docs/                # documentation
└── report/              # benchmark data and figures
```

## Known limitations

- **GLM-5.3-Flash requires SM90+**: its MLA dimensions
  (`qk_nope=256 / rope=0 / v=256`) have no attention backend in upstream vLLM;
  this is unrelated to the MoE backend. Expert-layer numerics were verified with
  real weights. See [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).
- **FP8 CPU kernel performance** is not yet optimized — correctness first.
- **Expert parallelism (expert_map) is not supported**; TP>1 uses weight sharding.
- **Interleaved gate/up layouts** (`SWIGLUOAI`, gpt-oss family) are unsupported.
- **GPU prefill chunk ceiling is `--max-num-batched-tokens 8192`**
  (DeepSeek-V4.1-Flash on A100-40GB): `16384` OOMs, and the OOM happens in
  **attention** (`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`), not in the MoE —
  a 16K chunk's activations plus staging do not fit.
- **Enabling GPU prefill requires an explicit KV cap** (`--kv-cache-memory`): by default
  vLLM fills VRAM up to `--gpu-memory-utilization`, so the prefill staging is rejected
  layer by layer and it silently falls back to the CPU. See `docs/RUNBOOK.md` §5.8.
- **③ CED prefill shortcut: mechanism implemented, disabled by default, no gain today.**
  Every conflict was traced to **two upstream invariants**: `mhc/tilelang.py:345` asserts
  `x.shape == (num_tokens, hidden_size)` (a layer may not return fewer rows than the step's
  `num_tokens`), and `positions` comes from a runner-level global buffer that a plugin-level
  layer wrapper cannot reach. Getting the actual gain therefore requires threading the
  effective token count through **upstream vLLM**. Evidence: `report/tuning/NOTES.md` §571-§604.

## License and third parties

Apache-2.0 for this project and the bundled engine. Third-party code (the SM80
port taken from `Lvllmds4-x`) keeps its Apache-2.0 SPDX attribution; no code from
the closed-source `lk_moe` is included. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Author

**大河马 (BigHippo)** `<dahema@me.com>`, with DeepSeek Harness assistance.
