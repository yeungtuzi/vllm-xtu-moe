# vllm-xtu-moe

> **XTU = X Transformers Unity** (pronounced *"xiao tu"*, Chinese for "little rabbit").
> An **upstream-vLLM plugin** that keeps **MoE expert weights on the CPU** while
> **attention and long prefill stay on the GPU** — so models whose experts do not
> fit in VRAM can still be served.

[中文](README.md) · English (default)

---

## The problem it solves

Modern MoE models carry tens to hundreds of GB of **expert weights**
(DeepSeek-V4 ≈ 137 GiB, Qwen3.8-Flash-Next 185 GB), far beyond a single GPU's
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
- different routing styles (GLM / DeepSeek / Qwen / Mixtral) are all supported.

## Support matrix

| Weight format | Engine kernel | Status |
|---|---|---|
| **BF16 / FP16** (unquantized) | `MOE_BF16` / `MOE_FP16` | ✅ |
| **FP8 e4m3 + 128×128 blocks** (vLLM `kFp8Static128BlockSym`) | `MOE_FP8` | ✅ layer-level numerics verified (48-layer self-check ≤1.4e-4) plus real end-to-end runs (Qwen3-30B-A3B-FP8 on one GPU, Qwen3.8-Flash-Next-FP8 on 2×A100 TP=2+EP) |
| **MXFP4** (e2m1 + e8m0 block 32) | `MOE_MXFP4` | ✅ |
| **NVFP4** | `MOE_NVFP4` | ✅ engine side |
| **INT4 / WNA16** (GPTQ / compressed-tensors group quantization) | `MOE_WNA16` | ✅ symmetric (zero point 8) works: the checkpoint layout is repacked once at engine construction, validated end to end on the real `Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`; asymmetric zero points and AWQ's N-packed layout raise an explicit error, see `docs/KNOWN_LIMITATIONS.md` |
| INT8 W8A8 | — | ❌ not implemented in the engine |

**Routing**: the plugin reuses upstream vLLM's router objects (softmax,
sigmoid + `noaux_tc`, `sqrtsoftplus`, grouped top-k, custom routing functions),
so GLM, DeepSeek and Qwen style routing all select experts correctly.

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
export XIAOTU_MOE_SINGLECOPY=1          # one copy of the weights (lower memory)
vllm serve <MODEL_DIR> \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

Self-check (prints the engine ISA variant, the active integration shims and the
registered backends):

```bash
VLLM_EXPERTS_LOAD_DEVICE=cpu python -m vllm_xiaotu_moe.mainline_shims
VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py   # backend selection probe
```

Full install notes, environment variables and per-model commands:
**[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**.

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

## License and third parties

Apache-2.0 for this project and the bundled engine. Third-party code (the SM80
port taken from `Lvllmds4-x`) keeps its Apache-2.0 SPDX attribution; no code from
the closed-source `lk_moe` is included. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Author

**大河马 (BigHippo)** `<dahema@me.com>`, with DeepSeek Harness assistance.
