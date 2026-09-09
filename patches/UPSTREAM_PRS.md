# Upstream PR 计划(D1–D8 决策后的执行状态)

> 目标仓库: `vllm-project/vllm`,基线已 rebase 到 `main@1454b71`
> 我们的 fork: `yeungtuzi/vllm`
> **D6 已改:PR 由我生成(draft),你检查后点 "Ready for review"**。

| PR | 分支 | 规模(vs 最新 main) | Draft PR | 状态 |
|---|---|---|---|---|
| PR1 | `xtu/pr1-experts-load-device` | 3 文件 +59/−2 | [#56118](https://github.com/vllm-project/vllm/pull/56118) | draft,待你 review |
| PR2 | `xtu/pr2-fp8-sm80-o-proj` | 4 文件 +422/−12 | [#56119](https://github.com/vllm-project/vllm/pull/56119) | draft,待你 review |
| PR3 | `xtu/pr3-sm80-port` | 21 文件 +6350/−119 | [#56120](https://github.com/vllm-project/vllm/pull/56120) | draft;**A100 运行时验证待补** |
| RFC | — | issue | — | 草稿见 `rfc_layerwise_gpu_prefill.md`,待你发 |

**你的动作**:逐个点开 → 检查代码/描述 → 点 **"Ready for review"** 即提交给 maintainer。
若想撤销,点 "Convert to draft" 或 Close 即可。

**重新生成(上游又变了时)**:
```bash
bash scripts/check_upstream_drift.sh          # 先看漂移
# 需要时:重新 rebase 三个分支并强推(步骤见 results.txt「上游推进」段)
```

**提交顺序建议**: PR2 → PR3(PR3 的 o_proj 正确性依赖 PR2)→ PR1(独立,可随时)。
三者互不冲突(只有 `vllm/envs.py` 在 PR1 与 PR3 各加了一组互不重叠的变量,
合入时可能有一次 trivial 冲突)。

---

## PR1

**标题**: `[MoE] add VLLM_EXPERTS_LOAD_DEVICE=cpu for GPU/CPU mixed expert placement`

**一键创建链接**(复制到浏览器):
<https://github.com/vllm-project/vllm/compare/main...yeungtuzi:vllm:xtu/pr1-experts-load-device?expand=1>

**描述**(粘贴到 PR body):

```markdown
A MoE model whose expert weights exceed device memory cannot be constructed at
all on a single 40 GB GPU (e.g. DeepSeek-V4-Flash has ~137 GiB of fp4 experts),
and the MXFP4 oracle always prefers a GPU backend, which then fails with
"b_q_weight is not on GPU" once the weights live on the host.

This adds an opt-in switch that keeps routed-expert weights on the host while
everything else (attention, router, shared experts) stays on the compute device:

- `vllm/envs.py`: new `VLLM_EXPERTS_LOAD_DEVICE` ("gpu" default, "cpu" opt-in).
- `routed_experts.py`: build expert weights under `torch.device("cpu")` when the
  switch is set, so they are never materialized on the GPU.
- `fused_moe/oracle/mxfp4.py`: with the switch set, select the CPU backend
  unconditionally and skip the AMX prepack, because the consumer is an
  out-of-tree CPU engine that reads the raw packed weights and e8m0 scales.

Default behaviour is unchanged (`VLLM_EXPERTS_LOAD_DEVICE=gpu`); with `gpu` the
oracle takes exactly the same path as before.

### Testing
- 2x A100-PCIE-40GB + AMD EPYC 9654, DeepSeek-V4-Flash, out-of-tree CPU MoE
  engine ([vllm-xtu-moe](https://github.com/yeungtuzi/vllm-xtu-moe)):
  the model constructs and serves where it previously OOM'd at construction.
- `VLLM_EXPERTS_LOAD_DEVICE=gpu` (default): unchanged behaviour on the existing
  GPU backends.
```

---

## PR2

**标题**: `[DS-V4] fix o_proj fp8 einsum on SM8.x (Marlin repacking + portable kernel)`

**一键创建链接**:
<https://github.com/vllm-project/vllm/compare/main...yeungtuzi:vllm:xtu/pr2-fp8-sm80-o-proj?expand=1>

**描述**:

```markdown
DeepSeek-V4's `o_proj` consumes `wo_a.weight` directly through a fused
per-group fp8 einsum (`apply_weights` is bypassed), so the weight must keep its
on-disk `[N, K]` layout with block scales. On Ampere (SM 8.0) the FP8 quant
method selects Marlin, which repacks that weight into its opaque int32 layout
and renames the block scales to `weight_scale_inv`; the einsum then reads
garbage. Independently, `vllm.utils.deep_gemm.fp8_einsum` has no SM8.x kernel at
all, so the op cannot run on A100 regardless of the packing.

- `quantization/fp8.py`: when the layer is a `bmm`-style consumer (`is_bmm`) and
  Marlin was selected, pre-dequantize the block-FP8 weight to bf16 in place
  (keeping `[N, K]` and the original scale layout) and turn Marlin off.
- `quantization/utils/fp8_utils.py`: add `_e4m3_uint8_to_f32` /
  `_f32_to_e4m3_uint8` (uint8-typed fp8 conversion; the `float8e4nv` Triton
  dtype is not available on Ampere).
- `models/deepseek_v4/nvidia/ops/fp8_einsum.py`: new portable Triton fp8 einsum
  for SM8.x/SM12x; `o_proj.py` dispatches to it when the arch has no DeepGEMM
  fp8_einsum, keeping the existing SM90/SM100/SM110 recipes untouched.

The portable kernel is taken from the vLLM fork
[Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x) (Apache-2.0); SPDX headers
retained.

### Testing
- A100-PCIE-40GB + DeepSeek-V4-Flash: `o_proj` output is numerically correct
  (previously the model emitted garbage) and generation is coherent.
- SM90/SM100/SM110 paths are not modified (same recipes as before).
```

---

## PR3

**标题**: `[DS-V4][SM80] portable Triton fallbacks so DeepSeek-V4 runs on Ampere/Ada`

**一键创建链接**:
<https://github.com/vllm-project/vllm/compare/main...yeungtuzi:vllm:xtu/pr3-sm80-port?expand=1>

**描述**:

```markdown
DeepSeek-V4 in mainline targets SM90+/SM100 only: sparse MLA goes through
FlashMLA/DeepGEMM, fp8 einsum through DeepGEMM, MHC prenorm through tilelang
and cutedsl, and several ops use the `float8e4nv` Triton dtype. On SM 8.0
(A100/A800) all of those are unavailable, so the model cannot even build.

This adds portable Triton fallbacks selected by compute capability and leaves
the existing SM90/SM100/SM110 paths untouched:

- `v1/attention/backends/mla/{sparse_mla_env,sparse_mla_kernels}.py`: portable
  sparse-MLA Triton kernels + `VLLM_TRITON_MLA_SPARSE*` switches.
- `models/deepseek_v4/nvidia/ops/{sm12x_deep_gemm_fallbacks,sm12x_mqa}.py`:
  DeepGEMM / MQA fallbacks for SM8.x and SM12x.
- `models/deepseek_v4/nvidia/flashmla.py`: SM8.x dispatch to the Triton path.
- `common/ops/*`: uint8-based fp8 conversion instead of `tl.float8e4nv`; new
  cache helpers for the portable decode/prefill path.
- `utils/{deep_gemm,import_utils}.py`: SM8.x branches; `has_cutedsl()` false on
  SM8.x (cutedsl emits SM90+ PTX only).
- `compressor.py` / `sparse_mla.py` / `mhc/tilelang.py` /
  `warmup/flashinfer_sparse_mla_warmup.py` / `mla/{indexer,sparse_swa}.py`:
  capability gates and scheduler-metadata skips.
- `envs.py`: `VLLM_DEEPSEEK_V4_INDEXED_D512_{SPLIT,CHUNKED}_PREFILL` and the
  `VLLM_TRITON_MLA_SPARSE*` switches.

Files are taken byte-for-byte from the vLLM fork
[Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x) (Apache-2.0) and only
adapted to mainline API changes (e.g. `combine_topk_swa_indices` now returns
`(idx, lens)`); SPDX headers retained. Debug-only instrumentation from our
development branch has been dropped for this PR.

Depends on: #PR2 (o_proj fp8 fix) for numerically correct DeepSeek-V4 output on
SM8.x.

### Testing
- A100-PCIE-40GB (SM 8.0) + DeepSeek-V4-Flash: the model builds and generates
  coherent text; end-to-end prefill/latency measurements are in the
  [vllm-xtu-moe experiment report](https://github.com/yeungtuzi/vllm-xtu-moe/blob/main/vllm-xiaotu-moe/docs/EXPERIMENT_REPORT.md).
- SM90/SM100 paths unchanged (capability gates only).
```

---

## 你提交时只需要

```bash
# 或者直接点上面的 compare 链接,粘描述即可
gh pr create --repo vllm-project/vllm --base main \
  --head yeungtuzi:xtu/pr2-fp8-sm80-o-proj \
  --title "[DS-V4] fix o_proj fp8 einsum on SM8.x (Marlin repacking + portable kernel)" \
  --body-file <(sed -n '/^```markdown$/,/^```$/p' ...)
```

（三个分支都已经在 `yeungtuzi/vllm` 上,`gh pr create` 会自动识别 head。）

---
## 修订记录

- **2026-09-09(第 1 版)** — 按 D1–D8 决策产出:PR1/PR2/PR3 分支已推送到
  `yeungtuzi/vllm`,标题/描述/测试说明齐备,等用户点提交(D6b)。
  依据:`docs/EXPERIMENT_REPORT.md` §4.1 的 D1–D8 决策 + 用户 2026-09-09 回复。
