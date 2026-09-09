# 主线改动清单(SM80 支持 + 通用混合模式)

基线:`vllm-project/vllm` @ `6c73b08dec2af5052288169663549687ba61f330`
补丁包:`patches/mainline_sm80_mixed_mode.patch`(27 文件,+6977/-133)
参考实现:fork `Lvllmds4-x`(guqiong96,vLLM `yhfgyyf/vllm-deepseek-v4-sm89` 系)
—— **纪律 #2:凡 fork 已有的,直接采用 fork 的实现,不自己造。**

## A. 通用 GPU/CPU Mixed Mode(与模型无关)

| 文件 | 改动 | 作用 |
|---|---|---|
| `vllm/envs.py` | +`VLLM_EXPERTS_LOAD_DEVICE` | `cpu` 时专家参数落 CPU |
| `.../fused_moe/routed_experts.py` | `with torch.device("cpu")` | 按上面开关在 CPU 构造专家参数 |
| `.../fused_moe/oracle/mxfp4.py` | 混合模式**强制** CPU 后端 + 跳过 AMX prepack | 否则 A100 选 Marlin → `b_q_weight is not on GPU` |
| `vllm/envs.py` | +5 个 `VLLM_TRITON_MLA_SPARSE*` | fork 的 Triton sparse-MLA 路径开关 |

插件侧(新项目 `vllm-xtu-moe`):`mixed_experts.py` 提供
`XiaotuCPUExperts(CPUExpertsMxfp4)`(`_supports_current_device` 放宽到 x86 无 AMX)。

## B. SM80(Ampere/Ada)可移植 Triton 路径

新增(fork 原文件,逐字节一致):

| 文件 | 行数 |
|---|---:|
| `v1/attention/backends/mla/sparse_mla_env.py` | 119 |
| `v1/attention/backends/mla/sparse_mla_kernels.py` | 3517 |
| `models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` | 711 |
| `models/deepseek_v4/nvidia/ops/sm12x_mqa.py` | 756 |
| `models/deepseek_v4/nvidia/ops/fp8_einsum.py` | 320 |

适配合并(主线比 fork 新,需把 Triton 分支适配进主线新版):

| 文件 | 要点 |
|---|---|
| `models/deepseek_v4/nvidia/flashmla.py` | 整体换成 fork 版(1112 行),仅把 `combine_topk_swa_indices` 调用适配主线签名(`out=(idx,lens)`) |
| `models/deepseek_v4/attention.py` | `_fused_qnorm_rope_kv_insert` 等按主线新版结构保留 |
| `v1/attention/backends/mla/{indexer,sparse_swa}.py` | SM80 跳过 DeepGEMM scheduler metadata;补 `prefill_gather_lens_cpu` |
| `models/deepseek_v4/common/ops/{cache_utils,fused_indexer_q,fused_inv_rope_fp8_quant,fused_compress_quant_cache}.py` | `.to(tl.float8e4nv)` → `_f32_to_e4m3_uint8()`(Ampere 无 fp8e4nv);fp8 缓冲以 uint8 视图传入 |
| `.../quantization/utils/fp8_utils.py` | +`@triton.jit _e4m3_uint8_to_f32` / `_f32_to_e4m3_uint8`(fork 原实现) |
| `models/deepseek_v4/compressor.py` | head=512 的 cutedsl 只在 SM89+ 用,Ampere 走 Triton |
| `utils/deep_gemm.py` | `_use_sm12x_mqa_fallback()` 扩到 SM8.x;`fp8_fp4_mqa_logits`/`paged`/`tf32_hc_prenorm_gemm` 加 SM80 分支 |
| `utils/import_utils.py` | `has_cutedsl()` 在 SM8.x 返回 False(cutedsl 只出 SM90+ PTX) |
| `models/deepseek_v4/sparse_mla.py` | `supports_compute_capability` 接受 major==8 |
| `model_executor/kernels/mhc/tilelang.py` | 首层 broadcast prenorm 走 `_torch_hc_prenorm_gemm`(主线 parity gap) |
| `warmup/flashinfer_sparse_mla_warmup.py` | SM80 不做 FlashInfer sparse-MLA warmup |

## C. 数值正确性的关键修复(本会话新增)

| 文件 | 改动 |
|---|---|
| `.../quantization/fp8.py` | **`is_bmm` 特判**:SM80 上 Marlin 会把 `wo_a.weight` 重打包成 int32,Marlin 布局会破坏 o_proj 的 fp8 einsum → 改为就地反量化成 bf16 并 `use_marlin=False`(fork commit `971c27995` 原样移植) |
| `models/deepseek_v4/nvidia/ops/o_proj.py` | 换成 fork 版:走 `deepseek_v4_fp8_einsum`(SM80 走 `DECODE_E4M3` + `B_BF16`),并保留主线 SM90/SM100/SM110 的 recipe 分支 |

## D. 调试脚手架(上游前应删除)

`XIAOTU_DEBUG_L1..L4` / `XIAOTU_TIMING` 埋点分布在
`models/deepseek_v4/nvidia/model.py`、`models/deepseek_v4/attention.py`、
`models/deepseek_v4/nvidia/flashmla.py`。全部 env-gated、默认关闭,但**建议上游前清理**。

## E. 验证(2026-09-09)

- 真实权重 DS-V4-Flash + A100 + 混合模式:输出文本正确(见 `docs/SESSION_2026-09-09.md`);
- 测试服 8071 HTTP 可用,`/v1/completions` 输出正确;
- 单请求/并发吞吐见会话报告。

## F. 核心算法源码位置

- 新项目自包含:`xiaotu_moe/csrc/`(C++ 源码:内核/调度/NUMA/引擎)+ `xiaotu_moe/build/*.so`
  (编译产物)+ `xiaotu_moe/rebuild.sh`(重编译脚本)。
- 引擎是重写版 header-only `MOE_V2`(`moe_v2.hpp` / `moe_v2_packed4.hpp` /
  `moe_v2_fp8.hpp` / `numa_pool.hpp`),`binding.cpp` 通过模板实例化导出
  `MOE_BF16/MOE_FP8/MOE_MXFP4/MOE_NVFP4/MOE_WNA16`。旧 `moe.cpp`/`mlp.cpp`/`linear.cpp`
  是早期 lk_moe 移植,未接入 binding(保留作参考)。

## 修订记录

- 2026-09-09 07:2x:首版(A–E)。
- 2026-09-09 07:5x:补 F(把核心 csrc + rebuild.sh 复制进新项目,使其自包含)。
