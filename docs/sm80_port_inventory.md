# SM80 支持移植清单(fork Lvllmds4-x → dev 主线 6c73b08)

生成: 2026-09-08 夜。依据: 人工 diff + 文件清点。

## A. fork 有、主线无的新文件(需新增)
- `v1/attention/backends/mla/sparse_mla_env.py` (119 行) — 主线缺失
- `v1/attention/backends/mla/sparse_mla_kernels.py` (3517 行) — 主线缺失
- `models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` (711 行) — 主线缺失
- `models/deepseek_v4/nvidia/ops/sm12x_mqa.py` (756 行) — 主线缺失
- `models/deepseek_v4/nvidia/ops/fp8_einsum.py` (320 行) — 主线缺失

## B. 两边都有但已分化的文件(需适配合并)
- `models/deepseek_v4/nvidia/flashmla.py` fork 1112 行 vs 主线 398 行
- `v1/attention/backends/mla/flashmla_sparse.py` fork 880 行 vs 主线 1032 行
- `v1/attention/backends/mla/indexer.py` fork 806 行 vs 主线 1416 行
- `v1/attention/backends/mla/sparse_swa.py` fork 733 行 vs 主线 1267 行
- `models/deepseek_v4/nvidia/model.py` fork 1396 行 vs 主线 1863 行

## C. 需要的 env 变量(主线缺失)
- VLLM_TRITON_MLA_SPARSE, VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE,
  VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE, VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE,
  VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE

## D. 核心机制
- sparse_mla_env.py: is_ampere_or_ada() (SM80/86/89) 启用可移植 Triton sparse-MLA 路径;
  SM90/100 仍走原生 FlashMLA/DeepGEMM → 加法式,不影响现有支持。
- 注释明示: SM8.x 无 FP4 tensor core → MoE fp4 回退 Marlin WNA16(或我们的 CPU 混合模式)。

## E. 风险
- 主线 flashmla.py(398 行)与 fork(1112 行)结构已分化,需把 Triton 分支适配进主线新版;
- 非 drop-in,预计多轮工程。

## F. 决定数值正确性的隐藏点:fp8 linear 的 Marlin 重打包(2026-09-09 补)
SM80 没有 fp8 tensor core → 主线/fork 的 fp8 block 量化 dense linear 都落到
`MarlinFP8ScaledMMLinearKernel`(W8A16),`prepare_fp8_layer_for_marlin` 会把
`layer.weight` **重打包成 Marlin 私有 int32 布局**。

这对普通 linear 无害(`apply_weights` 配套),但 DS-V4 的 **o_proj 绕过 linear.apply 直接读
`wo_a.weight`** 做 fp8 einsum(`deep_gemm_fp8_o_proj`)→ 读到打包后的乱数据 → 反量化出巨值
→ 注意力输出 inf → 采样全 0。fork 的官方修法(commit `971c27995`)是
`Fp8LinearMethod.process_weights_after_loading` 里特判 `getattr(layer,"is_bmm",False)`:
不做 Marlin 重打包,而是把 block-fp8 权重就地反量化成 **bf16 [N,K]** 并 `use_marlin=False`,
einsum 侧走 `deepseek_v4_fp8_einsum` 的 `B_BF16` 分支。**已照此移植。**

推论(移植时的检查清单):凡"绕过 linear.apply、直接消费量化权重张量"的 DS-V4/DeepGEMM
调用点,在 SM80 上都要确认该权重没被 Marlin 打包。目前只有 `wo_a` 属于此类。

## G. 已验证结果(2026-09-09 05:5x)
真实权重 + 混合模式(A100/GPU2):`"The capital of France is"` → `" Paris. The capital of
Spain is Madrid"`(token_ids `[11111,16,455,6102,294,16603,344,29168]`),数值正确。

## 修订记录
- 2026-09-09 05:5x:新增 F(wo_a/Marlin 重打包根因与修法)与 G(真实权重数值正确性验证)。
- 2026-09-08 夜:初稿(A-E,移植清单与风险)。

