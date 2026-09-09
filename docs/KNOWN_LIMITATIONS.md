# 已知限制

本文列出当前版本明确不支持或需要注意的情形。**数值正确性优先**:
凡是语义不确定的配置,插件会直接报错,而不是给出可能错误的结果。

---

## 1. 模型 / 硬件层面

### GLM-5.3-Flash 需要 SM90+

`zai-org/GLM-5.3-Flash` 的 MLA 维度为 `qk_nope_head_dim=256 / qk_rope_head_dim=0 /
v_head_dim=256`,在 vLLM 主线中**没有任何 attention 后端支持**:

```
sparse 打开(默认):
  ValueError: No valid attention backend found ... FLASH_ATTN_MLA / FLASHMLA /
  FLASHINFER_MLA / TRITON_MLA / FLASHINFER_MLA_SPARSE_SM90 / FLASH_ATTN_MLA_SPARSE /
  FLASHMLA_SPARSE 全部因 compute capability 或 sparse 被拒绝

sparse 关闭(--hf-overrides '{"index_topk": null}'):
  ValueError: No valid MLA prefill backend found with
  mla_dimensions=(qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256).
  Reasons: {FLASH_ATTN: [Model does not have supported MLA dimensions]}
  (FLASH_ATTN 仅支持 (128,64,128)/(192,64,256)/(64,64,128);FLASHINFER 仅 Blackwell 且 (128,64,128))
```

这与 MoE 后端无关:本插件对该模型**专家层**的数值已用真实权重验证
(`scripts/test_glm53_fp8_layer.py`,RMS 相对误差 9.3e-5)。
要在 A100 上端到端运行该模型,需要上游补充 256 维 MLA 的 attention 支持,
或使用 SM90+ 硬件。

## 2. 后端能力

| 限制 | 说明 |
|---|---|
| **不支持专家并行(expert_map / EP)** | 出现 `expert_map` 时插件直接报错;TP>1 走权重分片 |
| **不支持 `apply_router_weight_on_input`** | 少数模型使用该选项 |
| **不支持交错 gate/up 布局** | `SWIGLUOAI`(gpt-oss 系)把 gate/up 交错存在 `w13` 中;引擎按 packed 布局取数,因此该情形由 `_supports_activation` 拒绝 |
| **INT4(WNA16)组大小固定 128** | 组大小/零点的完整适配进行中 |
| **INT8 W8A8 未实现** | 引擎暂无该格式内核 |
| **monolithic `apply()` 拿不到 `input_ids`** | 依赖 `input_ids` 的路由(哈希路由)无法走该路径,插件会报错而非静默算错 |

## 3. 验证状态(诚实记录)

| 项 | 状态 |
|---|---|
| 引擎 vs torch 参考(BF16 路径) | ✅ 相对误差 ~1e-7 |
| 引擎 vs numpy 参考(FP8 block-128,真实 GLM-5.3 专家权重) | ✅ RMS 相对误差 9.3e-5 |
| 引擎 vs torch 参考(FP8,真实 Qwen3-30B 全部 48 层) | ✅ 相对误差 ~1e-7(`XIAOTU_VERIFY_LAYER=1`) |
| 引擎 vs torch 参考(MXFP4,真实 DeepSeek-V4 层权重) | ✅ RMS 相对误差 5.6e-3 |
| CPU 专家 vs GPU 专家端到端(bf16 微型模型) | ✅ greedy token 完全一致 |
| **CPU 专家 vs GPU 专家端到端(真实 fp8 模型)** | 🟡 **排查中**:同一 prompt 的 prompt-logprob 存在差异,层内检查(上表)通过,正在定位是权重加载还是采样路径的差异 |

## 3.1 跨请求一致性(已修复)

在真实 fp8 模型上曾复现:同一进程内连续两个请求,第二个请求会"继承"第一个请求的
上下文(例如重复前一个问题的答案)。定位结论:

- 引擎通过 `cudaMemcpyAsync` + CUDA host 回调读写设备张量,而 vLLM 的异步输出拷贝
  运行在独立的 non-blocking stream 上;
- 在引擎调用前后与设备同步即可消除该现象(实测 3/3 通过);
- 因此默认开启同步,可用 `XIAOTU_SYNC_DECODE=0` 关闭(仅建议用于性能实验)。

根因的进一步定位(具体是哪一个 stream 的哪次复用)仍在进行中。

## 4. 性能现状

| 项 | 现状 |
|---|---|
| **FP8 CPU 内核** | 当前是"先正确"的实现,吞吐 0.35–0.43 TFLOP/s,低于 MXFP4 路径(1.7–2.0 TFLOP/s),待优化 |
| **CPU prefill** | 长 prompt 在纯 CPU 上很慢(2K token 约 113 s),需开启逐层 GPU 流式(`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`) |
| **AMX** | 未接线;Intel AMX 机器目前会使用主线自带 CPU 内核(需自行验证) |
| **多 ISA 打包** | `scripts/build_engine_variants.sh` 可编出 5 个变体,但发布 wheel 目前默认只带单一变体 |

## 5. 运行注意事项

- 首次加载超大模型(数百 GB 专家权重)需要数分钟,请放大
  `VLLM_ENGINE_READY_TIMEOUT_S`;
- 不设 `XIAOTU_MOE_SINGLECOPY=1` 时,引擎可能按 NUMA socket 复制权重,内存占用约 2×;
- `--enforce-eager` 建议先开启,便于定位问题;稳定后可尝试关闭;
- CPU 引擎自带线程池,`OMP_NUM_THREADS` / `VLLM_CPU_OMP_THREADS_BIND`
  只影响主线自带的 CPU 内核。

## 6. 数值验证工具

出现"输出不连贯"时,依次使用:

```bash
XIAOTU_VERIFY_LAYER=1 <你的启动命令>     # 每层打印 rel_rms(引擎 vs torch 参考)
XIAOTU_MOE_PROFILE=1 <你的启动命令>      # 打印引擎相位耗时
python scripts/probe_oracle.py           # 确认后端选择
python scripts/tiny_moe_equiv.py         # CPU/GPU 专家端到端等价性
```
