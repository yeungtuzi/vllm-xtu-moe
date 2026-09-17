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
| **专家并行(expert_map / EP)** | ✅ 已支持(TP=2 + EP);EP 存储分片改动了专家分配/加载/映射,当前代码上的模型复验状态见 `docs/HANDOFF_v0.2.md` §5.1 |
| **不支持 `apply_router_weight_on_input`** | 少数模型使用该选项 |
| **不支持交错 gate/up 布局** | `SWIGLUOAI`(gpt-oss 系)把 gate/up 交错存在 `w13` 中;引擎按 packed 布局取数,因此该情形由 `_supports_activation` 拒绝 |
| **INT4/WNA16 只支持对称量化** | GPTQ / compressed-tensors 的 `w13 [E, K/8, 2I] int32` + 组缩放会在引擎构造时重排为引擎布局(`[E, 2I, K/2]` u8 + `[E, N, K/group]` 缩放),**zero point 必须为 8**(对称)—— GPTQ 存 `zp-1`,即检查点的 `qzeros` 全为 7。出现非 8 的零点(非对称 AWQ/GPTQ)或 AWQ 的 N-packed 布局时插件**显式报错**,不会静默算错 |
| **INT8 W8A8 未实现** | 引擎暂无该格式内核 |
| **monolithic `apply()` 拿不到 `input_ids`** | 依赖 `input_ids` 的路由(哈希路由)无法走该路径,插件会报错而非静默算错 |

## 3. 验证状态(诚实记录)

| 项 | 状态 |
|---|---|
| 引擎 vs torch 参考(BF16 路径) | ✅ 相对误差 ~1e-7 |
| 引擎 vs numpy 参考(FP8 block-128,真实 GLM-5.3 专家权重) | ✅ RMS 相对误差 9.3e-5 |
| 引擎 vs torch 参考(MXFP4,真实 DeepSeek-V4 层权重) | ✅ RMS 相对误差 5.6e-3 |
| CPU 专家 vs GPU 专家端到端(bf16 微型模型) | ✅ greedy token 完全一致 |
| **CPU 专家 vs GPU 专家端到端(真实 fp8 模型)** | ✅ 同一 prompt 下 **top-1 预测 20/20 一致**,实际 token 的 logprob 平均偏差 0.065(最大 0.24)——差异来自 GPU 侧对激活做动态 fp8 量化、CPU 侧用 bf16 |

## 3.1 跨请求一致性(已从根因修复)

在真实 fp8 模型上曾复现:同一进程内连续两个请求,第二个请求会"继承"第一个请求的
上下文(例如重复前一个问题的答案)。

**根因**:引擎通过 `cudaMemcpyAsync` + CUDA host 回调读写设备张量,而参数
(`qlen` / `ids` / `hidden` / 输出指针)保存在**每个引擎实例共享的可变状态**里。
当 vLLM 的异步调度让下一个 step 的 host 代码跑在本 step 的回调之前时,尚未执行的
回调会读到新 step 的参数,于是把新 step 的结果写进旧 step 的输出。

**修复**:每次调用分配一个**不可变的参数块**(由回调自己释放),回调不再读取
per-engine 的可变字段;pinned 缓冲仍由引擎复用,其读写受 stream 顺序保护。
复现用例 3/3 通过,且**无需**全设备同步。

调试时仍可用 `XIAOTU_SYNC_DECODE=1` 强制在引擎调用前后同步(会降低吞吐)。

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
- 分片布局在极端情况下(维度不整除 / 分配失败)会退回每 socket 一份副本,内存占用约 2×(安全网,正常配置不会触发);
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

---

## 7. 更新(2026-09-18)

### 7.1 已修复:**线程池丢票竞态(R113)** —— 见 `report/tuning/NOTES.md` §569
* 复现:专用压力器(`report/tuning/probes/stress_pool.{py,sh}`)**4/4 进程在 20~105 s 内 SIGABRT**,
  证据 `WATCHDOG(sharded) total=64 rem=1 exec=63 abandoned=48 node*: jobs=8 pulled=14`,
  **`snap_retry=0 snap_mismatch=0`**(seqlock 复读检不出"自洽但陈旧"的快照)。
* 根因:调用方"奇数发布 store(release)"在 x86 上只进 store buffer,紧随其后的票快照 load 可以先执行
  ⇒ worker 领到**本代**的票却按**旧**快照判越界 ⇒ `abandoned` 丢弃且不递减 ⇒ `remaining_` 永挂。
* 修复:两处发布的奇数 store 改 `memory_order_seq_cst`(x86 = `xchg` 全屏障)⇒ 快照之后领票者必看到新代。
* 验证:**4/4 SIGABRT → 0/4**;长跑 4×600 s(≈960 万次池调用)零 WATCHDOG;数值门/性能门**不变**。
* ⚠️ flat 路径同型分支被同一修复覆盖,但**未单独复现**(8 node 下压力器走 sharded)⇒ 记为待补验证。

### 7.2 新增已知**非缺陷**特性:长 prompt 的 prefill logits 不逐位可复现(§572)
* 无任何改动、同一请求重发:短 prompt(n≤300)**逐位相同**;`long_1024` Δlogprob 0.12~0.25;
  `long_4096` 0.13~0.38;**首个生成 token/贪心文本始终相同**。
* **两次独立否证**:①关 prefix cache 无效;②`GP_MIN=0`(MoE 全在逐位确定的 CPU 引擎)仍无效。
* 归因:vLLM 长序列 CSA2 稀疏注意力的原子累加(块数越多归约顺序越多)。
  ⇒ 长上下文验收请用"首 token/贪心文本一致",**不要**用逐位 logprob 门槛。

### 7.3 实验特性:**③ CED 预填充捷径**(默认关,未验收)
* 上游 vLLM 只有通用 `kv_sharing_fast_prefill`,对 V4.1 零接线;omlx PR#3607 已做同类功能
  (声称 prefill 快 74-79%,**长上下文精度仍在评估**)。
* 我们已实现,全部 env 门控(`XIAOTU_CED_FASTPREFILL=1`):eligible=19 层(21..39)、
  attention metadata 窗口改写、**层循环级隐藏状态切片**、返回前零填充回全长。
* **⚠️ 未取得验收数据(生成等价性 + 预填充提速)之前不要开启。**
