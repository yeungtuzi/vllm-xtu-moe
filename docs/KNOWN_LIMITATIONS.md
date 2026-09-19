# 已知限制

本文列出当前版本明确不支持或需要注意的情形。**数值正确性优先**:
凡是语义不确定的配置,插件会直接报错,而不是给出可能错误的结果。

---

## 1. 模型 / 硬件层面

### GLM-5.3-Flash:SM80 上必须用 `--kv-cache-dtype bfloat16`

`zai-org/GLM-5.3-Flash` 的 MLA 维度是 `qk_nope_head_dim=256 / qk_rope_head_dim=0(NoPE)/
v_head_dim=256`。**上游主线**的通用 sparse-MLA 后端全部要求 SM90+(或 Blackwell),
所以 stock vLLM 在 A100 上会报 `No valid attention backend found`
(`FLASH_ATTN_MLA_SPARSE` / `FLASHINFER_MLA_SPARSE_SM90` / `FLASHMLA_SPARSE` 均因
compute capability 被拒),这也是早期文档写"需要 SM90+"的原因。

本插件为此提供 **SM8x(Ampere/Ada)稀疏 MLA 后端**:它复用主线的 bf16 gather /
logical→physical topk 机制,只把最后的 attention 核换成可移植 Triton
`sparse_mla_fwd_with_sink`。该路线**只支持 bf16 KV cache**,因此启动时必须:

```
--kv-cache-dtype bfloat16     # fp8/fp4 KV 会 fail-closed(回到上游 SM90+ 候选池并报错)
```

实测(**2×A100-40GB / SM80 / TP=2 / 真实 FP8 检查点**):端到端跑通,贪心输出连贯
且三次逐字节可复现;性能与启动命令见 `docs/RUNBOOK.md` 与 `docs/MODEL_GUIDES.md`。

> 该后端由模型侧显式绑定(`glm5next` 的 DSA 层),**不改变其它模型的选路**;
> 本插件的 CPU 专家路径对该模型早有验证(`scripts/test_glm53_fp8_layer.py`)。

**上下文上限(硬件约束,2026-09-19 实测)**:2×A100-40GB / TP=2 / bf16 KV 下,
`--max-model-len` 的启动阶梯是:256K ✅(KV 池 **988,081 token**,3.77× 并发)、512K ✅(843,055)、
**704K ✅(763,177,但只剩 1.06× 并发)**、768K ❌(vLLM 自报上限 733,312)、
**1M ❌**(需 11.57 GiB 注意力 KV,只有 6.87 GiB)。
⇒ GLM 的模型上限是 1M,本机 bf16 KV 下的**可行上限约 733K**;交付配置取 **256K × 2 路**
(两路 256K 只占池子 53%)。

> ⚠️ **别把两个模型的 KV 单价搞混**(公开文档此前写错过):**GLM-5.3-Flash 是
> ~11.9-12.3 KB/token**(只有 11 层 NoPE 稀疏 MLA 带 per-token KV,每层 512 维 latent × 2 B);
> **DeepSeek-V4.1-Flash 才是 ~40 KB/token(23 GiB ↔ 615,660 token)**。

要上 1M 只有一条路:**fp8 KV**(SM8x 稀疏 MLA 的 fp8 变体 + NoPE 感知的 528 B blob;
现成的 `fp8_ds_mla` 656 B blob 只到 ~948K)。

> **决定(2026-09-19,项目方)**:GLM-5.3-Flash 在本硬件上**以 256K × 2 路为交付目标**,
> **不做 fp8 KV、不追 1M**;512K/704K 的「能起」只作为能力记录,不作为交付目标,也不再补
> 端到端长文验证。上表保留是为了说明硬件边界在哪,不代表待办。

**长 prompt 的 TTFT**:CPU 引擎预填充吞吐饱和在 ~33k 专家-token/s(`MODEL_GUIDES.md` §2.5)。
**FP8 GPU 预填充已实现**(`vllm_xiaotu_moe/gpu_prefill_fp8.py`,按引擎类别自动选后端),
4096-in / 64-out 的 TTFT 从 **29.3 s 降到 22.8 s(1.29×)**。装配是 3.62 GB/rank 的纯 H2D
固定成本(本机 H2D 天花板 26.86 GB/s,实测 1-D 与 pitched 2-D 同速),所以优化点是
**把它与 attention 重叠**(默认开启的 side stream),而不是"搬得更快"。

## 2. 后端能力

| 限制 | 说明 |
|---|---|
| **专家并行(expert_map / EP)** | ✅ 已支持(TP=2 + EP);EP 存储分片改动了专家分配/加载/映射,当前代码上的模型复验状态见 `内部交接 HANDOFF_v0.2pre.md` §5.1 |
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
| **GPU 预填充的显存是"借"来的** | staging 缓冲**进程级持久**(GLM-5.3/TP=2 实测 **7.59 GiB/rank**),吃的是 vLLM 用来定 KV 的那份激活预留 ⇒ 必须让出 KV(util ≤ 0.85)。预检不过时会**优雅退回 CPU 预填充**(慢但正确);若在 util 0.90 这种紧配置下强行开,长 prefill 会 OOM 崩服务(§601 事故,见 RUNBOOK §3.2) |
| **`expandable_segments` 的物理占用只增不减** | 开着它时进程会随碎片把已保留段一直留着,`nvidia-smi` 在空闲时也可能显示 ~98% 占用(实测 40,265/40,960 MiB)。这些段**可被本进程复用**,不是泄漏;但它意味着"nvidia-smi 剩余"不等于"还能新分配多少" |
| **AMX** | 未接线;Intel AMX 机器目前会使用主线自带 CPU 内核(需自行验证) |
| **投机解码(MTP)未开** | GLM-5.3-Flash 自带 1 层 MTP(`layers.45`,含 288 专家 MoE),vLLM 也注册了 `Glm5NextMTPModel` ⇒ 结构上可开;但插件的 draft 判定是为 DeepSeek DSpark 写的("同一 prefix 出现两次"),GLM 的 `layers.45` 只出现一次会被放 CPU,而实测 **CPU draft 时投机净负** ⇒ 当前关闭(分析见 `MODEL_GUIDES.md` §2.6) |
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

### 7.1 已修复:**线程池丢票竞态(R113)** —— 见 `内部调优记录 NOTES.md` §569
* 复现:专用压力器(`内部探针 probes/stress_pool.{py,sh}`)**4/4 进程在 20~105 s 内 SIGABRT**,
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
