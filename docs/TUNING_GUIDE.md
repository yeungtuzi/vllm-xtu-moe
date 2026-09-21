# 调优指南:在固定上下文长度下平衡「投机解码 / long prefill / 常驻 MoE」

> 本文回答一个具体问题:**每个硬件配置 × 每个模型都有自己的最优解吗?**
> **是。** 而且它不是调参玄学,可以写成一个**带硬约束的显存预算分配问题**。
> 本文给出恒等式、各选项的**实测单位代价/收益**、以及决策算法。
>
> 所有数字都标注了出处;凡未经实测的都明确写「需实测」,不做外推。

---

## 1. 恒等式:显存预算

对**每一张卡**:

```
util × VRAM(卡)  ≥  KV 池  +  GPU 预填充 staging  +  投机 draft  +  常驻 MoE  +  激活工作区  +  非专家权重
   (可调)            (硬约束)      (选项 2)          (选项 1)      (选项 3)      (∝ MBT)       (模型决定)
```

**关键区分:**

| | 性质 | 说明 |
|---|---|---|
| **KV 池** | **硬约束,先扣** | 必须 ≥ `要求的上下文长度 × 要求的并发`。这是业务前提,**不为性能让路** |
| 其余五项 | **性能预算,互相竞争** | 三者抢的就是这部分 |

⇒ **所以"最优解"= 在扣掉硬约束后,把剩下的显存按"每 GiB 换多少 tok/s"分给三个选项。**

硬件与模型通过**单位代价**进入这个分配:

* **硬件**:每卡显存(决定总预算)、NUMA/核数(决定 CPU 引擎速率 ⇒ 常驻层的**边际收益**)、PCIe(EP 合并成本)、有无 NVLink(TP 通信);
* **模型**:每 token KV(标尺,决定硬约束吃多少)、专家层大小(决定常驻层单价)、是否有 MTP draft 层、注意力是否 KDA/块大小固定。

---

## 2. 三个选项的实测单位账

### 2.1 KV 池(硬约束的度量)

**唯一权威来源是引擎自己报的 pool 大小**:服务端日志里那一行

```
GPU KV cache size: N tokens
```

把它和当时传入的 `--kv-cache-memory` 相除,就得到该模型的**每 token KV 开销**。三个模型各有一个
已知数据点(MiMo 是 TP=1,GLM/V4.1 是 TP=2;`--kv-cache-memory` 按 **rank** 计):

| 模型 | B/token | KiB/token | tok/GiB | **1M 需要** | **256K 需要** | 单卡 35.5 GiB |
|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | 19,505 | **19.0** | 55,050 | **19.0 GiB** | **4.8 GiB** | ✅ |
| **DeepSeek-V4.1** | 30,639 | **29.9** | 35,045 | **29.9 GiB** | 7.5 GiB | ✅ |
| **MiMo-V2.5** | 253,633 | **247.7** | 4,233 | **247.7 GiB** | **61.9 GiB** | ❌ |

反算依据(每一行都是「已知 cap ↔ 引擎报的 token 数」):

```
MiMo : 9.68 GiB ↔ 40,974 tokens
GLM  : 2.00 GiB ↔ 110,100 tokens
V4.1 : 1.17 GiB ↔ 41,078 tokens
```

`KV_CACHE_BYTES ≥ max(MAXLEN, SEQS × L_max) × (B/token)`,再留一个**小幅**余量(见下)。

> ⚠️ **不要按 `config.json` 推算每 token KV。** 我曾按
> `48层 × 4 KV头 × (192+128) × 2B` 给 MiMo 推出 120 KiB/token,而引擎实测是 **247.7 KiB/token**
> —— 差 2.06 倍。配置里的字段与引擎实际分配的布局并不一一对应(对齐/填充/额外层)。
> **只信 `GPU KV cache size`。**

**为什么"每模型最优解不同",这一张表就是主因**:同样 1M 上下文,
GLM 要 19 GiB、V4.1 要 30 GiB、MiMo 要 **248 GiB** —— 相差 13 倍。
MiMo 的 1M 在这台机器上**不可达**(6.3 倍于单卡);GLM 与 V4.1 的 1M 都装得下。

**余量该给多少**:用**小幅乘性余量**(1.10×)而非 1.25×。
乘性余量对"KV 昂贵"的模型会失控 —— MiMo 的 7.74 GiB × 1.25 = 9.68 GiB,
而 `--kv-cache-memory` 是**硬上限**、叠加在插件 staging/激活之上(后者不受 vLLM 的 util 记账),
实测直接把 39.49 GiB 的卡顶到 39.34 GiB 并 OOM。降到 1.10× 后 MiMo 取 8.52 GiB,正常启动。


### 2.2 选项 1:投机解码(MTP)

| | 值 | 出处 |
|---|---|---|
| 收益 | accept **1.46**(k=1) / 1.63(k=4);**TPOT −3%**(45.7→44.2 ms,**在噪声内**) | `docs/KNOWN_LIMITATIONS.md` §4 |
| 代价 | draft 层常驻 **3.38 GiB/rank**;**KV 池 −27%**(915,487→666,366 tok) | 同上 |
| 副作用 | ITL **脉冲化**(46→64 ms) | 同上 |
| k>1 | **明确更差**:k=4 掉到 11.2 tok/s | 同上 |

⇒ **默认 `SPEC_K=1`;只在"显存充裕且要单流低延迟"时才开;`k>1` 不要开。**

### 2.3 选项 2:long prefill(GPU 流式预填充)

| | 值 | 出处 |
|---|---|---|
| 收益 | 预填充 **1105–1117 t/s**(TP=2 + 常驻 0-5) vs CPU 路径 **261 t/s** | `NOTES` §94.5、§R17 |
| 代价 | staging **~7.59 GiB**(GLM,来自 `preflight` 日志) | `gpu_prefill_fp8.py` 的 `fp8_staging_bytes` |
| 代价 2 | 长 prompt TTFT **+17%**(关旁路流后) | `CHANGELOG.md` 事故复盘(2026-09-21) |

⇒ **长上下文场景必开;纯短上下文 + 显存紧张时可 `GPU_PREFILL=0` 把它整段让给 KV/常驻层。**

### 2.4 选项 3:常驻 MoE

| | 值 | 出处 |
|---|---|---|
| 收益 | **0.70 ms/token/层**(C=1)、**1.70**(C=4);5 层 ⇒ 单流 **+25%**、C=4 聚合 **+62%** | `NOTES` §309 |
| 代价 | **1.59 GiB/rank/层**(TP=2)、**3.19 GiB**(TP=1) | `NOTES` §48、`MODEL_GUIDES.md` |
| 上限 | **11 层**(2×A100-40GB);12/13 层失败 | `NOTES` §320 |
| 机制 | 层的代价是 **`CPU计算 + GPU工作` 串行相加**(实测 `+0.01 ms`),所以少几层 CPU 就真省几 ms | `NOTES:10423` |

⇒ **按"每 GiB 换多少"看,常驻层是目前**最划算**的一项**(1.59 GiB 换约 0.7 ms/token),
**但它的边际收益随层数递减不明显、而 KV 被挤掉的损失递增** ⇒ 逐层加,到预算为止。

> `scripts/bench_resident_sweep.sh` 会在**固定上下文长度**下扫 k=0/3/5/8/11,
> 并**回读日志确认实际常驻层数**,给出本机当下的曲线。

---

## 3. 决策算法

```
1. 扣硬约束:KV_pool = 要求的上下文长度 × 要求的并发 ÷ 标尺 × 1.25   (不可动)
2. 剩下的显存 = 性能预算 B
3. 按"每 GiB 收益"从高到低填:
   a. long prefill staging    ~7.59 GiB   —— 若负载含长 prompt,这一项必留,否则预填充慢 4×
   b. 常驻 MoE                1.59 GiB/层 —— 用剩余 B 逐层加,直到 B 用完或到 11 层上限
   c. 投机 draft              3.38 GiB    —— 收益最小(−3% TPOT,在噪声内),且 KV −27%;最后才考虑
4. 每一步都在**服务端日志里回读开关实际生效**(本项目的 env-bridge 会覆盖,见 §4)
5. 用目标负载实测一轮(不要只算):`scripts/bench_random_12cells.sh` 给出 12 格的
   prefill/decode,`scripts/bench_resident_sweep.sh` 给出常驻层曲线
```

> **顺序不是死的**:如果负载是"纯短 prompt + 高并发",把 3a 的 7.59 GiB 让给 3b/3c 往往更划算 ——
> 这正是本文想说"每配置有自己的最优解"的地方。

---

## 4. ⚠️ 两个必须遵守的操作纪律

1. **任何开关都要回读日志确认实际生效。** `scripts/serve_*.sh` 会把 `XIAOTU_*` 透传进
   env-bridge 文件,而插件在 `XIAOTU_ENV_FILE` 显式指定时**以文件为准覆盖** `os.environ`
   ⇒ 有些变量会被静默丢弃/覆盖。2026-09-21 单次会话内因此栽了**两次**
   (`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`、`GPU_PREFILL`)。
   例:常驻层要 `grep -c "GPU-resident" <server.log>` 核对层数。

2. **`util` 与 KV cap 必须同步设定**(`--kv-cache-memory` 是**硬上限**,
   不是"随 util 自动算")。只看 `--gpu-memory-utilization` 会让 KV 池与预期不符。

---

## 5. 一个算好的例子(2×A100-40GB,GLM-5.3-Flash,要求 32K × 2 并发)

| 项 | 值 | 依据 |
|---|---|---|
| 总预算(单卡) | 0.80 × 39.49 = **31.6 GiB** | `util=0.80` |
| ① KV 池(**硬约束**) | 32768×2 tok × 19,505 B/tok = **1.28 GiB** → 取 **1.5 GiB** | §2.1(引擎反算) |
| ② long prefill staging | **7.59 GiB** | §2.3 |
| ③ 常驻 MoE | 每层 **1.59 GiB** | §2.4 |
| ④ MTP draft | **3.38 GiB**(可关) | §2.2 |
| ⑤ 非专家权重 + 激活 + 图 | 实测约 **2.9–3.0 GiB**(启动 profile 的 peak activation) | `KNOWN_LIMITATIONS` §8.2 |

⇒ 剩余给常驻层 ≈ `31.6 − 1.5 − 7.59 − 2.9` ≈ **19.6 GiB** ⇒ **约 12 层**(1.59 GiB/层),
**与 `NOTES` §320 实测的 11 层上限吻合**。

**若关掉 MTP(④=0)** 并把这 3.38 GiB 给常驻层 ⇒ 多约 2 层 ⇒ 约 **+1.4 ms/token(单流)**。
**若把 long prefill 也关掉(②=0)** ⇒ 再多约 4.7 层,但**预填充从 1105 t/s 掉到 261 t/s** ——
**只在纯短 prompt 负载下才划算**。

> 上表里 ① ② ④ ⑤ 都有实测出处;③ 的"11–12 层"是**算出来的**,与 §320 的实测 11 层一致 ——
> 但**仍需用 `bench_resident_sweep.sh` 在本机复核**,因为 util、激活峰、标尺都会随版本变。

---

## 6. 相关文档

* `docs/RUNBOOK.md` —— 具体服务脚本的参数与默认值;
* `docs/KNOWN_LIMITATIONS.md` §4(MTP)、§8.2(MTP 与激活峰)、§8.3(GPU 预填充非法访存);
* `dev-docs/report/tuning/NOTES.md` §309/§320(常驻层原始实测)、`:10423`(串行机制);
* `dev-docs/report/tuning/TRIED_AND_REVERTED.md` R15/R16/R17(不要再试的方向);
* `scripts/bench_random_12cells.sh`(12 格)、`scripts/bench_resident_sweep.sh`(常驻层扫描);
* `CHANGELOG.md` —— 2026-09-21 的旁路流非法访存修复(**代价 +17% 长 prompt TTFT**)。

---

## 7. 🔑 规则:**遇到稀疏注意力模型,先找稠密/absorbed 路径,再谈优化**

> 2026-09-21 由 GLM-5.3-Flash 的 prefill 事故得出。**这是一条通用排查规则,不只对 GLM。**

### 7.1 为什么

稀疏注意力(DSA/topk 类)的**稀疏选择是按 query 各不相同的索引**。
这导致一个结构性问题:**同一个 program 内的多个 query 无法共享 KV tile ⇒ 无法用 `tl.dot`/张量核**。
上游对此的实现通常是一个 **per-token、逐元素 FMA、按索引 gather 的"便携 Triton"回退核** ——
它的存在意义是「**让这一档硬件能跑起来**」,而不是「跑得快」。

**实测代价(GLM-5.3-Flash,L=16384,C=1,A100):**

```
_sparse_mla_fwd_with_sink_kernel   22.325 s   58.80% 的 CUDA 时间
⇒ 该核比 FLOPs 下界慢约 **12,800 倍**     ← 不是算力受限
标度律:O(L^1.42)(L 4096→16384 增 4×,时间增 7.13×)
   对照:稀疏预期 5.0× / 稠密 L² 预期 16×
⇒ **稀疏确实在限制工作量,问题纯粹是"单位工作量效率"(无张量核)** —— 归一化后 0.71 ns/key-attend
```

**而修法往往不需要写核 —— 上游很可能已经为这个模型准备好了稠密路线:**

```
稠密 vs 稀疏的盈亏平衡:稠密多算 4.27×(topk=2048 vs L=16384)
  ⇒ 只要稠密每次效率优于 0.71/4.27 = **0.18 ns**,稠密就赢
  而 tensor-core flash attention 通常 0.01–0.05 ns/次 ⇒ **好 4–18 倍**
```

### 7.2 排查清单(按顺序,前两步是"零成本")

| # | 查什么 | 命令/位置 | 若成立 |
|---|---|---|---|
| **1** | 该模型的 attention 是否**调用** `get_mla_prefill_backend`? | `grep -rn get_mla_prefill_backend vllm/models/` | ✅ 只需设配置项 `mla_prefill_backend`(`vllm/config/attention.py:76`),**零代码** |
| **2** | 稠密 prefill 后端的**维度白名单**是否覆盖该模型? | `vllm/v1/attention/backends/mla/prefill/*.py` 的 `supports_mla_dimensions()` | ✅ 上游已适配,只差路由 |
| **3** | 本档算力上选择器选谁? | `prefill/selector.py:_get_mla_prefill_backend_priorities` | SM90 及更老 ⇒ `[FLASH_ATTN]`;Blackwell ⇒ TRTLLM_RAGGED/FLASHINFER/... |
| **4** | 若不满足 ⇒ **改路由(打补丁),而不是写核** | 模型侧硬绑稀疏后端的那几行 | 例:GLM-5.3 在 `glm5next/common/attention.py` |

### 7.3 GLM-5.3-Flash 的实测结论(作为范例)

| 检查 | 结果 |
|---|---|
| 模型是否调用选择器? | ❌ **不调用**。`get_mla_prefill_backend` 全树只被 `kimi_k3/nvidia/mla.py:103` 调用 |
| 白名单是否覆盖 GLM? | ✅ **覆盖,而且上游专门为它加了注释**:`flash_attn.py:313-331` 里有 `MLADimensions(256, 0, 256)`,注释写「GLM5Next NoPE layout ... run the same kernels as the (192, 64, 256) DeepSeek-V3.2 layout」 |
| SM80 选择器选谁? | ✅ `selector.py`:`else: # Hopper(SM90) and older → [FLASH_ATTN]` |
| 结论 | **稠密路线齐备,唯一障碍是 DSA(v32) 分支硬绑 `FlashMLASparseSM8XBackend`** |

**目标形态:**
```
prefill → FLASH_ATTN(分块、张量核、GLM NoPE 布局已白名单)
decode  → FlashMLASparseSM8XBackend(per-token gather —— **在那里是合理形态**)
```

**应落成 `patches/upstream/` 下的新补丁**(仓库已有 `pr2-fp8-sm80-o-proj.patch` 先例),
**不要直接改 rebase 树**。

### 7.4 换之前必须确认的四件事(否则是"提速了但变味了")

1. **prefill 会丢掉稀疏选择** ⇒ 前 `topk` 个 token 完全等价;更长的位置**看得更多**。
   **这是行为改变,必须做质量回归,不能只看速度。**
2. **indexer 在 prefill 可能仍需运行**(kpool 边界池的播种、`KpoolTailSpec` 跨 PD 传输)
   ⇒ 「注意力走稠密」≠「关掉 indexer」,两者分开处理。
3. **KV dtype**:SM8x 稀疏路径要求 bf16 KV;换稠密后该约束可能松动,需确认。
4. **decode 必须仍走稀疏** —— 那里 per-token 是正确形态,别一起改掉。

### 7.5 一句话

**稀疏注意力模型的 prefill 慢,先怀疑"走了回退核",而不是"硬件不行"。
上游往往已有稠密/absorbed 路线;先查模型是否调用选择器,再查维度白名单,
最后才考虑改路由或写核。**
