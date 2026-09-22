# v0.2.4 —— 支持 MiMo-V2.6-Flash-RL · 性能优化

**日期**:2026-09-22 · **基线**:上游 `133b71e0b`(补丁栈不变)

---

## 一、新增模型:MiMo-V2.6-Flash-RL

**检查点**:`MiMo-V2.6-Flash-RL`(166 GB;index `total_size` = **161 GiB**)

| 项 | 值 |
|---|---|
| 专家 | **MXFP4** — gate/up `U8 [2048,2048]`、down `U8 [4096,1024]` + **e8m0 block-32** `weight_scale` |
| 注意力 | FP8 E4M3 block-128(GA:4 KV 头 / head_dim 192 / v_head_dim 128;SWA:8 KV 头) |
| 架构 | 48 层(1 dense + 47 MoE)、**256 专家 top-8**、`moe_intermediate_size=2048` |
| 层型 | **9 GA + 39 SWA(窗口 128)**;`max_position_embeddings = 1048576` |
| 投机 | MTP 3 层权重 + `dflash/` 草稿 |

**关键判断**:专家是 **MXFP4 = V4.1 那条已在生产跑的 `MOE_MXFP4` 路径** ⇒ **无需新引擎**;
体积 161 GiB(V2.5 是 293)⇒ 单流 decode 带宽天花板约 **77 tok/s**。

**交付能力**

* ✅ **TP=2 / 1M 上下文 / 多模态**(三条口径均有实测)
* ✅ **投机解码用 MTP k=1**(实测 k=3 每步仅比 k=1 多 **4%**,而每步多算 2 个草稿 token ⇒ 不划算;
  多深度补丁已回退,**不再追**)
* ✅ 服务脚本 `scripts/serve_mimo26.sh`(把验证过的配方固化,含 KV 账与两个陷阱提示)

**实测(128K 上下文 / MBT=8192 / `seqs=4` / GPU 预填充开 / 含形状预热)**

| prompt | 并发 | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|
| 短 128 | 1 | **232.0** | **30.5** |
| 短 128 | 2 | **367.5** | **40.7** |
| 长 16,384 | 1 | **811.3** | **27.3** |
| 长 16,384 | 2 | **1455.3** | **39.6** |

**正确性证据**

* **层内数值门禁 94/94 层通过**:`rel_rms` **中位 9.25e-08**、最大 1.67e-04、无离群层
  (这台门禁此前在 MXFP4 上是**坏的**,本次修好并验证)
* **长上下文针测试命中**:6,277 token 与 **312,037 token** 两级都取回了埋在中段 60% 处的答案
* **多模态真图测试通过**:渲染 `ZQ7K42` 送图,两种问法都读对
* ✅ 加载:64 个 EP 分片 + `gate/up` 融合 + e8m0 scale **全部读对**(`w13 shape=(256,2048,2048)`)

---

## 二、性能优化

### 1. GPU 预填充:MiMo 此前**从未启用**(+2.3×)

`mixed_experts.py` 的判定是 `_gp_on = _gp_min > 0` ⇒ **门槛为 0(或不设)= 关闭** GPU 预填充,
而 `serve_mimo26.sh` 从未设置它 ⇒ 长 prefill 全走 CPU,**只有 ~310 tok/s**。
显式设 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=4096` 后:

| MiMo 长 16,384 | 关 | **开** |
|---|---|---|
| 聚合 prefill | 313.4 | **811.3**(2.6×) |
| TTFT | 52.3 s | **20.2 s** |

⇒ 该**反直觉语义已写入 `docs/MODEL_GUIDES.md` §0.1**,并规定:接入新模型必须显式给正数。

### 2. `--max-num-seqs` 默认统一为 **4**

此前 MiMo / DeepSeek-V4.1 的默认值是 **1** ⇒ **C≥2 的请求退化成串行**,并发基准被彻底污染
(实测:同一格里 MiMo 短 C=2 的聚合 prefill 236 → **54**,而 `seqs=2` 的模型是 115.8 → **129.6 上升**)。
改为 4 后 **短 C=2 由降转升(236 → 367,+58%)**。生产脚本仍可传更大值(64/128)。

### 3. **decode 口径修正为 `median ITL`**

`decode = C × 1000 / median(TPOT)` 中的 `median TPOT` 是**每请求平均**,
会把**同一 step 里混进来的 prefill 工作**算进解码时间 ⇒ 长上下文 C=2 被严重低估:

| 用例 | median TPOT 口径 | **median ITL 口径(修正后)** | 真实争用 |
|---|---|---|---|
| MiMo 长 C=2 | 9.8 | **39.6** | **1.36×** |
| GLM 长 C=2 | 2.9 | **29.6** | 1.36× |
| DeepSeek-V4.1 长 C=2 | 15.5 | **34.2** | 1.13–1.77× |

⇒ **真实解码争用只有 1.13–1.77×**,与"单路 1.5×"的直觉一致。README 表格口径已同步修正。

### 4. 新旋钮

`scripts/serve_mimo26.sh`:`SEQS`(默认 4)、`MM`(多模态)、`SPEC_K`(MTP)、
`RESIDENT_LAYERS`(专家层常驻)、`GPU_PREFILL_MIN`;
`scripts/serve_v41.sh` / `serve_glm53_mainline.sh`:`MM` / `RESIDENT_LAYERS` 等。

---

## 三、已知问题(诚实记录)

1. **`vram_policy.py` 会产出"起不来"的配置**:128K 时它下发 **0.5 GiB** KV,而引擎启动至少需要 **0.66 GiB**
   ⇒ `ValueError: To serve at least one request...`(**当时空闲显存 38.75 GiB**)。根因:KV 模型常量按 1M 标定,
   短 maxlen 下低估,且 0.5 GiB 下限低于引擎最低需求;另有"算了 15% 余量却没用"的死变量。**待修**。
2. **间歇性解码停顿**:约 **2/5** 次运行出现 `p99 ITL` **10–14 秒**(引擎在若干秒内只产 0.2 tok/s)。
   已**排除**:chunk 大小、KV 压力(峰值 26.8%)、抢占/重算(无日志)、MTP、内存/swap、外部 CPU 争用;
   在隔离重复中 **3/3 未复现**。**待继续定位**(已备 `XIAOTU_LAYER_TIMING` 仪器)。
3. **专家层常驻 GPU**:机制可用(实测每层每 rank 1.59 GiB),prefill **+4%** 但 decode **−2~6%** ⇒ 净收益抵消,**不作默认**。
4. **层内 `pre` 远大于 `eng`**:`XIAOTU_LAYER_TIMING` 实测每层 `pre`(我们的 Python 编排)中位 **228 ms**、
   `eng`(引擎调用)仅 **12 ms** ⇒ **编排层是主成本**,是后续优化方向(与 §515 的旧结论同向)。

---

## 四、上游动向(只读核对,不新增竞争 PR)

* **CED(#56752)**:代码作者是他人(其 WIP 被 rebase),**我们不认领**,只做独立验证;
  维护者跟踪 issue **#57448** 仍把 "Decoder: #56752" 列为未完成。
* **MiMo dflash**:上游**有** `dflash`/`dspark` 通道,但草稿模型缺 **EAGLE3 接口**
  (`RuntimeError: Model does not support EAGLE3 interface`)⇒ 按既定口径**只调研、不实现**。
* **SGLang #37983**(视觉 Triton 丢 window+sink):**已有修复 PR(#38142)**,因此**不提竞争 PR**,
  改为在 **A100/SM80** 上做独立验证(独立实测:补丁内核 0.26–0.27% 误差,修复前 65–95% 偏;full-attn 新旧逐位相同)。

---

## 五、文档与脚本

* 新增接入必读两条(**`docs/MODEL_GUIDES.md` §0.1**):
  `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` 的「0 = 关闭」陷阱;`--max-num-seqs` 默认 4 的规则。
* 新增 `docs/EXPERIMENTS.md` **B92–B128**(MiMo 接入全过程、口径修正、两个异常的分析与否证)。
* 新增脚本:`scripts/serve_mimo26.sh`;`scripts/serve_v41.sh` / `serve_glm53_mainline.sh` 增加旋钮。

**可复现**:所有原始 JSON 在 `dev-docs/report/tuning/raw/`;协议(`random` / 输入 128 或 16384 / 输出 128 /
每格 8 请求 / 每格不同 seed)与 README 表格一致。
