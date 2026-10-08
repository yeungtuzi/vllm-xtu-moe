# INT8 激活 × MXFP4 专家权重 —— 计算路径详解

> ⚠️ **命名变更(2026-10-07,用户定案)**:本文件里出现的旧环境变量名**已修改为**下面的规范名 ——
> `XIAOTU_MOE_W4A8` **已修改为** `XIAOTU_MOE_INT8`(int8 激活路径**总开关**,默认由权重格式决定);
> `XIAOTU_MOE_INT8_ALIGN` **已修改为** `XIAOTU_MOE_INT8_ALIGN`(实现选择:ALIGN tile,默认档);
> `XIAOTU_MOE_INT8_VNNI` **已修改为** `XIAOTU_MOE_INT8_VNNI`(实现选择:旧 VNNI tile,默认 0);
> `XIAOTU_MOE_I8_MIN_M` **已修改为** `XIAOTU_MOE_INT8_VNNI_MIN_TOKENS`(⭐ M 阈值,**默认 160**,唯一设定处)。
> 旧名仍被识别:会**告警一次**并映射到新名(**仍按旧值生效**),绝不静默退回 fp32 ✓


> 本文是本次发布的详细说明版,面向**使用方**。概述见 [README](../README.md),
> 原始实测数据见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。
> ⚠️ 文中标注"内部 harness"的复现命令**不随发行提供**(属本项目的内部开发工作区);其口径已在 §二/§三 写明,便于自行复刻。

---

## 一、这条路径解决什么问题

本引擎在 A100(SM80)上以 **CPU 承担路由专家计算**、GPU 承担注意力与稠密层的方式服务
qfn(第三方量化的 Qwen3.8-Flash-Next,**MXFP4-FP8**)。其中专家权重是 **MXFP4**:
每 32 个权重共享一个 `e8m0` 指数缩放,权重以 4-bit nibble 打包存放。

旧路径的瓶颈很明确:**权重是 4-bit,但计算在 bf16/fp8 上做**,即"先解包 → 再转高精度 → 再乘加"。
本项工作为它增加一条 **INT8 激活 × FP4 权重** 的计算路径(内部代号 W4A8 / ALIGN),
把激活也压到 8-bit 有符号整数,使乘加走 AVX512-VNNI 的整数点积指令。

关键点在于**权重侧零代价**:fp4 的 16 个码值取值集合为 `{0, ±1…±12}`,
**完全包含于 int8 的可表示范围** ⇒ fp4→int8 的权重转换是**逐位精确**的,不引入任何误差。
从 fp4 到 W4A8 的全部数值代价都来自**激活的 int8 量化**这一项。

---

## 二、实测收益(内核级)

真权重、真激活,单层 MoE,门控关闭对照,5 轮交错取中位数:

| 臂 | median(ms/层) | 相对 |
|---|---|---|
| base 实现(生产默认) | 24.9 | 1.00× |
| base 实现 + `-mavx512vnni` 旗标 | 24.5 | 1.02×(旗标本身无罪) |
| W4A8 源码,门控**关**(回退路径) | 29.1–31.3 | 0.78–0.85× |
| **W4A8 源码,门控开(ALIGN int8)** | **15.1–15.7** | ⭐ **1.54–1.62×** |

数值代价:max_abs ≈ **3.163e-03**;离线四路对拍确认该量级为激活 int8 量化的固有代价。

---

## 三、端到端语义验收

模型 = qfn;精度 = MXFP4-FP8 第三方量化;实验线 = ①VNNI-int8。

统一口径:`MTP k=3` + CPU 预填 + `THREADS=120` + `TASKSET=0-191` + `maxtok=1024` + `seqs=1` + `TP=1`/GPU0。

| 验收项 | base 臂 | int8 臂 | 逐项一致 |
|---|---|---|---|
| GSM8K 全量 200 题 | **193/200 = 0.9650** | **193/200 = 0.9650** | 194/200,**零净变化** |
| 256K 四针 | **4/4** + `finish_reason=stop` | **4/4** + `finish_reason=stop` | 完全一致 |
| Vision 23 | 18/23 | 17/23 | 18/23 |

> ⚠️ **关于"逐题一致"的正确读法**:**端到端(经 vLLM)不是逐次运行可复现的** ——
> 同一 prompt、greedy、固定 seed 连跑 10 次,基线自身就会产出 7–8 条不同的 token 序列。
> 因此上表残留的少数翻转属基线抖动,**不得**读作 int8 改造引入的缺陷 ✗。
> 判据是"**两臂分布无可比系统性偏差**",而非"逐次 token 严格相等"。
>
> 📌 **归因(已实测)**:引擎在**单 rank、同输入**下是**逐位确定的**
> (`scripts/test_engine_determinism.py`:12 次运行 **11/11 逐位相同**)
> ⇒ 不确定性来自 **TP=2 / EP 归约** 或 **vLLM 侧**,**不在引擎内核**。

### 内核级一致性(离线、同输入同权重)

`align_on` 与 `align_off` 两臂的 `w13` / `w2` / `s13` / `s2` 四个张量逐字节相同;
门控关 ≡ base 实现**逐字节一致**;门控开 ≡ 重构前**逐字节一致**。

---

## 四、如何启用

W4A8 路径由**检查点量化格式驱动**,默认**不启用**:

* 插件侧开关:`cfg.int8_activation`(`0`=off / `1`=ALIGN int8 / `2`=旧 VNNI 实现)
* 环境变量:`XIAOTU_MOE_INT8=1` 才打开
* 引擎变体:`avx512_bf16_vbmi_vnni`(int8 派发) 对比 `avx512_bf16_vbmi`(base 实现,生产默认)

引擎构建:

```bash
# 在仓库根目录执行(需要本机 C++ 工具链)
ONLY="avx512_bf16_vbmi_vnni" bash scripts/build_engine_variants.sh
```

**回退**:不设该 env 即退回 base 实现,行为与旧版逐字不变。

---

## 五、本次同时完成的相对工作

### 5.1 T1.4b/c noinline 重构(门控关路径回归修复)

旧版在门控**关**时走回退路径,耗时是 base 实现的 **0.78–0.85×**(即慢 16–22%)。
归因收窄两份 int8 tile 体在 `XIAOTU_TILE_BODY` 宏里被多组 `(MRT,NRT)` 实例化 ⇒
代码体积 / I-cache / 寄存器压力。将两个 int8 体移出宏、改为 `__attribute__((noinline))`
模板函数(`acc` 以指针传入,常量用 `std::integral_constant<int,MRT/NRT>` 传)之后:

| 指标 | 修复前 | **修复后** |
|---|---|---|
| 回退路径指令数 | −16% | **+1.49%** |
| 回退路径周期 | — | **+2.64%** |

### 5.2 MTP(多令牌预测)

MTP 已在 qfn 上打通。根因是 **compressed-tensors 的 FP8 MoE 方法在块量化下把 scale 参数名写死成
`w2_weight_scale`**,而检查点与映射表用的是 `weight_scale_inv`(vLLM 自身 `Fp8Config` 即此口径,
源码注释:*"name is `weight_scale` for tensor, `weight_scale_inv` for block"*)。
qfn 主体专家是 MXFP4、不走这条 FP8 路,**唯一暴露点就是 MTP 层**。

长推理档实测解码 **34.8–38.6 tok/s**(MTP 关 = 12.2–12.5 tok/s),平均 **2.44×**。
⚠️ 该修复目前是 **dev-only 补丁**,尚未进入正式补丁系列。

---

## 六、已知限制

* **GPU 预填充的性能取舍不在本次范围内**。本次只要求该路径行为正确。
  (旁注:在只有 PCIe x8 的 GPU0 上,GPU 预填充会因逐层权重 staging 的 DMA 开销而慢于 CPU 预填;
   生产用 GPU1/GPU2 是 x16,结论不可互推 ✗)
* **NVFP4 可用性未验证** — 契约侧已证,但缺少真实 NVFP4 抽样。
* **端到端的运行间非确定性**(见 §三 注):不影响语义正确性,但使"逐次 token 严格相等"类判据失效 ✗
  (⭐ 单 rank 引擎本身**逐位确定** —— 不确定性来自 TP=2/EP 归约或 vLLM 侧)。
* 巨模型(GLM-5.3-Flash 306 G / DeepSeek-V4.1-Flash 476 G)不用于实验,只用结构相似的替代模型。

---

## 七、复现

**公开可复现的部分**(在仓库根目录执行):

```bash
# 构建 int8 派发变体
ONLY="avx512_bf16_vbmi_vnni" bash scripts/build_engine_variants.sh

# 引擎单 rank 逐位确定性自检(本版实测:12 次运行 11/11 逐位相同)
python scripts/test_engine_determinism.py

# 引擎侧一致性
python scripts/test_fp8_decode_conformance.py
python scripts/test_wna16_repack.py
```

> ⚠️ **内核级 A/B 与端到端三项语义验收所用的 harness / 启动脚本属内部开发工作区,不随发行提供**。
> 其口径与判据已在 §二、§三 写明,便于使用方在自有环境复刻:
> * **内核级**:真权重 + 真激活、单层 MoE、门控关作对照、**5 轮交错取中位数**;
> * **端到端**:`MTP k=3` + CPU 预填 + `THREADS=120` + `TASKSET=0-191` + `maxtok=1024` + `seqs=1` + `TP=1`,
>   判据为**两臂分布无可比系统性偏差**(不是逐次 token 相等)。
