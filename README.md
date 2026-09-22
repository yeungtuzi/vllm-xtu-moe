# vllm-xtu-moe

> **XTU = X Transformers Unity**(读音:汉语「小兔」)—— 一个**高性能 MoE 推理加速层**。
>
> 这是一个面向**超大 MoE 模型**的 **vLLM 插件**,核心思路是把计算拆开:
>
> * **GPU** 负责注意力、KV cache、embedding 等**非专家部分**;
> * **CPU** 负责**专家权重**与**专家计算**;
> * 对**长 prompt**,还会把**一部分专家计算流式搬到 GPU**。
>
> 也就是说:在**"显存放不下专家权重"**的情况下,仍然让大 MoE 模型跑起来,
> 而且**尽量不改主线 vLLM**。
>
> **项目重点**
>
> * **DeepSeek / GLM / MiMo** 这类**超大 MoE**
> * **CPU + GPU 混合推理** · **低显存需求**
> * **高性能长上下文推理**
> * **兼容主线 vLLM,不需要 fork**
> * 主要面向**大模型部署与工程优化**

[**English**](README_EN.md) · 中文(默认)

> **📌 当前版本：v0.2.4**（2026-09-22）—— **支持 MiMo-V2.6-Flash-RL + 性能优化**：
> **MiMo-V2.6-Flash-RL 接入**（专家是 MXFP4 ⇒ 复用 V4.1 那条引擎路径；**TP=2 / 1M 上下文 / 多模态 / MTP k=1** 全部实测）；
> **GPU 预填充「0=关闭」陷阱修复**（MiMo 长 prefill **313 → 811 tok/s**）；**`--max-num-seqs` 默认统一为 4**（修掉 C≥2 被退化成串行）；
> **decode 口径改为 median ITL**（真实解码争用只有 **1.13–1.77×**）。
> 发行说明：[`RELEASE_NOTES_v0.2.4.md`](RELEASE_NOTES_v0.2.4.md)

---

## 项目特色

1. 任意 MoE 模型:不绑死某一代架构。已跑通 DeepSeek-V4 / V4.1 系列、
   GLM-5.3-Flash(含 A100 / SM80 的注意力后端)与 MiMo-V2.5。
   **MiMo-V2.6-Flash-RL 也已在 A100/SM80 上跑通**:骨架 / 1M 上下文 / 真权重长文本端到端均已实测通过
   **多模态**——图片 ✅（渲染 `ZQ7K42` 读对）、视频 ✅（红方块「出现又消失」，连**时序**都对）、**组合**（图 + 5,444 token 长文本 + MTP k=1）✅；**音频**需在**启动前**装好 `vllm[audio]`（含 `torchaudio`）才能用，见 `docs/MODEL_GUIDES.md` §0.1c。它走的是与 V4.1 相同的 **MXFP4** 引擎路径，体积 161 GiB ——
   见 `docs/MODEL_GUIDES.md` §3b 与 `docs/EXPERIMENTS.md` B92–B133。
2. 任意 x86 指令集:`scalar → AVX2 → AVX-512(base/VNNI/BF16/VBMI)`,
   运行时按 `/proc/cpuinfo` 自动选最高可用变体。
3. 显存优先级固定不变:`KV 池 → GPU 预填充 staging → 投机解码 draft → 激活工作区(∝ MBT)`;
   任何新功能都不得把这条顺序往后挤。
   > 2026-09-20 修订:①去掉"专家层常驻" —— 实测效果很差(每层 3.36 GiB,
   > 换来的收益抵不上它对 32K 预填充的挤压,用户裁定不再列为优先级);
   > ②补上"激活工作区" —— 它正比于 chunk(=MBT),是之前连续两次长 prompt OOM 的真凶,
   > 却一直没被列进优先级。
4. 单服务器多 NUMA 节点也只占用一份内存:每个 NUMA 节点只存取本地内存,
   在最大化性能的同时节省了内存占用。

---
## 实测平台

下文所有性能数字都在这台机器上取得:

| 项 | 配置 |
|---|---|
| CPU | 2× AMD EPYC 9654(192 物理核 / 384 线程,8 NUMA node) |
| 内存 | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| 系统 | Ubuntu 22.04 · conda env `lvllm` |

---

## 性能

* 口径：**聚合吞吐**——`prefill (tok/s) = 并发数 × prompt_tokens / TTFT`、`decode (tok/s) = 并发数 × 1000 / median(ITL)`
  > `decode` 取 **median ITL**(相邻 token 间隔中位数)而不是 median TPOT:`median TPOT` 是**每请求平均**,会把**同一 step 里混进来的 prefill 工作**算进解码时间,导致长上下文 C=2 时被严重低估(实测同一批数据:median TPOT 口径下 MiMo 长 C=2 = 9.8,而 median ITL 口径 = **39.6**,真实争用只有 **1.36×**;GLM 更极端:2.9 vs **29.6**)。详见 `docs/EXPERIMENTS.md` B126。
  （C=1 时即单流速率；等价于「该阶段的总 token 数 ÷ 该阶段墙钟」）。均由同一次 `vllm bench serve` 换算；
  `decode` 取 **median ITL**（`mean` 会被极少数离群步拉高,`median TPOT` 会被混入的 prefill 拉低）。
  > ℹ️ **decode 与 C 的关系**:真实解码争用很小(倍率 **1.13–1.77×**,按 median ITL);早期表里曾出现「长 C=2 的 decode 明显低于 C=1」,那是 **median TPOT 被混入的 prefill 步污染**所致,另一部分来自把 `--max-num-seqs` 设成 1 导致 C=2 退化为串行。两者均已修正,判据见 `docs/EXPERIMENTS.md` **B124/B126**。
* 数据集：**random 随机 token**，`--random-input-len` 固定为短 128 / 长 16384，输出 128；前缀缓存开；每格 8 个请求、**每格不同 seed**。
* 表中「实际 token」= `total_input_tokens / completed`（含 chat template 的少量开销）。
  配置细节、判据与 `MBT` 取舍见 `docs/TUNING_GUIDE.md` §8 与 `docs/EXPERIMENTS.md`。

| 模型(最优配置) | prompt | 实际 token | 并发 | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>TP=2 · util 0.85 · **MAXLEN 262144**<br>MBT 12288 · MTP 关 | 短 | 140 | 1 | **115.2** | **22.0** |
| | 短 | 140 | 2 | **140.8** | **27.8** |
| | 长 | 16,396 | 1 | **266.3** | **21.4** |
| | 长 | 16,396 | 2 | 待测 | 待测 |
| **DeepSeek-V4.1-Flash**<br>TP=2 · util 0.85 · **MAXLEN 262144**<br>MBT 8192 · **CED 开**（长两行） · 投机解码关 | 短 | 128 | 1 | **254.3** | **19.8** |
| | 短 | 128 | 2 | **350.4** | **28.0** |
| | 长 | 16,384 | 1 | **903.9** | **19.3** |
| | 长 | 16,384 | 2 | **1204.2** | **15.4** |
| **MiMo-V2.6-Flash-RL**<br>TP=2 · util 0.85 · **MAXLEN 131072**<br>MBT 8192 · **seqs 4** · **MTP k=1** · **GPU 预填开(门槛 4096)** · 含形状预热 | 短 | 128 | 1 | **232.0** | **30.5** |
| | 短 | 128 | 2 | **367.5** | **40.7** |
| | 长 | 16,384 | 1 | **811.3** | **27.3** |
| | 长 | 16,384 | 2 | **1455.3** | **39.6** |

GLM / DeepSeek-V4.1 未开投机解码（random 数据集对投机是最坏情况，且会挤占长上下文的显存）；**MiMo-V2.6 开了 MTP k=1**。MiMo 的 prefill 数字是 **GPU 预填开启后**的口径（`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=4096`；该变量不设或为 0 表示**关闭**，会让 prefill 掉到 ~310，见 `docs/MODEL_GUIDES.md` §0.1）。
> ⚠️ **长上下文 C=2 的 decode(9.8)是实测到的并发争用**:median TPOT 从 36.6 跳到 **203.6 ms**(同族的 GLM 长 C=2 更严重,45.9 → 700.8 ms;见 `docs/EXPERIMENTS.md` B115/B125)。
> ⚠️ **`--max-num-seqs` 必须 ≥2**:早期用 `seqs=1` 测过一批,`C=2` 会退化成**串行**、把短 C=2 的聚合 prefill 压到 54(那是**配置问题,不是模型问题**);判据见 **B124**。其余格子均为同一批实测。

## 快速开始

```bash
# 1) 主线 vLLM(本插件是插件,不需要 fork)
pip install vllm==2.5.0

# 2) 本插件(发行名 vllm-xtu-moe,**不发 PyPI**;二选一)
# (a) 从 GitHub Release 附件装:wheel 里已含 6 个 ISA 变体,无需本地编译器
pip install ./vllm_xtu_moe-0.2.2-cp312-cp312-manylinux_2_34_x86_64.whl
# (b) 源码安装(需要本地编译器):
# CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh && pip install -e .

# 3) 跑一个专家权重放不进显存的 MoE 模型
VLLM_EXPERTS_LOAD_DEVICE=cpu \
vllm serve <MODEL_DIR> --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 65536 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95
```

逐模型配方、显存核算、自检与排错 → **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**、
**[`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md)**。

---

## 版本变更(简)

| 版本 | 主题 |
|---|---|
| **v0.2.4** | **支持 MiMo-V2.6-Flash-RL + 性能优化** —— 新模型接入（MXFP4/TP=2/1M/多模态/MTP k=1，**层内数值门禁 94/94 层通过**、312K 针测试命中、多模态真图通过）；**GPU 预填充「0=关闭」陷阱修复**（长 prefill 313→**811** tok/s）；**`--max-num-seqs` 默认统一为 4**（短 C=2 prefill 236→**367**）；**decode 口径改为 median ITL**（真实争用 1.13–1.77×）；修好 MXFP4 上的层内数值门禁 |
| **v0.2.3** | **跟进上游 + GLM/MiMo 的 MTP** —— 补丁栈 rebase 到上游 `133b71e0b`(11 补丁/40 文件);**GLM-5.3-Flash 的 MTP 落地**(draft 层识别 + GPU 常驻,`SPEC_K=1..4`);**MiMo-V2.5(310B/15B)单卡端到端支持**,MTP k=1 decode +10%;显存契约重标定(GLM `GPU_UTIL` 0.85 → **0.82**) |
| **v0.2.2** | **支持 GLM-5.3-Flash** —— FP8 GPU 预填充接线(4K prompt TTFT 29.3 → 22.8 s)、256K × 2 路并发交付配置;修掉 e4m3 次正规数解码缺陷 + 新增全码字门禁 |
| **v0.2.1** | **针对引擎的显著性能优化** —— CPU MoE 引擎在全部真实形状上**反超 `lk_moe`**;DeepSeek-V4-Flash 同步受益 |
| v0.2 | DeepSeek-V4.1-Flash 全链路可用(1M 上下文 + GPU 预填充 + 投机解码)+ CPU 预填充路径优化 |
| v0.1.0 | 首个公开版:混合模式(CPU 专家 + GPU 其余)、AVX2 / AVX-512 多 ISA、DeepSeek-V4 系列 |

改动清单、性能对照与运行参数变更:
[**v0.2.4**](RELEASE_NOTES_v0.2.4.md) · [**v0.2.3**](RELEASE_NOTES_v0.2.3.md) · [**v0.2.2**](RELEASE_NOTES_v0.2.2.md) · [**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
(归档:[v0.2pre](RELEASE_NOTES_v0.2pre.md))

---

## 文档

**面向使用者 —— [`docs/`](docs/)**

| 文档 | 内容 |
|---|---|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | **运行手册**:安装、启动参数、显存核算、GPU 预填充配方、JIT 缓存、发布流程 |
| [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) | 逐模型的资源需求、启动命令与性能 |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | 实测数据与复现方式 |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | 已知限制与不支持的组合 |
| [`docs/INSTALL_MAINLINE.md`](docs/INSTALL_MAINLINE.md) | 主线 vLLM 环境准备 |

**面向开发者** —— 架构与主线集成、GPU 预填充实现、上游漂移、调优记录与全部内部报告,
**属于内部开发文档,不随本仓库发布**(只保留在本地工作副本里)。

---

## 致谢

本项目受到 **KTransformers** 和 **Lvllm** 项目启发,尤其是计算引擎 **`xiaotu-moe`** 深度借鉴了 **lk-moe** 思路,
**`xiaotu-moe` 全部代码均为独立编写**,特此致谢。

* KTransformers —— <https://github.com/kvcache-ai/ktransformers>
* Lvllm(及其 CPU MoE 引擎 **lk-moe**;本项目对照的 `lk_moe` 2.4.2 即来自此仓库)—— <https://github.com/guqiong96/Lvllmds4-x>
* 同时感谢 **vLLM**(<https://github.com/vllm-project/vllm>)提供的插件式扩展点,使本项目无需 fork 即可接入。

---

## 许可

Apache-2.0。第三方组件与致谢清单见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`NOTICE`](NOTICE)。
