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

> **📌 当前版本:v0.2.3**(2026-09-20)—— **跟进上游 + GLM/MiMo 的 MTP**:
> 补丁栈 rebase 到上游 `133b71e0b`;**GLM-5.3-Flash 的 MTP 落地并默认开**;
> **MiMo-V2.5(310B/15B)单卡端到端跑通**;GLM 显存契约重标定(`GPU_UTIL` → **0.82**)。
> 发行说明:[`RELEASE_NOTES_v0.2.3.md`](RELEASE_NOTES_v0.2.3.md)

---

## 目标与愿景

1. **任意 MoE 模型**:不绑死某一代架构。已跑通 DeepSeek-V4 / V4.1 系列,
   以及 **GLM-5.3-Flash**(含 A100 / SM80 的注意力后端,见支持矩阵)。
2. **任意 x86 指令集**:`scalar → AVX2 → AVX-512(base/VNNI/BF16/VBMI)`,
   运行时按 `/proc/cpuinfo` 自动选最高可用变体。
3. **显存优先级固定不变**:`1M 上下文 → GPU 预填充 → 投机解码 → 常驻`;
   任何新功能都不得把这条顺序往后挤。
4. **内存只用一份**:专家权重在内存里只有 **1 份**(不是"参考源张量 + 引擎副本"两份)。
5. **引擎效率对标业界最强**:`xiaotu_moe` 与 `lk_moe` 在**同一台机器、同一份权重、
   同一线程数**下对比,目标是不低于其 90%。

---

## 支持矩阵

| 模型 | 规模 | 专家格式 | 状态 |
|---|---|---|---|
| **DeepSeek-V4.1-Flash** | 748B | MXFP4(E8M0 block-32) | ✅ 端到端(TP=2) |
| **DeepSeek-V4-Flash**(0731) | 256 专家 / top-6 | MXFP4 | ✅ 端到端(基准引用历史数据,0.2.3 起不复测) |
| **GLM-5.3-Flash** | 321B / 18B active、288 专家 / top-8 | FP8 block-128 | ✅ **端到端(TP=2,A100/SM80)**,交付配置 **256K 上下文 × 2 路并发**,需 `--kv-cache-dtype bfloat16`;**0.2.3 起 `GPU_UTIL` 必须用 `0.82`**;**MTP 默认开 `SPEC_K=1`**(decode 21.9→22.6 tok/s,KV 池 −27%;k>1 反而更差) |
| **MiMo-V2.5** | 310B / 15B active、256 专家 / top-8 | FP8 block-128 | ✅ **单卡 A100-40GB(TP=1)端到端**,Hybrid SWA-128 + DiffKV(`TRITON_ATTN_DIFFKV`);**MTP `num_speculative_tokens=1` 实测 decode +10%**,k>1 不可用 |
| 其它即插即用 MoE | — | BF16 / FP8 | ✅ 走通用路径,未逐模型标定 |

**实测硬件平台(下文所有性能数字都在这台机器上取得)**

| 项 | 配置 |
|---|---|
| CPU | 2× AMD EPYC 9654(192 物理核 / 384 线程,8 NUMA node) |
| 内存 | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| 系统 | Ubuntu 22.04 · conda env `lvllm`(vLLM 2.5.0 基座) |

---

## 性能(最新版本 **v0.2.3**)

> **v0.2.3 变更**:补丁栈 rebase 到上游 `133b71e0b`;**GLM 生产 `GPU_UTIL` 从 `0.85` 降到 `0.82`**
> (上游改了 KV 定容 ⇒ 0.85 下两路并发会 OOM);GLM 的 MTP **默认开 `SPEC_K=1`**
> (decode 小赚 +3%,代价是 **KV 池 −27%**、单 token 间隔脉冲化);
> **MiMo-V2.5 新增支持**,MTP k=1 实测 **decode 15.6 → 17.2 tok/s(+10%)**。
> 详见 [`RELEASE_NOTES_v0.2.3.md`](RELEASE_NOTES_v0.2.3.md)。

> **怎么读** —— 表里的数字都是**耗时(ms/层),越小越好**,即"同样的活干得更快"。
> 对照方是 `lk_moe`(Lvllm 的 CPU MoE 引擎):同一台机器、同一份真实层权重、同一线程数,
> 且**每个 arm 的 `lk_moe` 分母都在同一个 session 里现量**,避免跨时段漂移。
> **比值 < 1.0 表示我们更快。**

### GLM-5.3-Flash(2×A100-40GB,TP=2,SM80;交付配置 256K × 2 路)

官方 `vllm bench serve`,随机数据 + `--ignore-eos`,每格不同 seed。

| 并发 | prompt / output | prefill (tok/s) | decode (tok/s) | 完成 |
|---|---|---|---|---|
| C=1 | 256 / 128 | **127** | **21.6** | 8/8 |
| C=2 | 256 / 128 | 78 | 12.8 | 8/8 |
| **C=1** | **4096 / 64** | **180** | **21.3** | 2/2 |

> **口径(全 README 统一)**:`prefill (tok/s) = prompt_tokens / TTFT`(首 token 前的预填充速率)、
> `decode (tok/s) = 1000 / TPOT`(首 token 之后的解码速率,不含 prefill)。
> 两者都由同一次 `vllm bench serve` 的 TTFT / TPOT 换算而来;
> **不再公布 `output tok/s`(含 TTFT 的混合值)、TPOT、ITL**。
> C=2 的 prefill/decode 是**单流**值(两路各自),不是聚合吞吐。

* **长 prompt 预填充**:4096-in 的 **prefill 140 → 180 tok/s(1.29×)** ——
  FP8 GPU 预填充把每层 3.62 GB/rank 的专家权重逐层流式搬上 GPU,并与 attention **重叠**;
* **解码不变**:C=1 **21.3–21.6 tok/s**,与 v0.2.1 持平(GPU 预填充只作用于 prefill);
* **上下文**:256K × 2 路并发是**本硬件的交付目标**;上表的 KV 池是 v0.2.2(util 0.85)口径
  = 988,081 token,**0.2.3 rebase 后同一 util 会涨到 1,018,328 并把激活余量吃掉**(两路并发 OOM),
  故 **0.2.3 交付改用 `GPU_UTIL=0.82`,KV 池 915,487**(32k 单请求 / 两路 14k+15k 实测均通过);
  512K/704K 实测「能起」但只作能力记录(704K 仅 1.06× 并发,上限约 733K),
  **1M 不在目标内**(需 fp8 KV,已决定不做;见 [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md));
* **正确性**:引擎确定性门禁 11/11;层门禁 rms_rel 4.4e-3;29,746-token 长文密钥检索完全命中;
  3 路 ~8K 并发(限两路)三个密钥全部正确;0 OOM。
* **MTP(v0.2.3 起默认开 `SPEC_K=1`)**:draft 第 45 层常驻 GPU(3.38 GiB/rank);
  接受长度 1.46,`256/128 C=1` 解码 **21.9 → 22.6 tok/s**(≈+3%,噪声内);代价是
  **KV 池 915,487 → 666,366(−27%,256K 并发 3.49× → 2.54×)**,且**单 token 间隔会脉冲化**
  (一步吐 1–2 个 token,流式体验变差);**k>1 反而更差**(k=4 掉到 ~12 tok/s)。
  数据见 [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) §4.3。

> FP8 引擎内层另有一个**默认关闭**的加速开关 `XIAOTU_MOE_FP8_BF16_MMA=1`
> (AVX512-BF16 `vdpbf16ps`,M≥6 快 1.17-1.20×,代价是权重舍入 bf16:两路 rms_rel 3.6e-3)。

### MiMo-V2.5(单卡 A100-40GB,TP=1,专家在 CPU)

**解码 —— MTP A/B**(256 in / 128 out,C=1,每格 **N=8×2** 重复)

| 配置 | decode (tok/s) |
|---|---|
| 不开 MTP | 15.5 / 15.6 |
| **MTP k=1**(推荐) | **17.2 / 17.1(+10%)** |

**预填充 —— 与 MTP 无关,取决于是否开 GPU 流式预填充**(同机实测)

| 配置 | 256 in | 1024 in | 2048 in | 4096 in |
|---|---|---|---|---|
| 专家在 CPU(GPU 预填充关) | 68 | 81 | 85 | **86** |
| **+ GPU 流式预填充** | — | — | — | **194(2.25×)** |

> ⚠️ **MiMo 的 TTFT ≈ 0.84 s 固定 + 11.4 ms/token**,所以**短 prompt 的 prefill (tok/s) 会被固定开销压低**(256 token 只有 68 tok/s),必须按**长 prompt** 读这个指标。
> GPU 流式预填充的代价是 staging **12.75 GiB/rank**(比 GLM 的 7.59 大,因为 `E=256×I=2048` 更宽)⇒ 要用更低的 `GPU_UTIL`(实测 0.65)换激活余量,KV 池随之变小(47,459),8K 上下文下 5.8× 并发仍够用。


* 负载 = 256 in / 128 out / C=1;接受长度 **1.74**(accepted 869 / drafts 1169);
**贪心输出与不开 MTP 逐字节相同**;
* **MTP k=3 不可用**:accept 1.016(p0 从 0.83 崩到 0.016)⇒ 反而慢 2.4×,与上游 PR #31180
  的 *"acceptance rate of 0"* 一致 ⇒ 只开 `num_speculative_tokens=1`;
* 加载:293 GiB / 17 分片,每层建一次 xiaotu FP8 引擎,整轮 ~25-30 min;详见
  [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) §3。

### DeepSeek-V4.1-Flash 服务级(等参数同机 A/B,TP=2,官方 `vllm bench serve`)

两个 arm 跑在**同一个 conda env**、唯一变量是 CPU MoE 引擎;prompt 逐字节相同、线程数都是 60。
**只报两个速率,不做折算**:`prefill (tok/s)`(首 token 前)与 `decode (tok/s)`(首 token 后)。
比值为 **我们 / `lk_moe`**,**> 1.0 表示我们更快**。

| prompt / output | `lk_moe` prefill (tok/s) | 我们 prefill (tok/s) | 比值 | `lk_moe` decode (tok/s) | 我们 decode (tok/s) | 比值 |
|---|---|---|---|---|---|---|
| 256 / 32 | 118.9 | 107.1 | **0.900×** | 44.7 | 31.1 | **0.695×** |
| 256 / 1024 | 118.6 | 105.1 | **0.886×** | 44.9 | 33.5 | **0.746×** |
| 8192 / 32 | 127.5 | 115.0 | **0.902×** | 45.4 | 34.3 | **0.756×** |
| 8192 / 1024 | 127.6 | 120.9 | **0.948×** | 44.8 | 34.7 | **0.775×** |

⇒ 服务级我们**仍慢**:prefill 慢 **5-11%**、**decode 慢 29-44%**(decode 只有 lk 的 0.70-0.78×),
但比上一版已收窄 **+8%~+15%**。

### GPU 预填充(把长 prefill 交给 GPU)

把长 prefill 的专家计算逐层流式搬上 GPU,支持两种专家格式:

* **MXFP4**(DeepSeek-V4.1-Flash 等):客户端 TTFT 实测 **快 2.0-2.8×**(阈值 ≥4096 才划算);
* **FP8 e4m3 block-128**(GLM-5.3-Flash):µbench 一样把长 prompt 的 TTFT 从
  **29.3 s 压到 22.8 s(1.29×,4096-in)。两点值得注意:
  1. 权重装配是 **3.62 GB/rank 的纯 H2D 固定成本**,已到本机 26.86 GB/s 的 H2D 天花板
     (1-D 与 pitched 2-D 同速,锁页/NUMA 交错/两 rank 并发都不改变),所以优化点是
     **把装配与 attention 重叠**(默认开启的 side stream),不是"搬得更快";
  2. GLM-5.3 的预填充 chunk 被 KDA state 的 `block_size=2176` 钉死,因此
     **插件默认阈值 4096 对 GLM 永远不会触发** —— `scripts/serve_glm53_mainline.sh`
     已把 GLM 专用默认改为 `GPU_PREFILL_MIN=1500`(实测盈亏平衡 ~1300)。

CPU 侧另有一个可选开关 `XIAOTU_MOE_FP8_BF16_MMA=1`:FP8 内层改走 AVX512-BF16
`vdpbf16ps`,**M≥6 快 1.17-1.20×**(全 CPU 预填充端到端 TTFT 1.12×),代价是权重被舍入到
bf16(两路 rms_rel 3.6e-3),**因此默认关闭**;`M=1` 的单流解码仍走精确 fp32 路。

开启方式与显存配方见 [`docs/RUNBOOK.md`](docs/RUNBOOK.md);

**完整的服务级对照、口径说明与复现命令**:[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)、
[`RELEASE_NOTES_v0.2.1.md`](RELEASE_NOTES_v0.2.1.md)。

---

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
| **v0.2.3** | **跟进上游 + GLM/MiMo 的 MTP** —— 补丁栈 rebase 到上游 `133b71e0b`(11 补丁/40 文件);**GLM-5.3-Flash 的 MTP 落地**(draft 层识别 + GPU 常驻,`SPEC_K=1..4`);**MiMo-V2.5(310B/15B)单卡端到端支持**,MTP k=1 decode +10%;显存契约重标定(GLM `GPU_UTIL` 0.85 → **0.82**) |
| **v0.2.2** | **支持 GLM-5.3-Flash** —— FP8 GPU 预填充接线(4K prompt TTFT 29.3 → 22.8 s)、256K × 2 路并发交付配置;修掉 e4m3 次正规数解码缺陷 + 新增全码字门禁 |
| **v0.2.1** | **针对引擎的显著性能优化** —— CPU MoE 引擎在全部真实形状上**反超 `lk_moe`**;DeepSeek-V4-Flash 同步受益 |
| v0.2 | DeepSeek-V4.1-Flash 全链路可用(1M 上下文 + GPU 预填充 + 投机解码)+ CPU 预填充路径优化 |
| v0.1.0 | 首个公开版:混合模式(CPU 专家 + GPU 其余)、AVX2 / AVX-512 多 ISA、DeepSeek-V4 系列 |

改动清单、性能对照与运行参数变更:
[**v0.2.3**](RELEASE_NOTES_v0.2.3.md) · [**v0.2.2**](RELEASE_NOTES_v0.2.2.md) · [**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
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
