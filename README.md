# vllm-xtu-moe

> **XTU = X Transformers Unity**(读音:汉语「小兔」)。
> 一个 **vLLM 插件**:让 **MoE 专家权重住在 CPU 内存、注意力与 KV cache 留在 GPU**,
> 从而在显存放不下专家权重的机器上把超大 MoE 跑起来。**不改主线、不需要 fork。**

[**English**](README_EN.md) · 中文(默认)

> **📌 当前版本:v0.2.1**(2026-09-18)—— **针对引擎的显著性能优化**:
> CPU MoE 引擎在全部真实形状上**反超参考实现 `lk_moe`**,DeepSeek-V4-Flash 同步受益。
> 发行说明:[`RELEASE_NOTES_v0.2.1.md`](RELEASE_NOTES_v0.2.1.md)

---

## 它解决什么问题

超大 MoE(DeepSeek-V4.1-Flash 748B、V4-Flash、GLM-5.3-Flash 321B…)的专家权重动辄
150-300 GiB,任何单卡/双卡都放不下;而**专家只占每 token 计算的一小部分**,
剩下的是注意力与共享层 —— 它们在 GPU 上跑得很快。于是把两者拆开:

* **非专家**(注意力、KV cache、embedding)→ GPU,正常走 vLLM 的高性能内核;
* **专家** → CPU 内存,由 `xiaotu_moe` 引擎按 NUMA 分片计算;
* **长 prefill**(token 数 ≥ 阈值)→ 再把专家逐层流式搬上 GPU 算。

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
| **DeepSeek-V4-Flash**(0731) | 256 专家 / top-6 | MXFP4 | ✅ 端到端 |
| **GLM-5.3-Flash** | 321B / 18B active、288 专家 / top-8 | FP8 block-128 | ✅ **端到端(TP=2,A100/SM80)**,需 `--kv-cache-dtype bfloat16`;贪心可复现,C=1 解码 ~22 tok/s(TPOT 45 ms);4K prompt TTFT **29.3 s(CPU)→ 22.0-22.6 s(FP8 GPU 预填充)** |
| 其它即插即用 MoE | — | BF16 / FP8 | ✅ 走通用路径,未逐模型标定 |

**实测硬件平台(下文所有性能数字都在这台机器上取得)**

| 项 | 配置 |
|---|---|
| CPU | 2× AMD EPYC 9654(192 物理核 / 384 线程,8 NUMA node) |
| 内存 | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| 系统 | Ubuntu 22.04 · conda env `lvllm`(vLLM 2.5.0 基座) |

---

## 性能(最新版本 **v0.2.1**)

> **怎么读** —— 表里的数字都是**耗时(ms/层),越小越好**,即"同样的活干得更快"。
> 对照方是 `lk_moe`(Lvllm 的 CPU MoE 引擎):同一台机器、同一份真实层权重、同一线程数,
> 且**每个 arm 的 `lk_moe` 分母都在同一个 session 里现量**,避免跨时段漂移。
> **比值 < 1.0 表示我们更快。**

### DeepSeek-V4.1-Flash(真实路由形状 `na≈226`,60 线程,单位 ms/层)

| 形状 | `xiaotu_moe` | `lk_moe` | 比值 |
|---|---|---|---|
| BS=227(每专家 ≈6 行) | **28.56** | 30.24 | **0.944×** |
| BS=1893 | **213.36** | 217.34 | **0.982×** |
| BS=8192 | **877.55** | 933.94 | **0.940×** |
| 解码 BS=1 | **0.37** | 0.43 | **0.861×** |

### DeepSeek-V4-Flash(0731,同协议)

| 形状 | `xiaotu_moe` | `lk_moe` | 比值 |
|---|---|---|---|
| BS=227 | **20.89** | 22.18 | **0.942×** |
| BS=1893 | 162.05 | **155.86** | 1.040× |
| BS=8192 | **646.60** | 668.10 | **0.968×** |
| 解码 BS=1 | 0.29 | 0.29 | 1.000× |

### DeepSeek-V4.1-Flash 服务级(等参数同机 A/B,TP=2,官方 `vllm bench serve`)

两个 arm 跑在**同一个 conda env**、唯一变量是 CPU MoE 引擎;prompt 逐字节相同、线程数都是 60。
**三项都是独立量,不做折算**——`output tok/s` 是整段解码的平均速率(**包含 TTFT**),
TTFT 与 TPOT(**不含 TTFT**)各自单独公布。

| prompt / output | `lk_moe` out tok/s | 我们 out tok/s | 比值 | `lk_moe` TTFT | 我们 TTFT | 比值 | `lk_moe` TPOT | 我们 TPOT | 比值 |
|---|---|---|---|---|---|---|---|---|---|
| 256 / 32 | 11.24 | 9.44 | **0.840×** | 2153 ms | 2391 ms | 1.111× | 22.37 ms | 32.19 ms | 1.439× |
| 256 / 1024 | 41.08 | 31.07 | **0.756×** | 2159 ms | 2436 ms | 1.128× | 22.26 ms | 29.83 ms | 1.340× |
| 8192 / 32 | 0.49 | 0.44 | **0.898×** | 64259 ms | 71242 ms | 1.109× | 22.03 ms | 29.13 ms | 1.322× |
| 8192 / 1024 | 11.76 | 10.53 | **0.895×** | 64208 ms | 67734 ms | 1.055× | 22.33 ms | 28.82 ms | 1.291× |

⇒ 服务级我们**仍慢 10-24%**(`ttft` 高 5-13%,纯解码 `TPOT` 高 29-44%),
但比上一版已收窄 **+8%~+15%**。

### GPU 预填充(把长 prefill 交给 GPU)

把长 prefill 的专家计算逐层流式搬上 GPU,支持两种专家格式:

* **MXFP4**(DeepSeek-V4.1-Flash 等):客户端 TTFT 实测 **快 2.0-2.8×**(阈值 ≥4096 才划算);
* **FP8 e4m3 block-128**(GLM-5.3-Flash):µbench 一样把长 prompt 的 TTFT 从
  **29.3 s 压到 22.0-22.6 s(1.3×)**。两点值得注意:
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

# 2) 本插件:先构建引擎原生扩展(6 个 ISA 变体),再安装
CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh
pip install -e .

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
| **v0.2.1** | **针对引擎的显著性能优化** —— CPU MoE 引擎在全部真实形状上**反超 `lk_moe`**;DeepSeek-V4-Flash 同步受益 |
| v0.2 | DeepSeek-V4.1-Flash 全链路可用(1M 上下文 + GPU 预填充 + 投机解码)+ CPU 预填充路径优化 |
| v0.1.0 | 首个公开版:混合模式(CPU 专家 + GPU 其余)、AVX2 / AVX-512 多 ISA、DeepSeek-V4 系列 |

改动清单、性能对照与运行参数变更:
[**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
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
