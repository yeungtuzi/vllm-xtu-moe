# 实测数据

以下数字全部来自本项目的实测环境;原始数据与绘图脚本在 `report/`,
图表在 `内部数据 fig/`。

---

> ### ✅ v0.2.1 最新数据(2026-09-18)—— **CPU 引擎反超参考实现**
>
> **所有数字都是耗时(ms/层),越小越好**;比值 = 我们 ÷ `lk_moe`,**< 1.0 表示我们更快**。
> `lk_moe` 的分母**每个 session 现量**,不复用历史值(跨时段漂移 1.7-3%)。
> 完整说明:[`../RELEASE_NOTES_v0.2.1.md`](../RELEASE_NOTES_v0.2.1.md) §2。
>
> **DeepSeek-V4.1-Flash**(真实路由形状 `na≈226`,60 线程):
>
> | 形状 | `xiaotu_moe` | `lk_moe` | 比值 |
> |---|---|---|---|
> | BS=227 | **28.56** | 30.24 | **0.944×** |
> | BS=1893 | **213.36** | 217.34 | **0.982×** |
> | BS=8192 | **877.55** | 933.94 | **0.940×** |
> | 解码 BS=1 | **0.37** | 0.43 | **0.861×** |
>
> **DeepSeek-V4-Flash(0731)**:
>
> | 形状 | `xiaotu_moe` | `lk_moe` | 比值 |
> |---|---|---|---|
> | BS=227 | **20.89** | 22.18 | **0.942×** |
> | BS=1893 | 162.05 | **155.86** | 1.040× |
> | BS=8192 | **646.60** | 668.10 | **0.968×** |
> | 解码 BS=1 | 0.29 | 0.29 | 1.000× |
>
> **服务级(等参数同机 A/B,TP=2,官方 `vllm bench serve`;三项都是独立量,不折算)**
>
> | prompt/output | `lk_moe` out tok/s | 我们 out tok/s | 比值 | `lk_moe` TTFT | 我们 TTFT | `lk_moe` TPOT | 我们 TPOT |
> |---|---|---|---|---|---|---|---|
> | 256 / 32 | 11.24 | 9.44 | **0.840×** | 2153 ms | 2391 ms | 22.37 ms | 32.19 ms |
> | 256 / 1024 | 41.08 | 31.07 | **0.756×** | 2159 ms | 2436 ms | 22.26 ms | 29.83 ms |
> | 8192 / 32 | 0.49 | 0.44 | **0.898×** | 64259 ms | 71242 ms | 22.03 ms | 29.13 ms |
> | 8192 / 1024 | 11.76 | 10.53 | **0.895×** | 64208 ms | 67734 ms | 22.33 ms | 28.82 ms |
>
> 服务级仍慢 10-24%(上一版是 0.70-0.82×,本轮收窄 8-15%)。`output tok/s` **包含 TTFT**,
> 所以 `8192/32` 那格很低(TTFT 占 98.9%);TTFT 与 TPOT(**不含 TTFT**)各自独立公布。
> 完整口径见 [`../RELEASE_NOTES_v0.2.1.md`](../RELEASE_NOTES_v0.2.1.md) §2.3。

> ### ✅ v0.2 数据(2026-09-17)
>
> **DeepSeek-V4.1-Flash(748B)** 已支持并验收。两组关键数字:
>
> **(a) GPU 预填充 vs CPU 预填充**(客户端 TTFT,唯一 prompt)
>
> | prompt | CPU | **GPU** | 加速 |
> |---|---|---|---|
> | ~3.7K | 24.17 s(151 tok/s) | **12.11 s(302 tok/s)** | **2.0x** |
> | ~7.0K | 42.17 s(165 tok/s) | **15.19 s(460 tok/s)** | **2.8x** |
>
> 成本模型:**每 chunk 约 8.9 s 固定 + 0.79 ms/token**(固定项 = 每 chunk 搬 143.6 GiB/rank 专家权重)。
>
> **(b) 官方 `vllm bench serve`**(TP=2 / MBT=8192 / GPU 预填充 / KV 封顶 4 GiB / 关前缀缓存)
>
> | prompt | TTFT C=1 | TTFT C=8 | 总吞吐 C=1 | 总吞吐 C=8 | TPOT C=1 |
> |---|---|---|---|---|---|
> | 32 | 0.47 s | 3.5 s | 37 | 95 | 36.2 ms |
> | 256 | 1.98 s | 10.9 s | 55 | 134 | 43.6 ms |
> | 1024 | 10.0 s | 23.6 s | 65 | 242 | 64.6 ms |
> | 4096 | 12.8 s | 46.4 s | 200 | 379 | 66.3 ms |
> | 16384 | 34.5 s | 170 s | 399 | 455 | 54.4 ms |
> | 32768 | 70.9 s | 333 s | 420 | 454 | 58.4 ms |
>
> 原始 json:`内部调优记录 logs/bench_serve_acc2/`;完整表:`内部基准参考 BENCH_REFERENCE.md` 第 7 节。
> V4.1 全链路验收(1M 上下文 / 内存峰值 / 确定性):`RELEASE_NOTES_v0.2.md` 第 1 节。
>
> 下文为**历史数据**(V4-Flash 等),未改动。

---

## 1. 硬件

| 项 | 配置 |
|---|---|
| GPU | 2 × NVIDIA A100-PCIE-40GB(SM 8.0,无 NVLink,PCIe Gen4 ×16) |
| CPU | AMD EPYC 9654:2 socket × 96 核,4 NUMA 域(NPS4) |
| 内存 | 1.5 TiB DDR5-4800(24 × 64 GiB),理论带宽 921.6 GB/s |
| 实测内存带宽 | 单流读 **860.5 GB/s**(93% 理论);Triad + NT 写 **743.6 GB/s** |
| ISA | AVX-512F/BW/VL/DQ、AVX-512 VNNI、AVX-512 BF16;**无 AMX** |
| 软件 | Python 3.12 · torch 2.13.0+cu130 · vLLM 主线 `6c73b08` |

> 编译参数(GCC 11.4,`-march=native` 与否)对结果影响 < 1%。

## 2. 正确性

| 检查 | 结果 |
|---|---|
| 引擎 vs torch 参考(MXFP4,真实 DeepSeek-V4 层权重) | RMS 相对误差 **5.6e-3** |
| 引擎 vs numpy 参考(FP8 block-128,真实 GLM-5.3 专家权重;层 10/11/15/20/45) | RMS 相对误差 **3.6e-5 ~ 3.1e-4**(门限 1e-3) |
| clamped SwiGLU(BF16 路径,5 组 limit/alpha/beta) | 相对 RMS ≤ 1.7e-3 |
| clamped SwiGLU(MXFP4,真实 DeepSeek-V4 权重) | 最大相对误差 ≤ 6.8e-4 |
| **CPU 专家 vs GPU 专家端到端**(微型 Mixtral,bf16) | greedy token **48/48 完全一致**;prompt-logprob 最大偏差 0.053(中位 0.016) |

`scripts/tiny_moe_equiv.py` 给出 CPU/GPU 两条路径在同一 prompt 上的前向对照
(与采样噪声无关);`scripts/probe_oracle.py` 验证后端选择。

## 3. 长 prefill:CPU vs GPU 流式(DeepSeek-V4-Flash)

| prompt 长度 | 纯 CPU prefill | 单卡 GPU 流式 | TP=2 + 专家分片 |
|---|---|---|---|
| 2 091 tok | 112.9 s | **5.64 s** | – |
| 4 137 tok | 439.4 s | **5.76 s** | 3.99 s |
| 8 229 tok | – | 8.64 s | – |
| 16 039 tok | – | 18.24 s(≈880 tok/s) | **13.96 s(≈1 149 tok/s)** |

- GPU 流式在 ≤4K token 时耗时基本不变(≈5.6 s),之后近似线性;
- 纯 MoE 吞吐(不计注意力/采样):1 卡 16K 上下文 **1 918 tok/s**、
  32K **1 994 tok/s**;2 卡 **3 803 / 3 979 tok/s**;
- 16K prefill 的内核级分解:MoE 31% · elementwise 29% · H2D 19% ·
  dense GEMM 18% · attention + indexer 2.9%。

### 阈值选择

| 权重副本 | CPU/GPU 交叉点 | 建议阈值 |
|---|---|---|
| 单份(`| ≈249 token | **384**(默认) |
| 双份(默认不设) | ≈550 token | 512–768 |

## 4. CPU 引擎吞吐

| 格式 | 形状 | 吞吐 |
|---|---|---|
| MXFP4 | 真实路由,每专家约 48 行 | **1.73 / 1.96 / 2.00 TFLOP/s**(B=512/2048/8192)≈ AVX-512 BF16 峰值的 15–18% |
| FP8 block-128 | E=128,H=2048,I=768,top-8 | **0.73 / 0.91 / 0.98 TFLOP/s**(B=512/2048/8192;gather-free 解码后提升 ~2.3×) |

小批量(decode)的每层耗时(同一形状,单层 MoE):

| batch | 每层耗时 | 说明 |
|---|---|---|
| B=1 | **0.745 ms**(优化前 7.1 ms) | 小批量 N-切片 + worker 子集 |
| B=4 / 16 / 64 | 1.8 / 5.5 / 12.2 ms | 同上 |
| B=256 / 1024 | 29.4 / 89.9 ms | 走批量分组路径 |

目标模型形状(E=256,H=2560,I=640,top-10):B=1 **0.826 ms/层**,B=4/16/64/256
为 2.4 / 12.6 / 23.2 / 42.6 ms(同一脚本,`bench_fp8_engine.py 256 2560 640 10 …`)。

- MXFP4 路径 **99.7% 的时间**在两个 GEMM 相位(gate/up 57%、down 28%);
- 权重流量约 10 GB/s,远低于机器可提供的 740 GB/s → **不是带宽受限**;
- 每 token 每层约 0.15–0.17 ms,且随 batch 增大而改善。

复现:`python scripts/bench_cpu_engine.py` / `python scripts/bench_fp8_engine.py`。

> FP8 内核的瓶颈曾是**每 8 字节一次 LUT gather**(微码指令,吞吐极低)。
> 改为**无 gather 的位运算解码**(AVX-512 16-wide / AVX2 8-wide)后,
> 引擎吞吐 0.35–0.43 → **0.73–0.98 TFLOP/s**。
>
> decode 的瓶颈随后变成**调度**:B=1 时每个 token 只有 top-k 个 (token, rank)
> 指派,老的逐 token 循环只用一个线程,单层 7.1 ms。改为**把单个 token 的 GEMV
> 按 N 行切片铺满线程池**(`split_range_n`),并按实测的池屏障代价
> (~1.8 µs/worker)把参与 worker 数收敛到 30 左右(其余 worker 停靠而非自旋),
> 单层降到 **0.745 ms(9.5×)**;同一份输入在新旧两条路径上**逐位一致**。
> 剩余瓶颈是内核里 e4m3→fp32 的位运算解码(约 1.7 MAC/cycle/线程,峰值 16),
> 下一步是**权重预转 bf16 镜像**(fp8 值在 bf16 中精确可表示,数值不变)。

## 4.0 INT4(WNA16)端到端(单卡 A100)

GPTQ 检查点在引擎构造时一次性重排(`w13 [E, K/8, 2I] int32` → `[E, 2I, K/2]` u8,
缩放 `[E, K/g, N]` → `[E, N, K/g]`,`groupN=1 / groupK=128`)。

> ⏸️ **暂不声明支持**:该模型的实测数据早于 v0.2 的改动(执行模型 / 小 batch 路径 / EP 存储分片),**未在当前代码上复验**。复验计划见 `内部交接 HANDOFF_v0.2pre.md` §5.1。

## 4.1 FP8 端到端 — GLM-5.3-Flash(2×A100-40GB,TP=2,SM80)

真实 `zai-org/GLM-5.3-Flash` **原生 FP8(block 128×128)** 检查点:专家常驻主机
(283.5 GiB,**单份 NUMA 分片**),GPU 只放非专家权重 + bf16 KV。
启动见 `scripts/serve_glm53_mainline.sh`;**必须 `--kv-cache-dtype bfloat16`**
(SM8x 稀疏 MLA 后端只支持 bf16 KV;fp8/fp4 会 fail-closed)。

**交付配置(v0.2.2 默认)**:`--max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 8192
--kv-cache-dtype bfloat16 --gpu-memory-utilization 0.85` + GPU 预填充阈值 1500。
(util 0.90 → 0.85 是 §601 崩溃修复的一部分:GPU 预填充的 staging 是进程级持久的
7.59 GiB/rank,必须从 KV 里让出来;KV 池只从 988,081 掉到 **971,949** token。)

`vllm bench serve`,随机数据、`--ignore-eos`,**每格不同 seed**(避免 prefix-cache 假性命中)。
口径:**`out tok/s` 含 TTFT;三项独立公布**。

| 并发 | prompt / output | out tok/s(含 TTFT) | TTFT 均值 | TPOT 均值 | 完成 |
|---|---|---|---|---|---|
| C=1 | 256 / 128 | **16.18** | 2022 ms | **46.37 ms** | 8/8 |
| C=2 | 256 / 128 | 19.36 | 3287 ms | 78.13 ms | 8/8 |
| **C=1** | **4096 / 64** | 2.49 | **22764 ms** | 46.95 ms | 2/2 |
| **C=1**(util 0.85 复测,2026-09-19) | **4096 / 64** | 2.50 | **22650 ms** | 46.50 ms | 2/2 |
| C=2 | 4096 / 64 | 2.59 | 34162 ms | 243.21 ms | 4/4 |

**长上下文崩溃复现(§601 验收)**:同一服务,util 0.90 时 **32,077-token** 请求崩掉进程;
util 0.85 + `expandable_segments` + 新预检口径下,同量级请求 **TTFT 108.4 s / 无 OOM**,
两路并发 14,084 + 15,423 token 也全部通过(预检余量 `slack +3.85 GiB`)。

**与 v0.2.1 的配置(4096 上下文 / MBT=1024 / 阈值 4096 ⇒ 全 CPU 预填充)对照**:

| 口径 | v0.2.1 配置 | **v0.2.2 交付配置** |
|---|---|---|
| 4096-in / 64-out C=1 TTFT | 29320 ms | **22764 ms(1.29×)** |
| 256-in / 128-out C=1 out tok/s | 16.20 | 16.18(**解码不变**) |
| 256-in / 128-out C=1 TPOT | 45.03 ms | 46.37 ms |
| 上下文×并发 | 4096 × 4 | **262144 × 2** |

* C=4 那一档不再公布:交付配置把并发限到 2(`--max-num-seqs 2`),换成"两路 **256K**"的
  上下文保证 —— 两路 256K 需 524,288 token,而池子是 **971,949 token**;
* **4096/64 的 C=2 TPOT 243 ms** 是 chunked prefill 的正常代价(长 prompt 的 chunk 与
  decode 步交错);长 prompt 场景建议"一路长 + 一路短"。

读法:

* **纯解码 ≈ 22 tok/s**(C=1,`1/TPOT`),且 **TPOT 与上下文长度基本无关**
  (4096-token 的 TPOT 仍 46.4 ms);
* 吞吐在 C≥2 就饱和、而 TPOT 反而升高 ⇒ 瓶颈是**每层引擎调用的延迟**,不是带宽;
* **TTFT 由预填充主导**(4096 token:全 CPU 为 29.3 s,交付的 GPU 预填充为 **22.8 s**)。

**GPU 流式预填充(FP8)已接线并实测**(`gpu_prefill_fp8.py`,按引擎类别自动选择,
无需开关)。同进程内切阈值、唯一 prompt(`scripts/probe_ttft.py`,UNIQUE=1):

| 口径(3860-token prompt,切 2176 + 1680) | TTFT |
|---|---|
| 纯 CPU 预填充 | 26.60 s |
| GPU 预填充 + **串行装配** | 26.63 s |
| GPU 预填充 + **装配与 attention 重叠(side stream,默认)** | **22.0-22.2 s** |
| 目标基线(CPU,MBT=1024 四段) | 29.3 s |

⇒ **端到端 29.3 s → 22.0 s(1.33×)**。装配是 3.62 GB/rank 的纯 H2D(M 无关的固定成本),
在空闲 GPU 上实测 134.9 ms = **26.86 GB/s**,即本机 H2D 天花板(1-D 与 pitched 2-D 同速,
`numactl --interleave` 与两 rank 并发都不影响)。所以优化点不是"搬得更快",而是
**不再让搬运排在 attention 后面**:DMA 只碰暂存缓冲与主机分片,attention 不读它们。
服务里装配实测 164-204 ms(高于隔离值),差额是与 vLLM 自己每层 PCIe 流量
(TP=2 的 all-reduce;GPU0/GPU1 之间无 NVLink)的争用,属于既有开销。

每层分项:`asm=207 ms`(权重 H2D 3.62 GB/rank)+ `kernels=23 ms`(GPU MoE,
比 CPU 引擎快约 10×)。

**为什么只有 5%**:GLM-5.3 的 KDA(mamba-like)state 只在 **block_size = 2176**
边界上写,调度器因此把预填充 chunk 钉死在 2176 token
(`v1/core/sched/scheduler.py` 的 `aligned_end = end // block_size * block_size`),
而 GPU 路径的每层权重 DMA 是**固定成本**,盈亏平衡点约 **1.9k token/chunk**。
⇒ 同一个后端在没有 state 分页的模型上(DeepSeek-V4.1:§3 表 ~7K chunk)是 **2.8×**;
GLM-5.3 想拿到更多,需要把装配与 attention 重叠(目标:207 → 135 ms 的 1D DMA 地板)。

正确性:GPU 装配**逐字节相同**(`scripts/test_gpu_prefill_fp8_assembly.py`)、
层输出 vs CPU 引擎 rms_rel **4.43e-3**(`test_gpu_prefill_fp8_vs_cpu.py`)、
真实服务 4835-token needle 检索**完全命中**。


稳定性/数值:贪心同一 prompt 三次**逐字节相同**;连续 24 请求 **24/24 成功**;
层内真实权重 RMS 相对误差 **3.6e-5 ~ 3.1e-4**(见 `MODEL_GUIDES.md` §2)。

## 4.2 使用指南与其它模型

DeepSeek-V4-Flash 的**逐步使用指南、完整初步性能表
(含数据来源与复现命令)**见 [`MODEL_GUIDES.md`](MODEL_GUIDES.md)。

## 4.3 GLM-5.3-Flash 的 MTP 投机解码(2026-09-20,实测)

配置:2×A100-40GB / TP=2 / `GPU_UTIL=0.82` / GPU 流式预填充开;官方 `vllm bench serve`;
`SPEC_K` 控制 `num_speculative_tokens`。draft = 检查点自带的第 45 层(3.38 GiB/rank 常驻 GPU)。

**接受长度**

| k | accept(random) | accept(ShareGPT) | 来源 |
|---|---|---|---|
| **1(默认)** | **1.469** | **1.459** | N=16/1327 drafts,直接实测 |
| 2 | — | 1.539 | † 由 k=4 运行的逐位接受率反推 |
| 3 | — | 1.595 | † |
| 4 | 1.619 | 1.626 | † (那次 p0=0.400) |

† k=2/3/4 由一次**独立的 k=4 运行**的 `num_accepted_tokens_per_pos` 反推(逐位:p0≈0.40、
p1≈0.14、p2≈0.06、p3≈0.03);与直接实测的 k=1 = 1.459 有 run-to-run 差异,**不要混着比**。

**端到端**:同机同参 A/B(`SPEC_K=0` vs `1`,256/128 C=1 N=8,各重复 2 次;ShareGPT N=16)

| 工作负载 | 不开 MTP | MTP k=1 | MTP k=4 |
|---|---|---|---|
| random 256/128 C=1 | 16.14 / 16.13 tok/s,TPOT **45.76 / 45.72 ms**,ITL 46.2 / 46.6 ms | 16.65 / 16.12 tok/s,TPOT **43.33 / 45.08 ms**,ITL 64.4 / 64.2 ms | (单请求)10.19 tok/s,TPOT 80.82 ms,ITL 129.9 ms |
| ShareGPT C=1 N=16/128 | 17.08 tok/s,TPOT 45.58 ms | 16.93 tok/s,TPOT **44.82 ms**,ITL 64.4 ms | 11.24 tok/s,TPOT 75.35 ms |
| **KV 池** | **915,487 token**(256K 并发 3.49×) | **666,366 token**(2.54×) | 508,519 |

**结论**:GLM 的单层回收 MTP **k=1 的 TPOT 小赚(−3%,噪声内),真正的代价是 KV −27%**;
**k>1 明确更差**(accept 只涨到 1.63,但每个 step 要验证 T=1+k 个 token ⇒ CPU 专家路径的
不同专家数近乎翻倍)。因此默认 `SPEC_K=1`(**按用户要求开**),`SPEC_K=0` 可关、`>1` 不要开。
同族对照:MiMo 的 k=1 因为 p0 高得多(0.83 vs GLM 0.46),收益是明确的 −9.4%(见 §4.4)。
数据与方法见 [`MODEL_GUIDES.md`](MODEL_GUIDES.md) §2.6。

> 复现:`SPEC_K=0|1 GPU_UTIL=0.82 bash scripts/serve_glm53_mainline.sh`,
> 接受长度取 `/metrics` 的 `vllm:spec_decode_num_accepted_tokens_per_pos_total`。

## 4.4 MiMo-V2.5 MTP(2026-09-20,单卡 A100-40GB / TP=1 / 专家在 CPU)

模型:`XiaomiMiMo/MiMo-V2.5`(310B / 15B active,48 层,256 专家 top-8,FP8 block-128);
`--language-model-only --trust-remote-code`,`--gpu-memory-utilization 0.85`,
GPU 流式预填充关(`GP_ACT_RESERVE_GIB=99`)。同一个服务进程分别跑 MTP 开/关。

| 工作负载(官方 `vllm bench serve`) | 不开 MTP | MTP k=1 | 不开 vs k=1 |
|---|---|---|---|
| **256 in / 128 out / C=1,N=8(×2)** | 10.88/10.93 tok/s,TPOT **64.56/64.00 ms**,ITL 64.1/63.5 ms | 11.54/11.59 tok/s,TPOT **58.02/58.48 ms**,ITL 99.9/99.9 ms | TPOT **−9.4%**(均值 64.28→58.25) |
| 2000 in / 64 out / C=1,N=1 | 2.31 tok/s,TTFT 23.6 s,TPOT 66.01 ms | 2.27 tok/s,TTFT 24.5 s,TPOT 58.48 ms | TPOT −11.4%(单请求) |
| 4000 in / 64 out / C=1,N=1 | 1.25 tok/s,TTFT 46.8 s,TPOT 68.31 ms | 1.22 tok/s,TTFT 47.1 s,TPOT 84.70 ms | TPOT +24%(单请求,噪声) |

* **接受长度**:256/128 N=8 的稳定口径 = **1.743**(accepted 869 / drafts 1169 = 74.3%);
  早期小样本给到 1.83~1.87。**k=3 = 1.016(p0 = 0.016 ⇒ 坏)**;
* 同 prompt 128-token 直接对拍:不开 **8.72 s** → k=1 **7.33 s(1.19×)**,且**贪心输出逐字节相同**;
* ⚠️ 只有第一行是**重复口径**(N=8 ×2,均值抖动 <1%);后两行是单请求,TPOT 抖动 ~10%,
  4000-token 那格方向相反多半是噪声,**要报数请补重复**;
* **ITL 被脉冲化**(63.6 → 99.9 ms)是 MTP 的固有权衡;k=1 每步只多吐 1 个 token,冲击比 GLM k=4 小;
* 结论:**MiMo 用 `num_speculative_tokens=1`;不要用 k>1**。方法/复现见
  [`MODEL_GUIDES.md`](MODEL_GUIDES.md) §3。

## 5. 服务端吞吐与时延(DeepSeek-V4-Flash,单卡)

| 指标 | 数值 |
|---|---|
| decode 吞吐(B=64 / 128,短上下文) | 74.6 / 80.3 tok/s |
| KV cache 占用 | ≈400 KiB/token |
| 启动期 pinned 缓存构建 | 27 s(优化前 114 s) |
| 并发(2K 上下文) | 聚合吞吐随并发近似线性到 1.93× |

> 吞吐瓶颈是 CPU MoE 引擎(占每步 80% 以上),不是注意力。

## 6. 复现方式

```bash
python scripts/bench_cpu_engine.py            # CPU 引擎(真实路由)
python scripts/bench_fp8_engine.py            # FP8 引擎
python scripts/bench_llm.py                   # 端到端 decode 吞吐
python scripts/server_concurrency_test.py     # 并发
python scripts/probe_oracle.py                # 后端选择
python scripts/tiny_moe_equiv.py              # CPU/GPU 专家等价性
python report/make_figs.py                    # 由 report/*.json 生成图表
```
