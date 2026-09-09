# 实测数据

以下数字全部来自本项目的实测环境;原始数据与绘图脚本在 [`report/`](../report/),
图表在 [`report/fig/`](../report/fig/)。

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
| 引擎 vs numpy 参考(FP8 block-128,真实 GLM-5.3 专家权重) | RMS 相对误差 **9.3e-5** |
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
| 单份(`XIAOTU_MOE_SINGLECOPY=1`) | ≈249 token | **384**(默认) |
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
> 引擎吞吐 0.35–0.43 → **0.73–0.98 TFLOP/s**;真实模型(Qwen3-30B-A3B-FP8,单卡)
> prefill 75 → **113 tok/s**,decode 1.1 → **5.8 tok/s**。
>
> decode 的瓶颈随后变成**调度**:B=1 时每个 token 只有 top-k 个 (token, rank)
> 指派,老的逐 token 循环只用一个线程,单层 7.1 ms。改为**把单个 token 的 GEMV
> 按 N 行切片铺满线程池**(`split_range_n`),并按实测的池屏障代价
> (~1.8 µs/worker)把参与 worker 数收敛到 30 左右(其余 worker 停靠而非自旋),
> 单层降到 **0.745 ms(9.5×)**;同一份输入在新旧两条路径上**逐位一致**。
> 端到端(Qwen3-30B-A3B-FP8,单卡,528-token 提示)prefill 113 → **182 tok/s**,
> decode 5.8 → **11.6 tok/s**。
> 剩余瓶颈是内核里 e4m3→fp32 的位运算解码(约 1.7 MAC/cycle/线程,峰值 16),
> 下一步是**权重预转 bf16 镜像**(fp8 值在 bf16 中精确可表示,数值不变)。

## 4.1 Qwen3.8-Flash-Next-FP8 端到端(目标模型,2×A100-40GB)

| 项 | 数值 |
|---|---|
| 配置 | `VLLM_EXPERTS_LOAD_DEVICE=cpu` + TP=2 + `--enable-expert-parallel` + `--cpu-offload-gb 12` |
| 后端选择 | `Using CPU Fp8 MoE backend`;每 rank `local=256 / global=512` 专家 |
| 权重加载 | **1583.8 s**(185 GB,2 rank 各自读全部 shard) |
| GPU 侧 | 每 rank 非专家权重 6.25 GiB;KV cache 20.27 GiB |
| 短问答 | 3 个问题 38.6 s(2026-09-09 复测 24.2 s),答案正确(北京 / 2 / MoE 解释) |
| 长 prefill | 531 token:TTFT **12.21 s**(44 tok/s;复测 **8.66 s** / 61 tok/s) |
| decode | ≈**0.7–1.3 tok/s**(512 专家 top-10;**瓶颈不在本引擎**,见下) |

> 该模型非专家权重约 65 GB(其中 51 GB 是一张 PLE n-gram 嵌入表),
> 单卡 40 GB 放不下,必须 TP=2 + 部分权重 offload 到 CPU。

**为什么 decode 没随内核提速**:该模型的引擎形状是 `E=256(local)/H=2560/I=640/top-10`,
引擎实测 **0.826 ms/层**(B=1,见 §4),而端到端 decode 实测约 **28 ms/层** —— 引擎只占 **3%**。
其余开销来自 `--cpu-offload-gb 12` 每步把 offload 的权重搬回 GPU(≈12 GB/步,PCIe 上约 0.5–1 s/token)
以及逐层 dense/超连接算子。**对这类"权重放不下、必须 offload"的模型,decode 的瓶颈是 offload 带宽,
不是 CPU 专家计算**;缓解办法是减少 offload 量(加显存/加卡)而不是优化 MoE 内核。

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
