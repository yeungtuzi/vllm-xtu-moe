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
| FP8 block-128 | E=128,H=2048,I=768,top-8 | **0.35–0.43 TFLOP/s**(待优化) |

- MXFP4 路径 **99.7% 的时间**在两个 GEMM 相位(gate/up 57%、down 28%);
- 权重流量约 10 GB/s,远低于机器可提供的 740 GB/s → **不是带宽受限**;
- 每 token 每层约 0.15–0.17 ms,且随 batch 增大而改善。

复现:`python scripts/bench_cpu_engine.py` / `python scripts/bench_fp8_engine.py`。

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
