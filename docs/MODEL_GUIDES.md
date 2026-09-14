# 模型使用指南与初步性能

本文给出**已在 A100(SM 8.0,无 AMX)上跑通**的模型的完整使用步骤与**初步**性能数据:

| 模型 | 专家格式 | 硬件 | 一句话结论 |
|---|---|---|---|
| **DeepSeek-V4-Flash**(0731) | MXFP4(4-bit + e8m0 块缩放) | 1 × A100-40GB(可 TP=2) | 单卡可跑;长 prefill 走 GPU 流式,16K ≈ 880 tok/s;批量 decode 74–80 tok/s |

> **"初步"的含义**:全部为**单机、单/双卡、离线少量请求**的实测,用于判断"能不能跑、量级多少"。
> 共享机器上有其他负载(每张表都给出数据来源与复现命令),**不是调优后的上限**。
> 安装主线 vLLM 与本插件见 [`RUNBOOK.md`](RUNBOOK.md) §1–2;硬件基线见
> [`BENCHMARKS.md`](BENCHMARKS.md) §1。

---

## 0. 共同前置

### 0.1 开启混合模式(CPU 专家 + GPU 其余)

```bash
export VLLM_EXPERTS_LOAD_DEVICE=cpu   # ★ 混合模式开关(取值只有 cpu / gpu)
export VLLM_ENGINE_READY_TIMEOUT_S=3600   # 首次加载要建 43/48 个引擎,别让默认超时打断
export VLLM_USE_FLASHINFER_SAMPLER=0  # 与 CPU 引擎的 host 回调同流,避免额外变量
```

启动日志里应能看到两行确认:

```
Using CPU Fp8 MoE backend out of potential backends: ['CPU', 'AITER', ...]   # 或 CPU Mxfp4
[vllm-xtu-moe] xiaotu MOE_FP8 engine: E=... H=... I=... topk=... group=... scales=yes
```

先做一次**不加载权重**的后端选择自检(秒级):

```bash
VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py
```

### 0.2 资源估算

| 资源 | 估算方式 |
|---|---|
| CPU 内存 | 专家权重全量 + 引擎快照一份(单份模式)⇒ 约 **2 × 专家权重**;另加 pinned 预取缓存(见 `GPU_PREFILL.md`) |
| 显存 | 非专家权重 + KV cache;`--gpu-memory-utilization` 与 `--cpu-offload-gb` 用来在两者之间挪 |
| KV | DeepSeek-V4-Flash ≈ **400 KiB/token** |

### 0.3 下载(国内镜像)

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
# 注意:若环境里设了失效的本地代理(ALL_PROXY=127.0.0.1:...),需要 env -u ALL_PROXY
huggingface-cli download <repo_id> --local-dir <dir>
```

---

## 1. DeepSeek-V4-Flash

### 1.1 模型与资源需求

| 项 | 值 |
|---|---|
| 架构 | `deepseek_v4`,**43 层**,256 routed experts / **top-6**,1 shared expert,H=4096,I=2048 |
| 专家权重 | **MXFP4**(`w1/w3 [2048, 4096]` I8 + `scale F8_E8M0`,block 32)≈ **3.19 GiB/层**,43 层合计 **137 GiB** |
| 权重目录体积 | 约 156 GB(含 dense/MLA 部分) |
| 显存 | 非专家权重 + KV;单卡 40 GB 用 `--gpu-memory-utilization 0.85` + `maxlen 8192` 足够 |
| CPU 内存 | 约 137 GiB ×2(权重 + 引擎快照)≈ 274 GiB;若启用长 prefill 的 GPU 流式,还会**逐层**构建 pinned K-major 预取缓存(见 `GPU_PREFILL.md` §2),需留额外空间 |

### 1.2 启动

**单卡(最简,验证用)**

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384   # 长 prefill 阈值;384 见 §1.4 的交叉点

vllm serve <DEEPSEEK_V4_FLASH_DIR> \
  --tensor-parallel-size 1 \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --kernel-config.enable_jit_warmup=false \
  --served-model-name DeepSeek-V4-Flash
```

**双卡(TP=2 + 专家分片)**

```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve <DIR> --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 16384 --max-num-seqs 16 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --kernel-config.enable_jit_warmup=false
```

> 纯 CPU prefill 只适合短提示:2K token 就要 ~93 s(§1.4)。长提示请让 GPU 流式路径接管
> (`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`,细节见 [`GPU_PREFILL.md`](GPU_PREFILL.md))。

离线冒烟(自带计时;`SKIP_TOK=1` 走固定 token 提示,`0` 走英文提示词模板):

```bash
CUDA_VISIBLE_DEVICES=0 TEST_MODEL=<DIR> CONCURRENCY=1 OUT_TOKENS=16 SKIP_TOK=0 \
  python scripts/bench_llm.py
```

### 1.3 自检与验收

| 检查 | 命令 | 期望 |
|---|---|---|
| 后端选择 | `VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py` | `fp8-dsv4 → CPU`(MXFP4 权重为 `mxfp4 → CPU`) |
| 输出正确性 | 按上面的 `bench_llm.py`(`SKIP_TOK=0`) | `"The capital of France is " → "Paris…"`,token 连续、无重复(真实权重下曾在 SM80 上出现 `attn_out=inf` 的 fp8/Marlin 重打包问题,已修) |
| 层内数值 | 启动前加 `XIAOTU_VERIFY_LAYER=1` | 打印 `rel_rms`,MXFP4 真实层 ≈ **5.6e-3**(bf16 输出精度) |

### 1.4 初步性能(单卡 A100-40GB)

**prefill:CPU vs GPU 流式(阈值扫描)** — 来源 `report/curve_thr.jsonl`(同进程内两条路径对照)

| prompt token | 纯 CPU TTFT | GPU 流式 TTFT | 说明 |
|---:|---:|---:|---|
| 139 | 5.69 s | 2.41 s | 交叉点之前 |
| 275 | 6.53 s | 4.96 s | ≈**250 token 交叉** |
| 411 | 9.85 s | 8.02 s | |
| 548 | 12.25 s | **5.66 s** | 之后 GPU 基本恒定 |
| 793 | 18.05 s | 5.67 s | |
| 1057 | 23.52 s | 5.65 s | |
| 2091 | **92.60 s** | 5.71 s | |

**长 prefill 与双卡** — 来源 `report/curve_tp1g.jsonl`、`report/curve_tp2.jsonl`、`BENCHMARKS.md` §3

| prompt token | 单卡 GPU 流式 | TP=2 + EP |
|---:|---:|---:|
| 2 091 | 5.64 s(371 tok/s) | – |
| 4 137 | 5.76 s(718 tok/s) | **3.99 s(1036 tok/s)** |
| 8 229 | 8.64 s(952 tok/s) | – |
| 16 039 | 17.74–18.24 s(≈900 tok/s) | **13.96 s(1149 tok/s)** |

- GPU 流式在 ≤4K 时耗时几乎不变(≈5.6 s)——**H2D 地板**(137 GiB / 25 GB/s ≈ 5.9 s),与 batch 内 token 数无关;
- 双 slot 预取把 H2D 藏进计算:单层 T=4K/16K/32K 从 217.6/356.1/539.0 ms 降到 **127.4/198.7/382.2 ms**;
- 引擎侧 MXFP4 吞吐(真实路由):**1.73 / 1.96 / 2.00 TFLOP/s**(B=512/2048/8192);
- 纯 MoE 微基准(43 层、top-6,H=4096/I=2048,来源 `report/moe_micro.json`):

| 上下文 | 单卡 | TP=2 |
|---:|---:|---:|
| 2 048 | 373 tok/s | 745 tok/s |
| 8 192 | 1 495 | 1 875 |
| 16 384 | **1 918** | **1 950** |
| 32 768 | 1 994 | 1 996 |
| 49 152 | 2 013 | 2 007 |

**decode(批量,短上下文)** — 复现:`GLM_MAXLEN=256 OUT_TOKENS=32 CONCURRENCY=<B> python scripts/bench_llm.py`

| batch | 输出吞吐 |
|---:|---:|
| 1 | 6.9 tok/s |
| 32 | 70.8 tok/s |
| 64 | **74.6 tok/s** |
| 128 | **80.3 tok/s** |
| 256 | 77.6 tok/s |

- 单序列长上下文 decode 明显更低(2K 上下文 64 token / 12.1 s ≈ **5.3 tok/s**)——CPU 专家是逐 token 全量读权重;
- 埋点(`XIAOTU_TIMING=1`)显示每步 **CPU MoE 占 ≈90%**,GPU 注意力只占 8%;
- TP=2 下单序列 decode 反而变慢(两 rank 各自算全量专家,见 §1.5)。

**并发(prefill 聚合,2K 上下文)** — 来源 `report/server_conc.jsonl`(真实 HTTP 服务端)

| 并发 | 聚合吞吐 | 相对 c=1 |
|---:|---:|---:|
| 1 | 356.5 tok/s | 1.00× |
| 2 | 1 298.5 | 3.64× |
| 4 | 1 998.4 | 5.61× |
| 8 | 2 002.9 | 5.62× |

> 已定位的上限:请求到达时序让 vLLM 把"第一个请求独占一个 step、其余合批一个 step"
> (埋点 `XIAOTU_DEBUG_QLEN=1` 可见每个 step 的 token 数),所以并发只能摊掉 H2D 地板,
> 突破不了"一次大 prefill"的吞吐(16K ≈ 900 tok/s)。

### 1.5 已知限制与调参建议

| 现象 | 建议 |
|---|---|
| 短提示也走 GPU 流式,反而更慢 | 交叉点 ≈250 token(单份权重)/ ≈550 token(双份),阈值 **384** 是安全默认 |
| decode 慢 | CPU MoE 是瓶颈(占每步 90%):提高 batch 到 64–128 收益最大;单序列长上下文收益有限 |
| TP=2 decode 反而慢 | 两个 rank 各自算全量专家(EP 只分权重、未分计算)—— 已知项:decode 优先用单卡,TP=2 只在长 prefill 上赚(16K 1.31×) |
| 首次加载 5–16 分钟 | 43 层 × 建引擎 + pinned 预取缓存(单卡加载实测 321 s,TP=2 942 s) |

---

## 2. 目标 FP8 模型(暂不声明支持)

> ⏸️ **暂不声明支持**:该模型的实测数据早于 v0.2 的改动(执行模型 / 小 batch 路径 / EP 存储分片),**未在当前代码上复验**。复验计划见 `docs/HANDOFF_v0.2.md` §5.1。

---

## 3. 数据来源与复现

| 数据 | 来源 / 复现命令 |
|---|---|
| DS-V4 阈值扫描(CPU vs GPU) | `report/curve_thr.jsonl`;`scripts/dsv4_prefill_curve.py`(env:`MODES=cpu,gpu LENS=…`) |
| DS-V4 单卡 / 双卡长 prefill | `report/curve_tp1g.jsonl`、`report/curve_tp2.jsonl` |
| DS-V4 纯 MoE 微基准 | `report/moe_micro.json`;`scripts/bench_gpu_moe_prefetch.py` |
| DS-V4 批量 decode | `scripts/bench_llm.py`(`CONCURRENCY=1/32/64/128/256`,`OUT_TOKENS=32`) |
| DS-V4 服务端并发 | `report/server_conc.jsonl`;`scripts/server_concurrency_test.py` |
| 引擎 MXFP4 / FP8 吞吐 | `scripts/bench_cpu_engine.py`、`scripts/bench_fp8_engine.py` |

图表由 `python report/make_figs.py` 从 `report/*.json` 重新生成,输出在 `report/fig/`。

> 免责声明:以上均为**单机初步实测**,共享机器上存在其他负载(会话记录里标注了当时的
> `load average` 与并发任务);不同批次之间的绝对值可能相差 10–30%,请以趋势与量级为准。
