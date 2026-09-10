# 模型使用指南与初步性能

本文给出两个**已在 A100(SM 8.0,无 AMX)上跑通**的模型的完整使用步骤与**初步**性能数据:

| 模型 | 专家格式 | 硬件 | 一句话结论 |
|---|---|---|---|
| **DeepSeek-V4-Flash**(0731) | MXFP4(4-bit + e8m0 块缩放) | 1 × A100-40GB(可 TP=2) | 单卡可跑;长 prefill 走 GPU 流式,16K ≈ 880 tok/s;批量 decode 74–80 tok/s |
| **Qwen3.8-Flash-Next-FP8** | FP8 e4m3 block-128 | 2 × A100-40GB(必须 TP=2 + 部分 offload) | 能跑通且答案正确;decode 受 `--cpu-offload-gb` 搬运带宽限制,不在 MoE 引擎 |

> **"初步"的含义**:全部为**单机、单/双卡、离线少量请求**的实测,用于判断"能不能跑、量级多少"。
> 共享机器上有其他负载(每张表都给出数据来源与复现命令),**不是调优后的上限**。
> 安装主线 vLLM 与本插件见 [`RUNBOOK.md`](RUNBOOK.md) §1–2;硬件基线见
> [`BENCHMARKS.md`](BENCHMARKS.md) §1。

---

## 0. 共同前置

### 0.1 开启混合模式(CPU 专家 + GPU 其余)

```bash
export VLLM_EXPERTS_LOAD_DEVICE=cpu   # ★ 混合模式开关(取值只有 cpu / gpu)
export XIAOTU_MOE_SINGLECOPY=1        # 权重只保留一份(NUMA 分片),省一半内存
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
| KV | DeepSeek-V4-Flash ≈ **400 KiB/token**;Qwen3.8-Flash-Next 每 rank 实测 **20.27 GiB**(maxlen 4096) |

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

## 2. Qwen3.8-Flash-Next-FP8

### 2.1 模型与资源需求(为什么必须 TP=2 + offload)

| 项 | 值 |
|---|---|
| 架构 | `Qwen4ExpForConditionalGeneration`,**48 层**,512 experts / **top-10**,H=2560,I=640,fp8 e4m3 block-128 |
| 权重构成 | 专家 ≈120 GB(fp8)+ 非专家 ≈65 GB,其中 **PLE n-gram 嵌入表 ≈51 GB** |
| 权重体积 | 约 **173 GiB**(本机下载实测;模型卡标注 185 GB) |
| 显存 | 单卡 40 GB 放不下非专家权重 → 报 `No available memory for the cache blocks`;必须 **TP=2 + `--cpu-offload-gb`**;实测每 rank 非专家 6.25 GiB + KV 20.27 GiB |
| CPU 内存 | 每 rank 约 60 GB 专家权重 + 引擎快照 ⇒ 合计约 240 GB |

### 2.2 启动

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ENGINE_READY_TIMEOUT_S=7200      # 加载实测 1500–1600 s

vllm serve Qwen/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --cpu-offload-gb 12 \
  --max-model-len 4096 --max-num-seqs 2 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --kernel-config.enable_jit_warmup=false
```

离线冒烟脚本(自带计时与连贯性检查,推荐先用它验收):

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_MOE_SINGLECOPY=1 \
  VLLM_ENGINE_READY_TIMEOUT_S=7200 TP=2 EP=1 CPU_OFFLOAD_GB=12 \
  SMOKE_MODEL=Qwen/Qwen3.8-Flash-Next-FP8 SMOKE_LONG_TOKENS=512 \
  python scripts/fp8_moe_smoke.py
```

> 若多模态处理器报错,加 `--limit-mm-per-prompt '{"image":0,"video":0}'` 只跑文本。

### 2.3 自检与验收

启动日志中确认三件事:

```
Using CPU Fp8 MoE backend ...                                   # 后端被选中
[vllm-xtu-moe] expert parallelism: local=256 global=512         # EP 生效
[vllm-xtu-moe] xiaotu MOE_FP8 engine: E=256 H=2560 I=640 topk=10 group=128x128
```

冒烟脚本应输出 4 个正确回答(北京 / 2 / MoE 解释 / 长文补全),并打印 TTFT 与 decode。
逐层数值可用 `XIAOTU_VERIFY_LAYER=1 XIAOTU_VERIFY_MAX=48` 打开:该自校验在最接近本模型的
fp8 权重上已跑过(48 层 × 2304 次调用,rel_rms **1.9e-7 … 3.9e-4**,差异来自 GPU 侧动态量化
激活 vs CPU 侧 bf16 激活,见 `docs/KNOWN_LIMITATIONS.md` §3);本模型本身只做了答案级验收。

### 2.4 初步性能(2 × A100-40GB,TP=2 + EP + offload 12 GB)

| 项 | 首次 | 复测(FP8 解码优化后) |
|---|---|---|
| 权重加载 | 1 583.8 s | **1 513.6 s** |
| 3 个短问答 | 38.6 s | **24.2 s** |
| 531-token prefill | 12.21 s(44 tok/s) | **8.66 s(61 tok/s)** |
| decode(500 token 级长上下文) | ≈1.3 tok/s | ≈**0.7 tok/s** |
| 答案正确性 | 北京 / 2 / MoE 解释 / 长文补全 | 同左 |

复现即 §2.2 的 `scripts/fp8_moe_smoke.py`(脚本自己打印 `LOAD_OK` / `TTFT` / `decode`),
汇总见 [`BENCHMARKS.md`](BENCHMARKS.md) §4.1。

**为什么 decode 没随内核提速**——这是本模型最需要知道的一点:

| 环节 | 实测 |
|---|---|
| 引擎在该形状(`E=256(local)/H=2560/I=640/top-10`)的耗时 | B=1 **0.826 ms/层**(B=4/16/64/256 → 2.4/12.6/23.2/42.6 ms) |
| 端到端 decode | ≈28 ms/层 |
| 结论 | **引擎只占 ≈3%**;其余是 `--cpu-offload-gb 12` 每步把 offload 权重搬回 GPU(≈12 GB/步,PCIe 无 NVLink)+ 逐层 dense/超连接算子 |

⇒ 对"权重放不下、必须 offload"的模型,**继续优化 MoE 内核几乎没有收益**;
要提速应减少 offload 量(加显存 / 加卡)或换更大的显存。

### 2.5 已知限制

| 项 | 说明 |
|---|---|
| 单卡不可用 | 非专家权重 + KV 超出 40 GB;这是**显存**限制,与本插件无关 |
| offload 成本 | `--cpu-offload-gb` 越大越省显存、decode 越慢,建议在"能起 KV"的前提下取最小值 |
| decode 慢 | 见 §2.4,瓶颈在 offload 带宽 |
| 加载近半小时 | 2 个 rank 各自读全部 shard + 建 48 个引擎;`XIAOTU_MOE_SINGLECOPY=1` 已是省内存配置 |

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
| Qwen3.8-Flash-Next 端到端 | `scripts/fp8_moe_smoke.py`(见 §2.2);汇总见 `BENCHMARKS.md` §4.1 |

图表由 `python report/make_figs.py` 从 `report/*.json` 重新生成,输出在 `report/fig/`。

> 免责声明:以上均为**单机初步实测**,共享机器上存在其他负载(会话记录里标注了当时的
> `load average` 与并发任务);不同批次之间的绝对值可能相差 10–30%,请以趋势与量级为准。
