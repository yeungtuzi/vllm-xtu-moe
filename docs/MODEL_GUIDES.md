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

> ⚠️ **`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` 的「0 = 关闭」陷阱(2026-09-22 实测,B120)**
> 判定在 `mixed_experts.py`:`_gp_min = gpu_prefill_min_tokens()`、**`_gp_on = _gp_min > 0`**
> ⇒ **门槛为 0(或干脆不设)表示「关闭 GPU 预填」**,而不是"永不拒绝"(与 `gpu_prefill.py` docstring 的字面读法相反)。
> 不设它 ⇒ 长 prefill 全走 CPU ⇒ **只有 ~310 tok/s,而不是 ~900**(MiMo 曾长期如此,见 `EXPERIMENTS.md` B120)。
> **⇒ 接入任何新模型时必须显式给正数**(V4.1/GLM 用 4096 或策略算出的值)。
>
> **验证方式(唯一直接证据)**:日志里必须出现
> `[vllm-xtu-moe] GPU prefill ACTIVE: first <N> tokens >= threshold <T>`。
> **⇒ 测 prefill 性能之前先 grep 这一行**;没有它,得到的数字是 **CPU 口径**,不能与 GPU 口径的数字放在同一张表里比较。

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
> (`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`,细节见 `GPU_PREFILL.md`)。

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

## 2. GLM-5.3-Flash(A100 / SM80 端到端已支持)

### 2.1 与前代不同的地方

* **45 层**语言塔:0–2 dense、3–44 稀疏 MoE、45 为 MTP;注意力是 **KDA 线性注意力 34 层
  + NoPE 稀疏 MLA(DSA)11 层**(层号 3/7/…/43);
* **288 路由专家 / top-8 + 1 共享专家**,`sigmoid + noaux_tc`,`routed_scaling_factor=2.5`,
  `swiglu_limit=10`;
* 检查点是**原生 FP8(e4m3,block 128×128)**:专家 ~**283.5 GiB**;`qk_rope_head_dim=0`(NoPE)
  使 `qk_nope_head_dim=256 / v_head_dim=256 / kv_lora_rank=512`。

### 2.2 启动

```bash
bash scripts/serve_glm53_mainline.sh        # 默认 TP=2、GPU 0/1、bf16 KV
# 可覆盖:TAG PORT MAXLEN MBT SEQS THREADS COMPILE=1(smoke)→ python scripts/glm53_smoke.py --port <PORT>
```

> ⚠️ **必须 `--kv-cache-dtype bfloat16`**:本插件的 SM8x(Ampere/Ada)稀疏 MLA 后端
> **只支持 bf16 KV**;fp8/fp4 KV 会 fail-closed(回到上游 SM90+ 候选池并报错)。脚本已默认设好。

### 2.3 实测(2×A100-40GB,TP=2,真实 FP8 检查点,`vllm bench serve`,随机数据 + `--ignore-eos`)

**交付配置**(脚本默认):`--max-model-len 262144 --max-num-seqs 2 --max-num-batched-tokens 8192
--kv-cache-dtype bfloat16 --gpu-memory-utilization 0.85`,GPU 预填充阈值 1500。
> ⚠️ util 从 **0.90 降到 0.85**(2026-09-19,§601 事故):GPU 预填充的 FP8 staging 是
> **进程级持久**的 ~7.59 GiB/rank,它必须从 KV 里让出来,否则长 prefill 的 attention
> 工作区会把服务 OOM 打崩(实测 util 0.90 下 28,553-token 请求崩掉整个进程)。
> 代价极小:KV 池 988,081 → **971,949** token(-1.6%,两路 256K 仍只占 54%),
> 4096/64 C=1 TTFT 22,764 → **22,650 ms**(无回退)。细节见 [RUNBOOK §3.2](RUNBOOK.md)。

| 并发 | prompt/output | out tok/s(**含 TTFT**) | TTFT 均值 | TPOT 均值 | 完成 |
|---|---|---|---|---|---|
| C=1 | 256 / 128 | **16.18** | 2022 ms | **46.37 ms** | 8/8 |
| C=2 | 256 / 128 | 19.36 | 3287 ms | 78.13 ms | 8/8 |
| **C=1** | **4096 / 64** | 2.49 | **22764 ms** | 46.95 ms | 2/2 |
| **C=1** | **4096 / 64**(util 0.85 复测) | **2.50** | **22650 ms** | 46.50 ms | 2/2 |
| C=2 | 4096 / 64 | 2.59 | 34162 ms | 243.21 ms | 4/4 |

与上一版(4096 上下文 / MBT=1024 / 阈值 4096 ⇒ 全 CPU 预填充)对照:

| | 旧配置 | **交付配置** |
|---|---|---|
| 4096/64 C=1 TTFT | 29320 ms | **22764 ms(1.29×)** |
| 256/128 C=1 TPOT | 45.03 ms | 46.37 ms(**解码不变**,GPU 预填充只作用于 prefill) |

* 纯解码 ≈ **22 tok/s**(C=1,`1/TPOT`),且**与上下文长度基本无关**;
* 吞吐在 C≥2 就饱和、TPOT 升高 ⇒ 瓶颈是**每层引擎调用的延迟**(不是带宽,见 `dev-docs` 的 R8);
* **4096/64 的 C=2 那一格 TPOT 243 ms** 是 chunked prefill 的正常代价:两个长 prompt 的
  chunk 与 decode 步交错,长 prompt 场景建议按"单路长 + 单路短"使用。
* 交付配置下 **3 个 ~8K token 请求并发**:2 个 35.9 s + 1 个 47.0 s(两路并行 + 一个排队),
  三个不同密钥**全部检索正确** ⇒ 并发下无跨序列 KV/state 串扰。

### 2.4 上下文 / 显存预算(TP=2,单卡 A100-40GB,交付配置)

| 项 | 实测(util 0.85,2026-09-19 复测) |
|---|---|
| 每 rank 非专家权重 | **~15.1 GiB**(self_attn 10.39 + embed/lm_head 2.36 + dense_mlp 1.08 + shared 1.01,检查点口径) |
| Available KV | **11.38 GiB** |
| **GPU KV cache size** | **971,949 tokens** |
| 该长度(256K)下的 KV 并发 | **3.71×**(两路 256K 只占 54%) |
| GPU 预填充 staging(RAM 里让出来的) | **7.59 GiB / rank**(进程级持久,见 RUNBOOK §3.2) |
| 预检余量(`GPU prefill ACTIVE ... slack`) | **+3.85 GiB** |
| 主机内存 | ~595 GB(引擎单份 NUMA 分片 + vLLM 源张量) |

**崩溃复现与验收(§601)**:同一台机器上,util 0.90 时 **32,077-token** 的真实请求
在 `chunk_kda_with_fused_gate` 里差 52 MiB 崩掉整个服务;改用 util 0.85 + `expandable_segments`
+ 新的预检口径后,同一量级的请求(**32,077 token,TTFT 108.4 s**)与
**两路并发 14,084 + 15,423 token** 请求全部跑通、零 OOM。

`--max-model-len` 实测阶梯(util 0.90、bf16 KV):

| maxlen | 启动 | KV 池 | 该长度下并发 |
|---|---|---|---|
| 256K | ✅ | 988,081 token | **3.77×** ← 交付 |
| 512K | ✅ | 843,055 | 1.61× |
| 704K | ✅ | 763,177 | 1.06×(零余量) |
| 768K | ❌ | vLLM 自报上限 733,312 | — |
| 1M | ❌ | 需 11.57 GiB,只有 6.87 GiB | 需 fp8 KV |

* **KV 单价 ~11.9-12.3 KB/token**(不是 V4.1 的 ~40 KB/token):只有 11 层 NoPE 稀疏 MLA
  带 per-token KV,每层 512 维 latent × 2 B = 1024 B。
* 池子会随 maxlen **变小**(KDA state 池随序列长度增长)⇒ 单请求上限与总容量互相挤,
  自洽天花板 ≈ **733K**;而把并发从 4 限到 2 反而让池子**变大**(919,520 → 988,081),
  因为 state 池按序列槽位分配。
* 长文可用性已验:29,746-token prompt 密钥埋文末,**检索完全命中**,约 3.5 ms/token
  (深度检索验证;不再补 512K+ 的整段 prefill 验证,已按上面的决定收口)。
* **要 1M 只有一条路:fp8 KV**(NoPE 感知 528 B blob;现成 656 B blob 只到 ~948K)。
  **决定(2026-09-19):不做 —— 本硬件上以 256K × 2 路为交付目标**;512K/704K 仅记录
  「配置能起」,不是交付目标。

### 2.4b 上下文长度能开多大(2×A100-40GB / TP=2 实测)

`--max-model-len` 实测启动阶梯(util 0.88、bf16 KV):

| `--max-model-len` | 启动 | KV 池 | 该长度下的并发 |
|---|---|---|---|
| 256K | ✅ | 919,520 tok | **3.51×**(推荐) |
| 512K | ✅ | 843,055 tok | 1.61× |
| 704K | ✅ | 763,177 tok | 1.06×(**零余量**) |
| 768K | ❌ | vLLM 自报上限 733,312 | — |
| 1M | ❌ | 需 11.57 GiB,只有 6.87 GiB | 需 fp8 KV |

* KV 成本 **~11.9-12.3 KB/token**(只有 11 层 NoPE 稀疏 MLA 带 per-token KV;
  每层 512 维 latent × 2 B = 1024 B)。注意这个数**不是** V4.1 的 ~40 KB/token。
* **池子会随 maxlen 变小**(KDA 线性注意力的 state 池随序列长度增长)⇒ 单请求上限与总
  容量互相挤,自洽天花板 ≈ **733K**。
* 长文可用性已验:704K 配置下 **29,746 token** 的 prompt、密钥埋在文末,
  **检索完全命中**;速率 ≈ **3.5 ms/token**(512K 单请求约 30 min)。
* 脚本默认已改为 `MAXLEN=262144 / MBT=8192 / SEQS=2`(**交付配置:256K × 2 路**)。
  实测该配置 **KV 池 988,081 token**(11.55 GiB,KV 并发 3.77×),两路 256K 只占 53%,
  剩下的是工作区;把并发从 4 降到 2 反而让池子变大(919,520 → 988,081),因为 KDA state 池
  按序列槽位分配。并发验收:3 个 ~8K token 请求同时发,**2 个 35.9 s + 1 个 47.0 s**
  (两路并行 + 一个排队),且三个不同密钥**全部检索正确**(无跨序列串扰)。
* **要 1M 只有一条路:fp8 KV**(SM8x 稀疏 MLA 的 fp8 变体 + NoPE 感知的 528 B blob;
  现成的 656 B blob 只到 ~948K)。`--mamba-ssm-cache-dtype` 对容量**无**帮助,
  TP>2 对 MLA 的 KV **无**帮助(每 rank 复制),TP=3 不整除、SM8x 无 DCP。

### 2.5 预填充成本模型(为什么长 prompt 的 TTFT 是秒级)

引擎侧微基准(`scripts/bench_fp8_mcurve.py`,E=288 / H=4096 / I=2048,60 线程;
**单进程满 I**,服务里 TP=2 每 rank I=1024 ≈ 一半工作量):

| 每专家行数 M | 1 | 6 | 12 | 28 | 57 | 114 |
|---|---|---|---|---|---|---|
| ms / 层 | 31.6 | 61.9 | 105.8 | 244.0 | 493.3 | 959.7 |
| 专家-token 吞吐 | 9.1k | 27.9k | 32.7k | 33.1k | 33.3k | **34.2k /s** |

**读法**:M≥28 后每层的"专家-token 吞吐"饱和在 **~33k/s** ⇒ 预填充是**吞吐受限**,
与并发无关。`M = B/36`,所以 4096-token 的 chunk 每层 M≈114 ⇒ 42 层约 40 s 量级,
与实测 **TTFT 22.8 s**(4096-in,GPU 预填充;旧配置全 CPU 时是 29.3 s)同量级。

**GPU 流式预填充(GLM 的 FP8 版本已接线)**:`gpu_prefill_fp8.py` 是 `gpu_prefill.py`
的 FP8 孪生(e4m3 单字节 + fp32 block-128),按引擎类别自动选择后端,无需开关。
实测(GLM-5.3-Flash,TP=2,`MBT=8192`):

| | 每层 | 说明 |
|---|---|---|
| 权重装配(H2D 3.62 GB/rank) | **207 ms** | 占预填充 wall time 的 ~90% |
| GPU MoE 内核(gate_up + down) | **23 ms** | 比 CPU 引擎快约 10× |
| 非 MoE(attention/dense/采样) | — | 反推 ~4.4 s / 2k token |

端到端(3860-token prompt,同进程切阈值):**CPU 26.60 s → GPU 25.38 s(0.95×)**。
**收益之所以小,是 GLM-5.3 的结构决定的**:它的 KDA(mamba-like)state 只在
**block_size = 2176** 的边界上写,调度器因此把预填充 chunk 钉在 2176 token
(`scheduler.py` 的 `aligned_end = end // block_size * block_size`),
而 GPU 路径每层的权重 DMA 是**固定成本**,盈亏平衡点约 1.9k token/chunk。
⇒ 对没有 state 分页的模型(如 DeepSeek-V4 系列)同一个后端能吃满 MBT,收益是 1.5-2×;
对 GLM-5.3 想要更多,需要把装配与 attention 重叠(把 207 ms 压向 135 ms 的 1D DMA 地板)。

**两个 GPU 预填充开关(各自独立,默认都是关闭的)**:

| 开关 | 作用 | 收益(实测) | 代价 |
|---|---|---|---|
| (默认开启,无需开关) | 装配走 side stream,与**同层 attention** 重叠 | **1.20×**(26.6 → 22.3 s) | **0 显存** |
| `XIAOTU_GP_ASM_PREFETCH=1`**(默认关,不推荐)** | ping/pong **跨层**预取(第二套 K-major) | 再 **1.07×**(22.6 → 21.1 s) | **+3.38 GiB/rank** ⇒ 需把 `--gpu-memory-utilization` 从 0.88 降到 **0.82**(KV 池变小,约 -285k token 容量) |

⚠️ **本开关默认关闭,建议保持关闭**:显存在这台机器上是比预填充时延更稀缺的资源。
它换来的只有 1.07×,却要占 3.38 GiB/rank(= KV 容量少 ~285k token 的 GLM 上下文),
而且在 util 0.88 下**会让引擎 OOM 死掉**(实测 38.02/39.49 GiB 已分配后一个 30 MiB 请求
失败);现在分配前会检查"槽位 + 4 GiB 工作区",不满足就拒绝并退回免费的那份重叠。
**PCIe 5.0 及以上更不建议开**:装配时间减半后会低于身后的计算,收益趋近 0,显存却照样要付。
它只在"装配是长杆 且 显存有余量"时才有意义(即本机 PCIe 4.0 + FP8 + 你不需要长上下文)。

**CPU 内核有个更划算的开关**:`XIAOTU_MOE_FP8_BF16_MMA=1` 让 FP8 CPU tile 改走
AVX512-BF16 `vdpbf16ps`(32 个 bf16/指令、激活无需转换)。GLM 形状 M-curve:
**M≥6 快 1.17-1.20×**(M=1 反而慢 13%,所以闸门只在 `M>4` 生效,单流解码仍走精确路)。
端到端(全 CPU 预填充,3862-token):**TTFT 26.60 → 23.72 s = 1.12×**,needle 检索仍完全命中。
代价:权重被舍入到 bf16(两路 rms_rel 3.63e-3),**因此默认关闭**。
注意在 GLM-5.3 上这个开关比 GPU 预填充更值(chunk 被 KDA 限制在 2176 ⇒ 预填充本来大多走 CPU)。

### 2.6 投机解码(MTP):**已实现,默认开 `SPEC_K=1`**(2026-09-20 实验完成)

> **结论先行**(2×A100-40GB / TP=2 / util 0.82 / 官方 `vllm bench serve`):
> GLM-5.3-Flash 的单层 MTP **接受长度 k=1 只有 1.46**,k=4 才到 ~1.63(饱和);
> 但 **k=1 的 TPOT 其实是小赚(约 −3%),真正的代价是 KV 池 −27%**。
> 因此**默认按用户要求开 `SPEC_K=1`**;`SPEC_K=0` 可关、`>1` **不要开**。
>
> | k | accept(random) | accept(ShareGPT) | out tok/s(256/128 N=8×2) | TPOT | KV 池 |
> |---|---|---|---|---|---|
> | 0 | 1.000 | 1.000 | 16.14 / 16.13 | 45.76 / 45.72 ms | **915,487** |
> | **1(默认)** | **1.469** | **1.459** | 16.65 / 16.12 | **43.33 / 45.08 ms** | 666,366 |
> | 2 † | — | 1.539 | — | — | — |
> | 3 † | — | 1.595 | — | — | — |
> | 4 † | 1.619 | 1.626 | — | 75 ms(单请求) | — |
>
> * † k=2/3/4 的 accept 是**用一次独立的 k=4 运行**的 `num_accepted_tokens_per_pos` **反推**的
>   (那次 p0=0.400,与上面直接实测的 k=1 = 1.459 有 run-to-run 差异,**两组不要混着比**);
>   逐位接受率衰减很快(p0≈0.40~0.46、p1≈0.14、p2≈0.06、p3≈0.03)⇒ **单层回收在 k≈4 就饱和**;
> * **为什么 k>1 明显更差**:CPU 专家路径下,被验证 token 数 T=1+k 越大,访问的
>   **不同专家数**越多(修正模型 `na(T)=256·(1−(31/32)^T)`:8 → 15.8 → 30.4 @T=1/2/4)
>   ⇒ 每个 decode step 的权重流量近乎翻倍,而 accept 只涨到 1.65 ⇒ 净负;
> * **k=1 的账**:accept 1.46 × 每步约 2× 成本 ⇒ TPOT 基本持平(实测 −3%,在噪声内);
>   换来的是 **KV 池 915,487 → 666,366(−27%)**,256K 并发 3.49× → 2.54×
>   (仍满足"2 路 256K"交付口径,但余量变小);
> * **ITL 脉冲化**:中位 46 → 64 ms(k=1;k=4 会更差)。
>
> 复现:`SPEC_K=0|1 GPU_UTIL=0.82 bash scripts/serve_glm53_mainline.sh`。

**实现(2026-09-20 落地)**:插件补上了 GLM 的 **draft 层判定** —— `layers.45` 在模型里只
构造一次,DSpark 那套"同 prefix 第二次出现"认不出它;现在按"**层号 ≥ 目标模型
`num_hidden_layers`**"识别(`hybrid_model.is_spec_draft_layer_index`),命中即**强制 GPU 常驻**
(实测 **3.38 GiB/rank**),不再落 CPU。启动日志会打印:

```
[vllm-xtu-moe] spec draft layer model.layers.45.mlp.experts (idx=45) kept GPU-resident
[vllm-xtu-moe] GPU-resident(V4.1) model.layers.45.mlp.experts: 3.38 GiB on cuda:0
```

**实现(2026-09-20 落地)**:插件补上了 GLM 的 **draft 层判定** —— `layers.45` 在模型里只
构造一次,DSpark 那套"同 prefix 第二次出现"认不出它;现在按"**层号 ≥ 目标模型
`num_hidden_layers`**"识别(`hybrid_model.is_spec_draft_layer_index`),命中即**强制 GPU 常驻**
(实测 **3.38 GiB/rank**),不再落 CPU。启动日志会打印:

```
[vllm-xtu-moe] spec draft layer model.layers.45.mlp.experts (idx=45) kept GPU-resident
[vllm-xtu-moe] GPU-resident(V4.1) model.layers.45.mlp.experts: 3.38 GiB on cuda:0
```

**⚠️ MTP 的显存契约(必看)**:开 MTP 后启动 profile 报的 `peak activation` 从 2.9 GiB 掉到
**0.84 GiB**(draft 只在 decode 跑,profile 量不到),vLLM 因此定出**更大的 KV 池**;叠加
draft 的 3.38 GiB ⇒ **util 0.85 下 32k 预填充会 OOM 打死引擎**。要么 `GPU_UTIL=0.82`,
要么 `GP_PREFILL=0`。详见 `KNOWN_LIMITATIONS.md` §8。

**以下为 2026-09-19 的原始评估(保留备查)**:

**它其实"能开"**(三条都已经在机器上核实过):

| 事实 | 证据 |
|---|---|
| 检查点自带 **1 层 MTP** | `config.json: num_nextn_predict_layers = 1`;权重在 `model.language_model.layers.45.*`(1760 个张量:`eh_proj`/`enorm`/`hnorm`/`shared_head.norm` + **288 专家的 MoE**(1728 个)+ DSA/MLA attention(indexer/kv_a_proj…)) |
| vLLM 主线支持该 draft | `registry.py`:`"Glm5NextMTPModel": ("vllm.models.glm5next","Glm5NextMTP")`;`speculative.py` 的 `MTPModelTypes`/`SpeculativeMethod` 接受 `method="mtp"` |
| 同机有正收益先例 | DeepSeek-V4.1 的 **DSpark**(**另一条** speculative method,见下):**16.64 vs 13.85 t/s(+20%)**,TPOT 36.75 vs 42.86 ms,贪心输出逐字节一致(RUNBOOK §5.4) |

> ⚠️ **DSpark ≠ MTP,别把两者的结论互相套用**。这个 vLLM 里是两条独立实现:
> * `dspark`(`DSparkModelTypes`):DeepSeek-V4.1 专属,上游 commit `e77daef89e
>   [Model] Support DeepSeek-V4.1-Flash (#56214)` 引入;draft = `mtp.0/1/2` **3 层**、
>   block=5、自带 parallel drafting;**只在 V2 GPU model runner 实现**
>   (`vllm/config/vllm.py:_get_v1_model_runner_unsupported_features` 明确把 `dspark` 列为 V1 不支持)。
> * `mtp`(`MTPModelTypes` ⊂ `EagleModelTypes`):通用 MTP,**跑在我们这条 V1 runner 上**
>   (`vllm/v1/worker/gpu/model_runner.py`),GLM-5.3-Flash 的 draft 就是 `layers.45` 这 1 层。
> 插件里那套 draft 逻辑(按 prefix 第二次出现判定、`XIAOTU_DRAFT_LAYERS` 默认 3)是**为 DSpark 写的**。

启用方式(要重启服务):
`--speculative-config '{"method":"mtp","model":"<同一 ckpt>","num_speculative_tokens":1..3}'`
(单层 MTP 通过复用 hidden state 支持 k>1)。

**但现在开的硬前提没满足,而且 CUDA graph 这一条要说清楚(2026-09-19 复核)**:

1. **draft 必须常驻 GPU,而插件的 draft 识别不认 GLM 的形态**。插件是按 DSpark 写的:
   靠"同一个 prefix 第二次出现"判断 draft 层(`_instance_index(prefix) > 0`),并**强制其 GPU 常驻**
   (`hybrid_model.py:_is_draft`,默认 `XIAOTU_MOE_RESIDENT_DRAFT=1`);理由是实测
   **draft 走 CPU 时投机净负(7.11 vs 不开 10.76 t/s)**。GLM 的 MTP 层 prefix
   (`Glm5NextMultiTokenPredictor` 用 `f"{prefix}.layers.{idx}"`,`idx = num_hidden_layers = 45`)
   在整个模型里**只出现一次**(主模型是 0..44)⇒ 它会被当成普通层,按
   `XIAOTU_MOE_GPU_RESIDENT_LAYERS` 决定去留,默认**落 CPU** ⇒ 按插件自己的实测口径收益变负。
   **不改代码的临时办法**:`XIAOTU_SPEC_DECODE=1` + `XIAOTU_MOE_GPU_RESIDENT_LAYERS=45`
   (该 env 支持 `0-4,10` 这种写法 ⇒ 单值 `45` 合法)。
2. **CUDA graph:对 `mtp` 是"能用、待实测",不是"不能用"**。
   * 那句"CUDA graph 下草稿模型捕获会崩 ⇒ 强制 eager"是 **DSpark 专属**:
     `scripts/serve_prod_8070.sh` 在开 SPEC 时固定 `EAGER=1`,注释写明"换来单路延迟 ~+43%"。
   * 通用 MTP 路线**有图捕获代码**:`vllm/v1/worker/gpu/spec_decode/speculator.py` 里的
     `init_cudagraph_manager(cudagraph_mode)` / `capture()`,`MTPSpeculator` 继承
     `AutoRegressiveSpeculator` 走同一套;vLLM 的 V1 兼容性黑名单里只列了 `dspark`、
     `adaptive draft verification`,**没有禁 `mtp` 的 CG**。
   * 而且**当前 8070 本身就在跑 CUDA graph**(启动日志 `Capturing CUDA graphs (PIECEWISE): 3`
     + `(FULL): 2`),target 层的 CPU 专家路径在图下已经证明是安全的 ⇒ 我们的 MoE 路径不怕图。
   * **仍未验证的只有一件事**:draft 层(第 45 层)在我们的插件路径下捕获图是否干净。
     历史上 DSpark 那次崩过但**没有留下归因**,所以实验第一步(启动捕获阶段)就能确定:
     崩了就当 CG 不可用(DSpark 那样退 `--enforce-eager`),没崩就保持 CG。

**另外两个代价**:MTP 层也是 DSA/MLA,会多一个 KV 组(略吃 KV 池);它的 288 个专家在
CPU 引擎里是多一次逐层调用,若放 GPU 则需要 ~1.7 GiB/rank 的 FP8 专家常驻(`w13 1.13 + w2 0.56`)。

**结论**:结构上可行、值得做一次实验,但**不是把开关一开就行**。实验清单(需 1 次重启 ≈ 5 min + 约半小时跑数):

1. 先让 draft 认得出:临时 `XIAOTU_SPEC_DECODE=1` + 把第 45 层纳入常驻
   (`XIAOTU_MOE_GPU_RESIDENT_LAYERS=45` 或给插件加 GLM 的 draft 判定),确认日志无
   `draft layer ... forced onto CPU` 告警;
2. 量**接受率**(`/metrics` 的 spec_decode 系列,或 vLLM 日志的 acceptance rate);
3. 同进程 A/B:`256/128 C=1` 与 `4096/64 C=1` 的 TPOT/out tok/s;目标是像 DSpark 那样
   TPOT 降 >15% 且贪心输出语义不变;
4. 若 TPOT 反而变差或 draft 掉回 CPU,就**保持关闭**(当前状态)。

#### 2.6.1 GLM 能不能照 MiMo-2.5 那样"补成 3 层 MTP"来提高接受长度?—— **不能**,但有一条自己的路

> 背景:MiMo-2.5 的调研发现,**检查点里带了 3 层 MTP 权重**,而 vLLM 用两处硬编码常量
> (`_MIMO_V2_*_NUM_MTP_LAYERS = 1`)只跑第 1 层 ⇒ "改常量即可解锁 3 层链"(详见
> `dev-docs/MIMO25_ANALYSIS.md` §9)。**GLM-5.3-Flash 不是这个情况**:

| | MiMo-2.5 | **GLM-5.3-Flash** |
|---|---|---|
| 检查点里的 MTP 层数 | **3**(`model.mtp.layers.{0,1,2}`,各 16 个张量) | **1**(只有 `model.language_model.layers.45.*`;层索引实测 0..45,1760 个张量,`eh_proj`/`enorm`/`hnorm`/`shared_head` 各 1) |
| vLLM 侧挡在哪 | **两处常量**硬编码只跑第 1 层 ⇒ 改代码就能解锁 | **代码本来就是 N 层通用的**(`Glm5NextMultiTokenPredictor` 用 `spec_step_idx % num_mtp_layers` 建 `range(...)`),但**没有第 2/3 层的权重可加载** |
| 想"3 层链"的代价 | 改 2 处常量 + 实测 | **只能自己训/蒸馏 MTP 头**(需数据与算力,不在本项目范围) |

**但有两点让 GLM 的"单层多步"不是纯将就**:

1. **vLLM 的单模块路径就是多步回收**:`Glm5NextMultiTokenPredictor.forward/compute_logits` 里
   `current_step_idx = spec_step_idx % num_mtp_layers` ⇒ 1 层会被**反复复用**做 k>1 步草稿;
   `use_multi_module_mtp()` 的判据是 `min(num_mtp_layers, num_speculative_tokens) > 1`,
   GLM 是 `min(1, k) = 1` ⇒ 走**单模块**(回收)而不是 MiMo 那条多模块链。
2. **GLM 的检查点显式开了 `index_share_for_mtp_iteration = True`** ⇒ `MTPSpeculator` 会走
   "step 0 自算 top-k、step 1+ 复用索引"(`spec_decode/mtp/speculator.py:48`)——即
   **单层多步是模型设计内的用法**,draft 步骤本身也更便宜。

**收益预期(借用 MiMo 修正后的模型,别重复它的第一轮错误)**:CPU 路径上被验证的 token 数 T 越大,
访问的**不同专家数**越大(`na(T)=256·(1−(31/32)^T)`:8 → 15.8 → 30.4 @T=1/2/4),每 token 权重流量
只降 ~10% ⇒ **投机主要买的是"摊销每层 0.5–0.8 ms 的固定 dispatch",不是带宽**。
MiMo 的结论是"**1 个模块 k=3 的净收益≈0,3 个模块在同一验证成本下才有 1.6–1.9×**";
GLM 只有 1 层(回收),因此**预期介于两者之间**:比 MiMo 的 1 模块略好(因为有 index sharing),
但**不要期待 +100%**;同机锚点仍然是 DSpark 3 层只 +20%。

**该做的实验(与 §2.6 同一套,只是把 k 扫开)**:

1. 先解决 draft 判定(见上面第 1 条),否则 draft 落 CPU ⇒ 实测 **−34%**,什么都没意义;
2. 只量**接受长度**:`--num-speculative-tokens`(vLLM)取 **1 / 2 / 3 / 4**,看 accept 的边际;
   判据沿用 MiMo 的表:k=3 时 accept ≥2.5 ⇒ 不值得继续;≈1.5–1.8 ⇒ 值得;k=1 已 ≥1.6 ⇒ 先只上 k=1;
3. 只有当 k=3 的 accept 明显高于 k=1 时才做 TPOT/吞吐 A/B;注意 **ITL 会脉冲化**
   (MiMo 报告实测中位 22.4 → 159.6 ms),交互式流式体验会变差。

**外部信号(仅标题级,未深读)**:社区把 GLM-5.3-Flash 的 MTP 就当作"**第 45 层**"处理
([HF 讨论:某 NVFP4 repack 丢了 layer 45 的 MTP 权重](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4/discussions/1)、
[带 MTP 的 MLX 量化](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-oQ2-MTP));SGLang 侧把它叫 NextN,
并有 [TP8 下 draft forward 越界的 issue](https://github.com/sgl-project/sglang/issues/37548)。
⇒ **没有任何来源显示 GLM-5.3-Flash 存在第 2/3 层 MTP 权重。**

**顺带解释"上游为什么只跑第 0 层"**(这对 GLM 有直接含义:说明"单层多步"是**受支持的路径**而非降级方案):
vLLM 的多模块 MTP 是 **2026-07-30** 才合并的([PR #48892](https://github.com/vllm-project/vllm/pull/48892),为 **Inkling** 的 8 模块做的);
其 body 写明**最大难点**是:第 2..N 个 MTP 模块会吃到前一个模块产出的、**可能被 target 拒绝**的草稿 token,
一旦被拒,这些层的 KV 里就残留**脏值**,必须让 scheduler 支持**对上一 decode 步的 token 做 re-prefill**
(要同时改 scheduler / KV cache manager / coordinator);而**复用第 0 层没有这个问题**——第 0 层的 KV 只依赖已接受的 prefix。
MiMo 那边对应的多层 MTP PR([#31180](https://github.com/vllm-project/vllm/pull/31180))至今仍是 **draft**,
作者自述 "produces acceptance rate of 0"。⇒ 我们给 GLM 开 MTP 时走 `min(1,k)=1` 的单模块回收 + `index_share_for_mtp_iteration`,
是与上游实现现状一致的用法。

### 2.7 数值与稳定性

* 层内数值(`GLM_MODEL=<ckpt> python scripts/test_glm53_fp8_layer.py <layer> 8`):
  真实权重 RMS 相对误差 **3.6e-5 ~ 3.1e-4**(门限 1e-3);
* 引擎确定性门禁 **11/11 逐位一致**;跨层预取开/关输出**逐字节相同**(语义透明);
* 服务级:`vllm bench serve` 各档 **8/8、8/8、2/2、4/4 全部成功**,
  3 路 ~8K 并发(限两路)全部正确,全程 **0 OOM / 0 CUDA 错误**;
  ⚠️ 同一 prompt 重复请求的贪心输出**会**偶发 token 级抖动(near-tie 翻转,与
  batching/prefix-cache 命中有关),这是服务栈既有现象,不作为验收项;
* 主机内存 ~595 GB(引擎单份 NUMA 分片 + vLLM 源张量),GPU 每 rank 36.5 GiB(受
  `gpu-memory-utilization` 的 KV 预留支配)。

---

## 3. MiMo-V2.5(A100 / SM80:单卡端到端已支持;MTP 用 k=1)

> 命名:`MiMo-2.5` 是 **`XiaomiMiMo/MiMo-V2.5`**(310B / 15B active,48 层,256 专家 top-8,
> Hybrid SWA-128 + DiffKV,官方只有 FP8 block-128)。完整调研见内部
> `dev-docs/MIMO25_ANALYSIS.md`。检查点 ~295 GB(17 分片 + `model_mtp.safetensors`)。

**启动(TP=1 单卡 A100-40GB,专家在 CPU)**:

```bash
CKPT=/home/user/.cache/modelscope/models/XiaomiMiMo--MiMo-V2.5/snapshots/master
VLLM_EXPERTS_LOAD_DEVICE=cpu CUDA_VISIBLE_DEVICES=2 \
VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1 \
XIAOTU_GP_ACT_RESERVE_GIB=99 \
python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --tensor-parallel-size 1 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --language-model-only --trust-remote-code
```

**P0 oracle 自检(2026-09-20,实测通过)**:

| 检查 | 实测 |
|---|---|
| 注意力后端 | `[mimo_v2.py:319] Using TRITON_ATTN_DIFFKV for attention.` ✅ SM80 的必需回退 |
| MoE 后端被插件换掉 | `Using CPU Fp8 MoE backend out of potential backends: ['CPU', …]` ✅ |
| Model Runner | `Using V2 Model Runner` ✅ |
| 每层引擎参数 | `xiaotu MOE_FP8 engine: E=256 H=4096 I=2048 topk=8 group=128x128 scales=yes routing=sigmoid/grouped1x1/bias swiglu=plain` ✅ |

**已知坑**:

* 即使 `--language-model-only`,架构仍解析成 `MiMoV2OmniForCausalLM`(会加载 audio
  encoder/quantizer);**必须 `--trust-remote-code`**(否则仓库自定义代码被拒);
* `--dtype` 只能是 `bfloat16`(KV 也只支持 bf16,见 `_shard_fp8_qkv_proj` 的 kv 格式约束);
* TP 被 `num_kv_heads % tp_size == 0` 卡住(GA KV=4 / SWA KV=8)⇒ 3 卡上 **TP=3 不可用**,
  取 TP=1(单卡)或 TP=2;
* **加载很慢**:每层要建一次 xiaotu 引擎(实测 dummy 下 ~2 min/层 ⇒ 48 层 ≈1.5 h),
  实权重还要先读 282 GB;Dummy 加载用于**只验后端**时可用 `--load-format dummy`。
* **GPU 流式预填充可用,而且收益很大(实测)**:L=4096 的 TTFT **47.4 s(CPU)→ 21.1 s(GPU,2.25×)**,
  0 错误。代价是 staging **12.75 GiB/rank**(比 GLM 的 7.59 大,因为 `E=256×I=2048` 更宽)
  ⇒ 要配更低的 `GPU_UTIL`(实测 `0.65`)让出激活余量,KV 池随之变小(47,459 token,
  8K 上下文下 5.8× 并发,够用)。给 MiMo 起服务时**不要**再用 `XIAOTU_GP_ACT_RESERVE_GIB=99`。

**MTP(2026-09-20 实测,TP=1 / GPU 2 / 单请求)**:检查点带 **3 层** dense MTP
(`model.mtp.layers.{0,1,2}`);主线原来只建第 1 层,本项目已改成
`min(checkpoint_layers, num_speculative_tokens)`(照 Inkling 模板)⇒ k=1 只建 1 层、
k=3 走多模块链。实测**结论明确:k=1 可用,k=3 不可用**:

| 方案 | 接受长度 | p0 | 128-token 生成耗时 | vs 不开 |
|---|---|---|---|---|
| 不开 MTP | 1.000 | — | **8.72 s** | 1.00× |
| **MTP k=1(单模块)** | **1.83~1.87** | 0.83~0.87 | **7.33 s** | **1.19×** |
| MTP k=3(多模块) | 1.016 | **0.016** | 21.4 s | 0.41× |

* **k=1 是 MiMo 的正解**:83% 的首位接受率、解码 **+19%**,与 §2.6 对"1 模块 k=1"的
  预期(1.5-1.7 accept、+20-40%)一致;draft 是 dense(不需要插件常驻逻辑)。
* **k=3 多模块路径在当前上游实现下是坏的**:p0 从 0.83 掉到 **0.016**(比随机还差),
  接受长度 1.016 ⇒ 反而慢 2.4×。这正是分析与上游 PR
  [#31180](https://github.com/vllm-project/vllm/pull/31180) 说的 *"Load the weights but
  produces acceptance rate of 0"*,也与 §9.9 的解释吻合:多模块链的第 2..N 个模块会吃到
  可能被拒的脏 KV,需要 re-prefill 才能干净;上游那套 re-prefill 显然没对 MiMo 的
  SWA/DiffKV 生效。
* **推荐**:MiMo 只开 `num_speculative_tokens=1`;不要开 k>1。

复现:`--speculative-config '{"method":"mtp","model":"<ckpt>","num_speculative_tokens":1}'`。

---

## 3b. MiMo-V2.6-Flash-RL(A100 / SM80:**骨架、1M、MTP k=1 已在 P1 通过**;完整端到端待 P4)

> 2026-09-22 新增。事实来源 `docs/EXPERIMENTS.md` **B92–B95**;计划 `dev-docs/MIMO26_PLAN.md`。
> 检查点:`/home/user/.cache/modelscope/models/MiMo-V2.6-Flash-RL`(166 GB;index `total_size` = **161 GiB**)。

**与 V2.5 的关键差异(实测张量头,不是抄 config)**

| 项 | V2.5 | **V2.6** |
|---|---|---|
| 专家精度 | FP8 E4M3 block-128 | **MXFP4**:gate/up `U8 [2048,2048]`、down `U8 [4096,1024]` + **e8m0 block-32** `weight_scale` |
| 注意力 | FP8 | FP8 E4M3 block-128(`qkv_proj.weight_scale_inv` 为 F32) |
| 体积 | 293 GiB | **161 GiB** |
| 专家布局 | — | **gate/up 分离**(非 w13 融合)⇒ **加载期要融合** |
| 分片 | — | **`model_pp0_ep{0..63}_shard0`**,每片含**每一层**的 4 个专家;元数据 `tp_size:4` 是虚惊(非专家权重未被切,TP=2 安全) |
| 投机 | MTP 3 层 | MTP 3 层 + `dflash/` 草稿(**仅调研**) |

**⇒ 最重要的一条**:专家是 **MXFP4 = V4.1 那条已在生产跑的 `MOE_MXFP4` 路径**
⇒ V2.5 分析里"官方只有 FP8、非 FP8 路径都没接线"的缺口**不存在**;
161 GiB(V2.5 293)⇒ 单流 decode 权重流量 ≈4.8 GB/token ⇒ 带宽天花板 **~77 tok/s**(V2.5 为 39)。

**架构**:48 层(1 dense + 47 MoE)、256 专家 top-8、`moe_intermediate_size=2048`、
`hybrid_layer_pattern` = **9 GA + 39 SWA(窗口 128)**、`max_position_embeddings = 1048576`;
GA = 64 Q / **4 KV** / head_dim 192 / v_head_dim 128,SWA = 64 Q / **8 KV** / 192 / 128。

**启动配方(dummy 骨架,已实测)**
```bash
VLLM_EXPERTS_LOAD_DEVICE=cpu CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
  --model <CKPT> --served-model-name mimo26 --tensor-parallel-size 2 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --max-model-len 1048576 --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 --language-model-only --trust-remote-code \
  --kernel-config '{"enable_jit_warmup": false}' --load-format dummy
```
**P1 实测**:骨架 **110 s** 起服务;`Resolved architecture: **MiMoV2OmniForCausalLM**`
⇒ **多模态不需要 `--hf-overrides`**;`[mimo_v2.py:319] Using **TRITON_ATTN_DIFFKV**`
⇒ **hybrid SWA + DiffKV + sink 在 SM80 可用**;
`MOE_MXFP4 engine: E=256 H=4096 I=1024 topk=8 scales=yes routing=sigmoid/grouped1x1/bias **swiglu=plain**`
(`I=1024` 是 TP=2 分片值;`swiglu=plain` 即**无 clamp**,与 V4.1/GLM 的 `clamp@10.0` 不同);
**1M**:`max_model_len=1048576`、KV 池 **2,078,802 token**、**1.98× 并发**(160 s)。

**KV 账(已与实测对账,可直接用于预算)**
* GA(9 层)= `9 × 4 × 320 × 2 B = 23,040 B/token`(全机);**TP=2 每 rank ≈ 11.25 KiB/token**;
* SWA(39 层)= `39 × 8 × 320 × 2 B × 128` ⇒ **每序列常数 ~25.6 MB**,不随上下文增长;
* ⚠️ 引擎**补齐 6 个 padding 层、浪费 15.38%** KV ⇒ **算 1M 显存时必须显式扣掉**;
* **MTP 层按 SWA 处理,不吃全长 KV**(实测池只少 1.3%,与 SGLang "MTP 不给全长 KV" 一致)。

**多模态**:检查点带 vision+audio 塔但 `architectures` 写纯文本 ⇒ **vLLM 自己解析成 omni 类**;
`--language-model-only` 会关掉 mm_prefix(**反而放开后端选择**)。真输入待 P4。
⚠️ SGLang 有一个 **open** issue([#37983](https://github.com/sgl-project/sglang/issues/37983)):
其**视觉** Triton 路径丢 window+sink(静默算成全注意力)——
**vLLM 这条路已核过不成立**(`mimo_v2_omni._forward_window_attn` 与 `triton_prefill_attention` 里
`SLIDING_WINDOW_Q/K`、`USE_SINKS`/`SINKS_BIAS_KEY0` 都有真实实现,不是"收了参数不用")。

**MTP**:`--speculative-config '{"method":"mtp","model":"<CKPT>","num_speculative_tokens":1}'`。
**只用 k=1。** 真权重实测(B98):k=1 首位接受率 **68.5%**(自然文本),接受长度约 1.69;
而 k=3 的第 2/3 位只有 **9.7% / 0.7%** ⇒ 每步仅多 **4%**,还要多算 2 个草稿 token(专家在 CPU)⇒ **不划算**。
上游 #41905 只复用第 0 层;我们的多深度补丁 `eddc6d0eb7` 已于 2026-09-22 **回退掉**(B100)——
k=1 下两者逐字等价,那 17 行零收益。
⛔ **不要再追多深度 MTP**(用户决定,2026-09-22)。

---

## 4. 数据来源与复现

| 数据 | 来源 / 复现命令 |
|---|---|
| DS-V4 阈值扫描(CPU vs GPU) | `report/curve_thr.jsonl`;`scripts/dsv4_prefill_curve.py`(env:`MODES=cpu,gpu LENS=…`) |
| DS-V4 单卡 / 双卡长 prefill | `report/curve_tp1g.jsonl`、`report/curve_tp2.jsonl` |
| DS-V4 纯 MoE 微基准 | `report/moe_micro.json`;`scripts/bench_gpu_moe_prefetch.py` |
| DS-V4 批量 decode | `scripts/bench_llm.py`(`CONCURRENCY=1/32/64/128/256`,`OUT_TOKENS=32`) |
| DS-V4 服务端并发 | `report/server_conc.jsonl`;`scripts/server_concurrency_test.py` |
| 引擎 MXFP4 / FP8 吞吐 | `scripts/bench_cpu_engine.py`、`scripts/bench_fp8_engine.py` |

图表由 `python report/make_figs.py` 从 `report/*.json` 重新生成,输出在 `内部数据 fig/`。

> 免责声明:以上均为**单机初步实测**,共享机器上存在其他负载(会话记录里标注了当时的
> `load average` 与并发任务);不同批次之间的绝对值可能相差 10–30%,请以趋势与量级为准。
