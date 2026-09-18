# 调参报告:DeepSeek-V4-Flash-0731 的推荐运行参数

> 本报告是 `docs/TUNING_PLAN.md` 的执行结果;原始数据、日志、复现命令见
> `report/tuning/`(`summary.jsonl` 是全部测量的一行一条汇总,`raw/*.json` 是
> vLLM 原生结果,`logs/*.log` 是服务端与客户端完整日志)。
> 测试机:3 × A100-PCIE-40GB(生产 8070 已关闭)、EPYC 9654(无 AMX)、1.5 TiB DDR5。
> 全部为**单机实测**;测量期间的 `uptime` 与 GPU 占用记录在每个配置的
> `report/tuning/logs/<TAG>.env` 里。

---

> ### ✅ 状态更新(2026-09-17,v0.21.0)
>
> 本文主体是 **DeepSeek-V4-Flash-0731** 的调参结论,那些数字**仍然有效**。
> 但自 **v0.21.0** 起本项目另外支持了 **DeepSeek-V4.1-Flash(748B)**,
> 且 **GPU 预填充推荐开启**(实测比 CPU 快 **2.0-2.8x**,阈值 **4096**;
> 开启时必须显式 `--kv-cache-memory` 封顶 KV 池,否则会被逐层静默拒回 CPU)。
> V4.1 的推荐参数见 `docs/RUNBOOK.md` 5.9 节与 `RELEASE_NOTES_v0.21.0.md`;
> 端到端性能见 `report/tuning/BENCH_REFERENCE.md` 第 7 节。
>
> ⚠️ 本文中若出现「GPU 预填充默认关闭 / 不推荐」之类的表述,**以 v0.21.0 为准**
> (594-602 节已修完)。全文其余部分未改动。

## 0. 结论速览

| 模型 | 推荐配置 | 实测 |
|---|---|---|
| **DeepSeek-V4-Flash** | **单卡**、`--max-model-len 262144`、`--kv-cache-dtype fp8_ds_mla`、`--kv-cache-memory-bytes 12GiB`、`--max-num-batched-tokens 8192`、`XIAOTU_MOE_THREADS=96`、`--max-num-seqs 64`、`--enforce-eager` | ShareGPT(C=64,输出 128)聚合 **48–58 tok/s**;32K prompt TTFT 41 s;128K prompt TTFT 247 s |

- **256K 上下文不需要 TP=2**:DS-V4 单卡 KV 只需 7.7 GiB(fp8 MLA 下 29.5 KB/token),
  实测容量 437 337 token。
- 与生产参考(lk-moe,≈100 tok/s)相比,我们目前 **48–58 tok/s**,差距 ~2×,
  瓶颈已定位(§4),下一步优化方向明确。

---

## 1. DeepSeek-V4-Flash-0731

### 1.1 推荐启动命令(单卡,256K 上下文,已验证)

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_EXPERTS_LOAD_DEVICE=cpu
# 权重布局固定为 NUMA 分片(1 份内存,每个线程只读本地 node;单拷贝模式已删除)。
                                      #   C=64 实测分片 vs 单拷贝 53.4 → 73.6 tok/s(详见 docs/PERFORMANCE_OPTIMIZATION.md)
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export XIAOTU_MOE_THREADS=192         # ★ 2026-09-10 更新:分片模式下 96→192 有 1.75× 收益
                                      #   (每层 compute 15.7→12.1 ms);未分片时加线程无效,
                                      #   关掉分片时加线程无效,所以布局必须是分片(=默认,已无开关)
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384    # 长 prefill 阈值(见 §1.4)

vllm serve <DEEPSEEK_V4_FLASH_DIR> \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_ds_mla \
  --kv-cache-memory-bytes 12884901888 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --kernel-config.enable_jit_warmup=false \
  --served-model-name DeepSeek-V4-Flash-xiaotu
```

**为什么是这些值**

| 参数 | 值 | 依据 |
|---|---|---|
| `--kv-cache-dtype` | `fp8_ds_mla` | MLA 专用 fp8:KV **29.5 KB/token**(bf16 翻倍);262144 token 只要 7.7 GiB |
| `--kv-cache-memory-bytes` | 12 GiB | **必须显式限制**:不限制时 KV 吃满 40 GB,GPU prefill 的 K-major staging(每层 ~2 GiB)直接 OOM(实测 4/8 请求失败)。12 GiB 给 staging + 激活留 ~5 GiB |
| `--max-num-batched-tokens` | 8192 | **上限**:32768 时 vLLM 的"单请求 max_model_len KV 需求"变成 25.83 GiB > 单卡可得,直接拒绝启动(实测);8192 是已验证可用值 |
| `XIAOTU_MOE_THREADS` | 96 | 48 → 29.4 tok/s,96 → 48.5,**192 → 50.3(只 +4%)** ⇒ 96 是性价比拐点(也把 96 个核留给别的进程) |
| `--max-num-seqs` | 64 | 并发是吞吐主开关(§1.3),64 已能跑满;更大需 KV 支持 |
| `--enforce-eager` | 开 | CUDA graph **不再崩溃**(§2.2 修了 use-after-free)但对吞吐无收益(48.8 vs 48.5),eager 更省事 |

### 1.2 256K 上下文可行性(单卡)

```
GPU KV cache size: 437,337 tokens, Maximum concurrency for 262,144 tokens per request: 1.67x
```

- 单卡 A100-40GB 上 `--max-model-len 262144` 一次起服务成功;
- 视 KV 上限不同,262 144 上下文可同时服务 1–2 条请求;
- 32K / 128K prompt 实测 TTFT 见 §1.4(**可行但慢**,原因已定位)。

### 1.3 ShareGPT 吞吐(单卡,输出 128–256 token)

| 并发 C | 线程 | batching | graph | 聚合输出吞吐 | TTFT | TPOT |
|---:|---:|---|---|---:|---:|---:|
| 4 | 96 | 8192 | eager | 30.5 | 9.9 s | 122 ms |
| 8 | 96 | 8192 | eager | 43.8 | 6.7 s | 170 ms |
| 16 | 96 | 8192 | eager | 37.4 | 7.3 s | 401 ms |
| 32 | 96 | 8192 | **graph** | 52.6 | 8.0 s | 580 ms |
| 64 | 96 | 8192 | **graph** | 54.9 | 15.9 s | 1106 ms |
| **64** | 96 | 8192 | eager | **48.5 / 57.6** | 17.0–18.0 s | 1188 ms |
| 64 | **48** | 8192 | eager | 29.4 | 17.8 s | 2054 ms |
| 64 | **192** | 8192 | eager | 50.3 | 16.9 s | 1148 ms |
| 64 | 96 | 8192 | graph | 48.8 | 16.9 s | 1186 ms |

读数:
- **并发是主开关**(C=4→64:30→55+),但远非线性:每 step 的固定成本太大(§4);
- **引擎线程 48→96 几乎翻倍**,说明 CPU MoE 在关键路径上;
- CUDA graph 对吞吐**没有**收益(与 eager 持平),只影响启动开销;
- `--max-num-batched-tokens` 想调大到 32768 会被 §1.1 的 KV 校验挡住(单卡不能两者兼得)。

### 1.4 长上下文(256K 配置下的实测)

| prompt | 输出 | TTFT | 说明 |
|---:|---:|---:|---|
| 32 768 | 64 | **41.0 s** | ≈800 tok/s prefill |
| 131 072 | 64 | **246.8 s** | ≈530 tok/s prefill |

原因:每层专家权重 3.19 GiB × 43 层 = **137 GiB**,`--max-num-batched-tokens 8192`
意味着一轮 prefill 被切成 16 块 → **每块都要把 137 GiB 重新流一次**(每块 ≈5.5 s H2D
+ ≈8 s GPU MoE)→ 16 × 13.5 s ≈ 216 s,与实测一致。
⇒ 长上下文要提速,必须**减少分块数**(更大的 `--max-num-batched-tokens`)或
**双卡分摊**(TP=2 时每 rank 只流一半专家)。

### 1.5 单卡 vs TP=2(参考历史数据)

| 场景 | 单卡 | TP=2 + EP |
|---|---|---|
| 4 137 token prefill | 5.76 s | **3.99 s** |
| 16 039 token prefill | 17.7–18.2 s | **13.96 s** |
| decode(单序列) | **更好**(TP=2 两 rank 冗余算全量专家) | 更差 |

⇒ **decode 用单卡,长 prefill 用 TP=2**;若必须二选一,单卡(256K 上下文 + 48–58 tok/s)。

---

## 2. 目标 FP8 模型(暂不声明支持)

> ⏸️ **暂不声明支持**:该模型的实测数据早于 v0.2 的改动(执行模型 / 小 batch 路径 / EP 存储分片),**未在当前代码上复验**。复验计划见 `docs/HANDOFF_v0.2pre.md` §5.1。

---

## 3. 复现方式

```bash
# 服务端(参数化;TAG 决定日志/结果文件名)
MODE=dsv4 TAG=myrun PORT=8081 TP=1 GPUS=2 MAXLEN=262144 KV_DTYPE=fp8_ds_mla \
  GPU_UTIL=0.90 KV_MEM_BYTES=12884901888 SEQS=64 MAX_NBT=8192 \
  PREFILL_MIN=384 THREADS=96 OMP=48 EAGER=1 bash scripts/tune_serve.sh

# ShareGPT 压测
TAG=myrun_c64 PORT=8081 MODEL=DeepSeek-V4-Flash-xiaotu C=64 N=64 OUT=128 \
  TOKENIZER=<MODEL_DIR> bash scripts/tune_client.sh     # 结果进 report/tuning/summary.jsonl

# 长上下文(合成 prompt)
TAG=longctx PORT=8081 MODEL=DeepSeek-V4-Flash-xiaotu C=1 N=1 OUT=64 IN_LEN=131072 \
  TOKENIZER=<MODEL_DIR> bash scripts/tune_client.sh

# 一组"重启型"参数的串行扫描(每个变体独立端口、自动等显存释放)
SWEEP="THREADS=48 THREADS=96 THREADS=192" C=64 N=64 OUT=128 bash scripts/tune_sweep_serve.sh
```

---

## 4. 瓶颈定位与下一步(优化起点)

**每 step 成本分解**(单卡 DS-V4,C=16,eager):

| 组成 | 每 step | 占比 |
|---|---:|---:|
| GPU 注意力(43 层) | 0.076 s | 25% |
| **CPU MoE 引擎内部**(A 0.65 + B 0.36 + C 0.02 ms/层) | 0.044 s | 14% |
| `cpu_decode` 入队 | 0.002 s | 1% |
| **其余**(D2H→host 回调→H2D 往返 + GPU dense/超连接/路由) | **0.183 s** | **60%** |

- 引擎每层只花 ~1.03 ms,但整个 step 每层要 6.5 ms ⇒ **每层的 GPU↔CPU 往返延迟是主要成本**,
  它随 batch 增大而增长(C=64 时 ~25 ms/层),所以吞吐卡在 ~50 tok/s。
- 与 lk-moe(≈100 tok/s)的差距 ~2×,可动的方向(按预期收益排序):
  1. **减少每层往返**(43 次/step):例如把 `ids/weights` 的 D2H 合并进 hidden 的那次拷贝、
     或在 GPU 上直接做 gather/写回,少一次同步点;
  2. **降低并发时的引擎成本**:高并发下每层活跃专家数 ~200、每专家 token 数 ~1.5,
     `XIAOTU_MOE_NCGU/NCD`(每专家 N 分块)与 worker 子集启发式可针对这个形状调优;
  3. **长上下文 prefill**:提高 `--max-num-batched-tokens`(需先解决 vLLM 的
     max_model_len KV 校验)或长 prefill 走 TP=2,把 137 GiB 的流式遍数从 16 降到 1–4。

## 5. 已知限制(本轮实测确认)

| 限制 | 说明 |
|---|---|
| DS-V4 单卡 256K + 大 batching 不可兼得 | `--max-num-batched-tokens ≥ 32768` 时 vLLM 要求 25.83 GiB KV > 单卡可得,拒绝启动 |
| 长 prompt TTFT 大 | 128K prompt 需 247 s(43 层 × 137 GiB 权重按 8192-token 分块重复流式) |
| 并发扩展次线性 | C=64 时每层往返 ~25 ms,聚合 48–58 tok/s(生产参考 ~100 tok/s) |
| 未验证项 | `--async-scheduling`、MTP 投机解码、`XIAOTU_MOE_NCGU/NCD` 调优 |
