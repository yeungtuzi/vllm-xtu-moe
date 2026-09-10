# 调参计划:DeepSeek-V4-Flash-0731 与 Qwen3.8-Flash-Next 的最优运行参数

> 状态:**执行中**。原始数据落在 `report/tuning/`,最终结论写进
> [`TUNING_REPORT.md`](TUNING_REPORT.md)。
> 本文件是"怎么测"的协议与阶段计划;结论只在报告里,避免两处数字不一致。

## 0. 目标与验收标准

| 项 | 要求 |
|---|---|
| 上下文 | **≥ 262 144 token**(DS-V4-Flash 模型上限 1 048 576;Qwen3.8-Flash-Next 上限正好 262 144) |
| 单次输出 | 支持 **最多 131 072 token**(实际压测只生成 4–8K,避免测试时间失控) |
| 数据集 | **ShareGPT**(`ShareGPT_V3_unfiltered_cleaned_split.json`,94 145 段对话),自然长度提示词 |
| 指标 | 输出吞吐(tok/s)为主,同时记 TTFT / TPOT / 端到端时延 / 失败率 / 加载时间 |
| 参考基线 | 生产环境 `lk-moe/deepseek-v4-flash-0731` ShareGPT 最大吞吐 **≈100 tok/s** |
| 交付物 | 两个模型各一份**可直接用于实际工作**的推荐参数 + 数据 + 复现命令 |

**"可用于实际工作"的定义**:服务能起来、长上下文不 OOM、输出正确、吞吐在候选参数里最优,
并且参数之间不冲突(例如 `--max-num-seqs × --max-model-len` 不能超过 KV 容量)。

## 1. 硬件与软件基线

| 项 | 值 |
|---|---|
| GPU | 3 × A100-PCIE-40GB(SM 8.0,无 NVLink),本轮**全部可用**(生产 8070 已关闭) |
| CPU / 内存 | EPYC 9654 2×96 核(NPS4,8 NUMA 域,无 AMX)/ 1.5 TiB DDR5(实测单流读 860 GB/s) |
| vLLM | 主线 `6c73b08` + 本插件(`VLLM_EXPERTS_LOAD_DEVICE=cpu`,`XIAOTU_MOE_SINGLECOPY=1`) |
| 测量纪律 | 每轮记录 `uptime` 与 GPU 占用;同一配置重复 3 次取中位;**关闭 prefix caching**(避免命中缓存) |

## 2. 测量协议(所有阶段统一)

### 2.1 服务端

`scripts/tune_serve.sh`(本轮新增,参数化)负责启动,固定项:

```
VLLM_EXPERTS_LOAD_DEVICE=cpu  XIAOTU_MOE_SINGLECOPY=1  HF_HUB_OFFLINE=1
VLLM_USE_FLASHINFER_SAMPLER=0  VLLM_ENGINE_READY_TIMEOUT_S=7200
--host 0.0.0.0 --port <固定端口,避开 8070> --enforce-eager --kernel-config.enable_jit_warmup=false
--served-model-name <name> --no-enable-prefix-caching
```

可变项(扫描维度):`--tensor-parallel-size` / `--enable-expert-parallel` /
`--max-model-len` / `--max-num-seqs` / `--max-num-batched-tokens` /
`--gpu-memory-utilization` / `--cpu-offload-gb` / `--kv-cache-dtype` /
`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` / `XIAOTU_MOE_THREADS` / `OMP_NUM_THREADS` / MTP 投机。

**每次启动都要记录**:
- 加载耗时(`init engine ... took` 行);
- `GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: x.xx×`
  ⇒ 由它反推 **KV/token 与"262 144 上下文能否放下"**;
- `Using CPU Fp8 MoE backend` / `xiaotu MOE_*  engine: E=… H=… I=…` 两行(确认后端与引擎形状)。

### 2.2 客户端(ShareGPT)

```bash
vllm bench serve \
  --backend openai --host 127.0.0.1 --port <port> --model <name> \
  --dataset-name sharegpt --dataset-path /home/user/lvllm/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 4096 --ignore-eos \
  --num-prompts <N> --max-concurrency <C> --request-rate <inf|R> \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
  --save-result --result-dir /home/user/lvllm/report/tuning --result-filename <tag>.json
```

- **吞吐扫点**:`C ∈ {1, 2, 4, 8, 16, 32, 64}`,`N = max(4×C, 16)`,`output-len = 4096`(每轮 4–8K);
- **长上下文点**:`--sharegpt-input-len`(或合成 prompt)到 32K/128K/256K,验证可行性并测 TTFT;
- 每点重复 3 次,取**中位**并记录最好值;
- 记录 `report/tuning/<tag>.json`(vLLM 原生结果文件)+ 汇总到 `report/tuning/summary.jsonl`。

### 2.3 正确性

每个最终候选配置跑一次 ShareGPT 少量样本(4–8 条),人工核对输出连贯;
DS-V4 另跑 `"The capital of France is"` 固定提示比对首 token。

## 3. 阶段计划

### P0 — 准备与骨架(预计 20 分钟)

1. 写 `scripts/tune_serve.sh`(服务端)与 `scripts/tune_sharegpt.sh`(客户端 + 结果归集);
2. 统一结果目录 `report/tuning/`,写 `summary.jsonl` 追加器;
3. 确认 ShareGPT 文件、模型目录、端口空闲(8070 已关)。

### P1 — 256K 上下文可行性(最关键的一步)

目标:找到"能服务 262 144 上下文"的最小配置。

| 候选 | 说明 |
|---|---|
| DS-V4:TP=1 + `--kv-cache-dtype fp8` + util 0.90 | KV 减半;单卡 decode 更快 |
| DS-V4:TP=2 + EP + `--kv-cache-dtype fp8` + util 0.90 | 双卡 KV 翻倍;prefill 更快 |
| Qwen:TP=2 + EP + `--cpu-offload-gb {12, 24}` | 权重必须 offload;KV 需 ≥ 256K |

判定:服务能起 + `Maximum concurrency for 262144 tokens per request ≥ 1.0×` + 128K 长提示冒烟通过。
若 256K 放不下 ⇒ 回落 128K,并明确记录"可行的最大上下文"与瓶颈(KV 容量/权重)。

### P2 — 吞吐扫描(ShareGPT,输出 4K)

按"影响最大 → 最小"逐个变量,每次只动一个(其余固定为 P1 最优):

| # | 维度 | 取点 | 是否需要重启 |
|---|---|---|---|
| 1 | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | 0 / 128 / 384 / 1024 / 4096 / 16384 / 1e9(全 GPU) | 否(可用 `..._FILE` 运行期切换) |
| 2 | 并发 `--max-concurrency` | 1 / 2 / 4 / 8 / 16 / 32 / 64 | 否 |
| 3 | `--max-num-batched-tokens` | 1024 / 2048 / 4096 / 8192 / 16384 / 32768 | 是 |
| 4 | `--max-num-seqs` | 4 / 8 / 16 / 32 / 64 / 128 | 是 |
| 5 | `--kv-cache-dtype` | auto / fp8 | 是 |
| 6 | `XIAOTU_MOE_THREADS` | 48 / 96 / 192(与 `OMP_NUM_THREADS` 联动) | 是 |
| 7 | 并行度 | TP=1 vs TP=2(+EP) | 是 |
| 8 | `--enforce-eager` | on / off(CUDA graph) | 是 |
| 9 | MTP 投机解码 | 开 / 关(`num_nextn_predict_layers=1`) | 是 |
| 10 | 输出长度 | 4096 / 8192 | 否 |

优先做**不需要重启**的维度(1、2、10),再按预期收益排(3、4、5、6、7、8、9)。
DS-V4 单次启动 5–16 分钟 ⇒ 整个 P2 控制在 **≤ 12 次启动**;Qwen 启动约 25 分钟 ⇒ **≤ 6 次**。

### P3 — 长上下文验收(每个模型 1–2 次启动)

- 32K / 128K / 262K 提示的 TTFT、峰值显存、是否成功;
- 输出 4–8K 时该上下文下的 decode 速率;
- 记录 `Maximum concurrency` 与实测可用并发的差距。

### P4 — 报告与固化

1. `docs/TUNING_REPORT.md`:两个模型的**推荐参数表**(可直接复制粘贴的命令)、
   "为什么是这些值"的证据链、与生产参考(≈100 tok/s)的对比、剩余优化空间;
2. `report/tuning/` 原始数据 + `scripts/tune_*.sh` 复现入口;
3. 更新 `docs/RUNBOOK.md` 的参考命令(指向报告)、`ROADMAP.md`/`BACKLOG.md` 的待办与修订记录。

## 4. 风险与对策

| 风险 | 对策 |
|---|---|
| 262 144 上下文装不下(KV 或权重) | 依次尝试:KV fp8 → 降 `--gpu-memory-utilization` 之外的手段(减 offload/加卡)→ 回落到可行的最大上下文并明确记录 |
| 单次启动太慢,扫点成本高 | 优先做"不需要重启"的维度;用 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE` 运行期切阈值;同一次启动里扫完并发与输出长度 |
| CUDA graph 与 CPU host 回调冲突 | 先 `--enforce-eager`,再单独试关掉;若崩溃则记录为已知限制 |
| MTP 投机在主线未接 | 先探测(`--speculative-config`),不支持就记为未验证 |
| 共享机器其它负载干扰 | 每次测量记录 `uptime`/GPU 占用;必要时重测 |
| 长输出测试时间失控 | 生成长度固定 4–8K,不用 128K(128K 只作为"支持"而非测试项) |

## 5. 与后续优化起点的关系

本轮产出的"最优参数"是**后续优化的零点**:之后任何内核/调度改动,都在同一组参数、
同一套 ShareGPT 协议下对比,避免"参数变了却当成优化收益"。
