# 生产环境 8070(固化规格)

> ⭐ **本文件是生产口径的唯一真源** ✓ —— 需要恢复生产时,**只照这份 + `scripts/bringup_prod_8070.sh` 做**,
> 不要再翻旧文件、不要凭印象猜 ✗(2026-09-30 起因:服务器重启后我照了一份**过时**的记录启动 ✗,
> 漏了 LMCache、用错 MAXLEN、拼错 env 名 —— 用户连续纠正三次 ✓)

---

## 1. 口径(用户 2026-09-30 明令)

**DeepSeek-V4.1-Flash · 768K 上下文 · GPU 预填 on · dspark on · LMCache on · 监控栈 on**

## 2. 一键恢复(机器重启后照这条做)

```bash
cd /home/user/lvllm/vllm-xiaotu-moe
bash scripts/bringup_prod_8070.sh            # 起全套(LMCache → vLLM → 认领真 PID → 监控栈)
bash scripts/bringup_prod_8070.sh --status   # 只看状态(只读 ✓)
```

⚠️ **8070 是本 agent 自身的推理后端** ⇒ **重启它会打断当前会话** ✗(AGENTS.md"自服务环境纪律" ✓)。
本脚本只用于**机器重启后的恢复** ✓;日常调试请用 **GPU2 + TP=1** ✓。

## 3. 参数表(唯一真源;脚本 §"参数" 段与之一致 ✓)

| 参数 | 值 | 为什么是这个值 |
|---|---|---|
| served name | `DeepSeek-V4.1-Flash` | 正式全称(旧别名已废弃,B159)|
| `MAXLEN` | **786432**(768K) | 1M + GPU 预填 **实测必 OOM** ✗(历史两次,峰值 98%);`serve_v41.sh` 有护栏会**拒绝启动** ✗ |
| `MBT` | 4096 | 激活工作区 ∝ MBT;4096 是"给 KV 留得下"的口径 |
| `MAXSEQS` | 1 | 单路 768K |
| `GPUS` / `TP` | 0,1 / 2 | 生产占 GPU0+GPU1(GPU2 空闲可调试 ✓)|
| `GPU_UTIL` | 0.90 | — |
| `COMPILE` / `EAGER` | 1 / 0 | `FULL_DECODE_ONLY` + compile(不加 `--enforce-eager`)|
| `SPEC` | 1 | **dspark k=5** |
| `KV_DTYPE` | `fp8_ds_mla` | **bf16 KV 的 1M 极限配置已证伪**(B162 OOM)|
| `KV_CACHE_BYTES` | **3221225472**(3 GiB) | 768K + LMCache 放不下 5.6 GiB 那套 |
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | 384 | **GPU 预填门槛**(0 或不设 = 关闭 ✗)|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | **必须带** ✓ |
| `EXTRA_ENV` | `XIAOTU_GP_ACT_RESERVE_GIB=1.5` | ⚠️ 变量名是 **`RESERVE`** 不是 `RESERVED` ✗ |
| `LMCACHE` | **1** | 口径的一部分,不能省(用户明令)✓ |
| `WARMUP` | 1 | 预热 8192 / 32768 两个形状 |
| `PORT` / `TAG` | 8070 / `v41_8070` | 生产端口**固定 8070**,不随模型漂移 |

**解析器参数**(`serve_v41.sh` 已内置默认 ✓ ⇒ 起服务时无需手写 ✓):
`--enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v3
--default-chat-template-kwargs {"thinking":true}`(依据 B160 ✓)

## 4. LMCache 服务端(必须先起 ✓)

```bash
CHUNK_SIZE=2176 TRANSFER_MODE=lmcache_driven ENABLE_MODULES= L1_GB=100 L2_GB=100 bash scripts/serve_lmcache.sh
```
| 参数 | 值 | 依据 |
|---|---|---|
| `CHUNK_SIZE` | **2176** | 必须是该模型 vLLM block 的倍数;V4.1 需 64 的倍数、GLM 需 2176 ⇒ **2176 = 64×34 同时满足** ⇒ 一个服务端服务两个模型(B172/B198)|
| 端口 | 5555(MP)/ 8080(HTTP) | connector 默认 `tcp://localhost:5555` |
| `L1_GB` / `L2_GB` | 100 / 100 | L1=CPU 内存层;L2=`fs_native` 磁盘(重启后 prefix cache 仍在 ✓)|

## 5. 监控栈(口径要求 on ✓)

| 服务 | 端口 | 启动 |
|---|---|---|
| Prometheus 2.45.6 | 9090 | `proc.sh spawn prometheus …/prometheus --config.file=…/prometheus/prometheus.yml --storage.tsdb.path=…/prometheus/data --web.listen-address=127.0.0.1:9090 --web.enable-lifecycle` |
| Grafana 11.4 | 3000 | `proc.sh spawn grafana …/grafana-v11.4.0/bin/grafana server --homepath …/grafana-v11.4.0 --config …/grafana/grafana.ini` |
| node_exporter 1.8.2 | 9100 | `proc.sh spawn node_exporter … --web.listen-address=127.0.0.1:9100 --collector.textfile.directory=…/textfile` |
| xtu_exporter(自写) | 9100 textfile | `proc.sh spawn xtu_exporter <py> …/textfile_exporter.py …/textfile/xtu.prom` |
| coremap / coremap_png | 8787 | `proc.sh spawn coremap <py> …/web/serve.py` / `coremap_png <py> …/coremap_png.py …/web/coremap.png` |

**看板**:`http://127.0.0.1:3000/d/dsh-overview`

## 6. ⭐ 真 PID 与日志(用户 2026-09-30 定的策略)

⚠️ **`proc.sh spawn dsv41_prod …` 的 PID 文件记的是【包装脚本】** ✗ —— `serve_v41.sh` 内部
`nohup` 起真服务,包装脚本**立刻退出** ⇒ `proc.sh status dsv41_prod` 会显示 `NOT RUNNING` ✗
(我因此一度误判"服务死了" ✗ ⇒ 其实还在加载 ✓)。

**正确做法**(脚本已自动做 ✓):
```bash
APIP=$(ss -ltnp | grep ':8070 ' | grep -oP 'pid=\K[0-9]+' | head -1)   # 端口派生真 PID ✓
bash scripts/proc.sh adopt vllm_prod_8070 "$APIP"                       # 记进 PID 文件 ✓
# 并把真服务日志路径写进日志头 ✓:
#   [adopt] real_service_log=dev-docs/report/tuning/logs/v41_8070.log
```
* 停服务用 **`proc.sh stop vllm_prod_8070`** ✓(按 PID 文件,不按名字 ✗)
* **真服务日志** = `dev-docs/report/tuning/logs/v41_8070.log` ✓
  (包装日志 `dsv41_prod.log` 只有启动几行,且它的"预热完成"是**假信号** ✗ —— 那时 API 还没起 ⇒
   `Successful requests: 0` ✗)
* **判断"是否在推进"** 的正确方式 ✓:`wc -l v41_8070.log` 是否增长 + `ps -o %cpu,stat` 看 worker 是否 `R`/`D` ✓

## 7. DSH 侧配置(`~/.dsh/profiles/web/cordis.patch.yml`;0.2 格式 ✓)

⚠️ **DSH 升级到 0.2 会删掉 `settings.yaml`** ✗(2026-09-30 实测:只剩若干 `.bak*` 与 `.imported` ✓)
⇒ 升级后必须按 **0.2 格式**重写 ✓(字段名变了 ✓)。

⭐⭐ **2026-10-07 又踩一次(重启生产后暴露)**:
`本轮运行失败 provider "epyc-a100-server" model "DeepSeek-V4.1-Flash" does not support reasoning effort "high"` ✗
* **根因**:迁移到 patch 层时**漏了 `reasoningEfforts`**;而这条自定义 route 在 pi-ai 目录里不存在
  ⇒ 能力只能是 `base?.reasoning ?? false` = **无** ⇒ `dsh-llm resolveCallWithInfo()` 对任何
  **显式**档位在**发请求之前**就抛 `UNSUPPORTED_REASONING_EFFORT` ✗
* **逐行依据**:`dsh-llm/lib/index.js` `resolveCallWithInfo` 2174-2192;
  `dsh-llm-pi-ai/lib/index.js` `resolveModelReasoning` 567-590 ✓
* **完整复盘**:`docs/EXPERIMENTS.md` **B335** ✓

```yaml
llm-pi-ai:
  providers:
    epyc-a100-server:
      apiKeyEnv: EPYC_A100_SERVER_API_KEY
      api: openai-completions
      baseURL: http://127.0.0.1:8070/v1
      reasoning: high            # ⭐ "Default" 档 = high(缺它 ⇒ Default 落成 off 的拼写 none)
      streamIdleTimeoutMs: 1500000
      models:
        - id: DeepSeek-V4.1-Flash
          name: DeepSeek-V4.1-Flash
          contextWindow: 524288   # = 生产 /v1/models 的 max_model_len(512K)
          input: [text, image]
          compat:
            thinkingFormat: openai          # ⭐ vLLM 只认【顶层 reasoning_effort】
            supportsReasoningEffort: true   # ⭐ 缺它 ⇒ 选了档位也不改变请求
            supportsDeveloperRole: false    # ⭐ 缺它 ⇒ system 被改成 developer ⇒ system prompt 被丢
          reasoningEfforts:                 # 键 = UI 档位;值 = 发给 API 的拼写
            "off": none
            minimal: low
            low: low
            medium: high
            high: high
            xhigh: xhigh
            max: max
```

**实测(离线探针,零生产流量 ✓)**:`node dev-docs/dsh_wire_probe_v41.mjs` ⇒
UI 出现 7 档(`off…max`),每档送出的顶层 `reasoning_effort` 依次为
`none/low/low/high/high/xhigh/max`,且 **system 消息未被改成 `developer`** ✓

**0.2 与旧格式的差别(踩过 ✓)**:
| 项 | 旧 | **0.2** |
|---|---|---|
| 多模态字段 | `inputModalities` | **`input`** ✓ |
| 推理能力 | `reasoning: true` + `thinkingLevelMap` | **`reasoningEfforts`** ✓(旧名会被 schema **静默丢弃** ✗)|
| 选择器等级来源 | — | **`reasoningEfforts` 的【键】** ✓ |
| 送出的拼写 | — | **`reasoningEfforts` 的【值】** ✓ |
| 默认档 | — | **route 级 `reasoning`** ✓(省它 ⇒ Default 落成 off 的拼写)|
| 是否真的发 effort | `compat.supportsReasoningEffort` | 同名,**必须显式 `true`** ✓ |
| system 角色 | — | **`compat.supportsDeveloperRole: false`** ✓ |

**思考强度映射(关键 ✓)**:DSH 的等级键 = `off/minimal/low/medium/high/xhigh/max` ✓;
而 **V4.1 只认 `low/high/xhigh/max` / `none` / 整数 1–100** ✓ —— **没有 `medium`** ✗
(发 `medium` 直接 **400**,B160 ✓)⇒ **`medium` 必须映射成 `high`** ✓(上表已如此 ✓)。

⚠️ **为什么 `thinkingFormat` 用 `openai` 而不是 `deepseek`** ✓(B335 实测):
`deepseek` 分支会发 `thinking:{type:…}` 且**在 off 档不发 `reasoning_effort`** ✗ ——
而 vLLM 没有顶层 `thinking` 字段(其 `OpenAIBaseModel` 是 `extra="allow"`,只会静默忽略)
⇒ **Off 档会静默失效** ✗ ⇒ 改用 `openai`(只发顶层 `reasoning_effort`)✓

⚠️ **YAML 陷阱** ✓:`off` 作**键名必须加引号** ✗ 否则 YAML 1.1 会解析成**布尔 `False`** ✓
(`on/yes/no` 同类)⇒ 写成 `"off": none` ✓。

**⭐ 改完必须两步都验(缺一不可)** ✓:
```bash
bash scripts/check_dsh_settings.sh     # ① 配置自检(0.2 字段级,缺失即 ❌)
node dev-docs/dsh_wire_probe_v41.mjs   # ② 离线上线行为(假 endpoint,不碰生产)
```
**生效方式**:patch 层由 dsh 监听;**若前端模型目录仍是旧能力,点一次「刷新」**;
仍不行再重启 `dsh web` ✓ —— ⛔ **不要因此重启 8070**(那是本 agent 自己的推理后端,一次 ≈40 分钟)✗

**vLLM 侧**已是默认 ✓(`serve_v41.sh` 的 `TOOL_PARSER` / `REASONING_PARSER` / `DEFAULT_CHAT_KWARGS` ✓)。

## 8. 状态自检(一条)

```bash
bash scripts/bringup_prod_8070.sh --status
```
应看到:8070 / 5555 / 8080 / 9090 / 3000 / 9100 / 8787 **全部 RUNNING** ✓,
真 PID 文件有值 ✓,`/v1/models` 返回 `DeepSeek-V4.1-Flash` ✓。

## 9. 冷启动与容量(记录)

* 就绪约 **3–6 分钟**(权重 48 分片 + engram 表 47 GiB/rank 卸载到 pinned host ✓)
* 启动后 GPU ≈ **23 GiB/卡**;首次预填后 staging 驻留 ⇒ 峰值更高(GPU2 空闲 ✓)
* 冷启动首个请求可能 **~19 s**(CPU 专家权重页缓存冷);热后 ~0.4 s


---

## 10. ⭐ 已知限制与诊断(2026-09-30 实测,别再猜 ✓)

### 10.1 LMCache 的命中规则:**只按"逐字节完全相同的前缀"命中** ✓

**实测证据** ✓(同一 prompt 连发两次,vLLM 侧 `usage.prompt_tokens_details.cached_tokens` 是权威 ✓):
```
第 1 次: prompt_tokens=5235  cached_tokens=0        ← 首次全未命中
第 2 次: prompt_tokens=5235  cached_tokens=5120     ← 命中 97.8% ✓
LMCache 侧: Retrieved 609,280 tokens in 0.040 s     ← 60 万 token 检索仅 40 毫秒 ✓
```
⇒ ⇒ **服务重启后"又等 20 分钟"不是 LMCache 失效** ✗,而是**新会话的 prompt 开头就不同** ✗ ——
**第一个分叉 token 之后的所有 chunk 全部失效** ✓ ⇒ 整段上下文重新 prefill ✓。

**为什么分叉这么致命** ✓:LMCache 按 **chunk**(本机 `chunk_size=2176` ✓,blake3 ✓)存/取,
命中要求从第 0 个 chunk 起**连续**一致 ✓ ⇒ 开头差一个 token ⇒ 后面全部落空 ✓。

**看命中率的三个计数器** ✓(`/metrics` ✓):
```bash
curl -s --noproxy 127.0.0.1 http://127.0.0.1:8080/metrics | grep -E "lookup_(requested|hit)|hit_l[12]"
```
本次实测 ✓:`lookup_requested_tokens=6,238,592` / `lookup_hit_tokens=3,579,520` ⇒ **命中率 ≈ 57.4%** ✓;
⚠️ `lookup_hit_l1_tokens=0` ✗ ⇒ **重启会清空 L1(CPU 内存层)**,之后全部走 L2 磁盘 ✓(慢于 L1,但远快于重算 ✓)。

**改善** ✓:**让新会话的开头尽量逐字节不变**(只**追加**,不要**重排/改写** system prompt、技能清单、handoff 等开头内容 ✓)
⇒ 直接抬高命中率 ✓。

### 10.2 ⭐ 为什么 `MBT=4096` 时 GPU 预填【比纯 CPU 还慢】(实测 + 原码)

**结论先给** ✓:本配置下 ~154 tok/s 是**正常但吃亏**的值 ✗ —— **GPU 流式有「每 chunk ~5.6 s」的固定代价**,
而 `MBT=4096` 把 768K 切成 **192 个 chunk** ✗ ⇒ 5.6 s × 192 ≈ **18 分钟** ✓ ✓。

**证据链(全部仓库原文 ✓)**:

| 事实 | 出处 |
|---|---|
| GPU 流式在 **≤4K token 时耗时基本不变(≈5.6 s)**,之后近似线性 | `docs/BENCHMARKS.md` §3 |
| `MBT=4096` ⇒ **154.5 tok/s**(4 chunks / 16,384 token / TTFT 106.1 s) | `docs/EXPERIMENTS.md` **B2** MBT 曲线 |
| **纯 CPU** ≈ **3.9 ms/token = ~256 tok/s**("13.8K × 3.9 ms ≈ 54 s")| `docs/BENCHMARKS.md`(V4.1 修复前后对照) |
| **~2000 tok/s 是「纯 MoE 核级」吞吐**(1,918 / 1,994,不计注意力/采样)| `docs/BENCHMARKS.md` §3 |
| **1,725 tok/s** 那组的前提是"**能吃满 chunk**" | `docs/BENCHMARKS.md`(原文:"V4.1-Flash 能吃满 chunk,所以是另一个量级") |
| 每层装配 **207 ms**(权重 H2D 3.62 GB/rank)⇒ **装配占 90%**;隔离环境 134.9 ms = **26.86 GB/s**(本机 H2D 天花板)| `docs/BENCHMARKS.md` |

⇒ ⇒ ⇒ **算式**:`MBT=4096` 时 `154 tok/s`(= 6.5 ms/token)✗ **慢于**纯 CPU 的 256 tok/s ✓
⇒ **所以"GPU 预填 ACTIVE"并不等于"更快"** ✗ —— 它在本 MBT 下是净亏 ✓
(日志里**没有** `GPU prefill DISABLED for this process -> staying on CPU` ✓ ⇒ 路径没被拒 ✓,
是**路径本身在 4096 这一档不划算** ✓ —— 原码 `mixed_experts.py:1628` 就是那句回退提示 ✓)

**⚠️ 我之前写错的两处,已撤销** ✗:
1. ✗ "staging 与 MBT 无关 ⇒ 加大 MBT 几乎免费" —— **错** ✓:`staging_bytes()` 里确实没有 MBT ✓,
   但 **vLLM 的激活工作区 ∝ MBT** ✓(`scripts/serve_v41.sh:121` 原文:"激活工作区 ∝ MBT")⇒ **加大 MBT 要吃显存** ✓(你说得对 ✓)
2. ✗ "像 CPU 预填是 PCIe 的天然性能" —— **不准确** ✓:准确说法是**每 chunk 的固定代价摊不薄** ✓(见上表 ✓)

**⚠️ 另一条你点出的**:`max_num_seqs` **成比例吃 KV** ✓ —— 原码 `vllm_xiaotu_moe/vram_policy.py:195`:
`kv_gib = kv_per_m × (maxlen/1e6) × max_num_seqs` ✓ ⇒ **KV 需求线性 ∝ 并发数** ✓
⇒ 所以 `MAXSEQS=1` 是**有意为之** ✓(它不是配置错误 ✓,是锁定配置的一部分 ✓)。

### 10.2.1 可选的改进(⚠️ 需用户决定;重启 8070 会打断当前会话 ✗)

| 方案 | 做法 | 代价 / 前提 |
|---|---|---|
| **A ⭐ 最省** | 把门槛提到 **> MBT**(如 `MBT=4096` + `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=8192`)⇒ **GPU 预填永不触发** ⇒ 走纯 CPU(**~256–310 tok/s**)⇒ 比现在(154)**≈ 快 2×** | **零显存代价** ✓;只是放弃 GPU 流式 ✓ |
| **B** | **加大 MBT**(8192 / 16384)⇒ chunk 数 192→96/48 ⇒ 5.6 s 摊薄 ⇒ 才可能接近 1,000+ tok/s | ⚠️ **激活工作区 ∝ MBT** ⇒ 要吃显存 ✗;本配置峰值已 86.4%、余量仅 ~5.5 GiB ⇒ **必须实测**,可能 OOM ✗;⛔⛔ **本条已于 2026-10-09 被用户【否决】:不再测 `MBT=16384`、也不再往上推 MBT**(理由:**显存宝贵、继续扩大的收益不明显**)⇒ **方案 B 作废** ✓ |
| **C** | 缩 **staging 字节**(换更小位宽权重 / 少常驻)⇒ 直接降每 chunk 固定代价 | 依台账,这是**唯一已量化**的杠杆 ✓ |

⇒ ⚠️ **判别方法**:改完先用 `[fp8-asm]`(需 `_trace` ✓)或 `Avg prompt throughput` 实测复核 ✓ ——
**"GPU prefill ACTIVE" 不能单独当"更快"的判据** ✗(本次就是反例 ✓)。

### 10.3 诊断速查

| 想知道 | 命令 |
|---|---|
| 服务/监控是否都在 | `bash scripts/bringup_prod_8070.sh --status` |
| GPU 预填是否真启用 | `grep "GPU prefill ACTIVE" dev-docs/report/tuning/logs/v41_8070.log \| tail -1`(看 **slack 正负** ✓) |
| 缓存命中率 | `curl -s --noproxy 127.0.0.1 http://127.0.0.1:8080/metrics \| grep -E "lookup_(requested\|hit)"` |
| 单次请求命中多少 | 看响应 `usage.prompt_tokens_details.cached_tokens` ✓ |
| 预填吞吐 | `grep "Avg prompt throughput" dev-docs/report/tuning/logs/v41_8070.log \| tail -5` |

---
*相关记录:`docs/EXPERIMENTS.md` B142–B214(生产配置演进)、B160(思考强度配方)、B198(LMCache chunk)、B214(DSH 配置字段)。*
