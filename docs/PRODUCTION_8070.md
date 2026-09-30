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

## 7. DSH 侧配置(`~/.dsh/settings.yaml`;0.2 格式 ✓)

⚠️ **DSH 升级到 0.2 会删掉 `settings.yaml`** ✗(2026-09-30 实测:只剩若干 `.bak*` 与 `.imported` ✓)
⇒ 升级后必须按 **0.2 格式**重写 ✓(字段名变了 ✓)。

```yaml
llm-pi-ai:
  providers:
    {
      epyc-a100-server:
        {
          apiKeyEnv: EPYC_A100_SERVER_API_KEY,
          api: openai-completions,
          baseURL: http://127.0.0.1:8070/v1,
          compat: { thinkingFormat: deepseek },
          streamIdleTimeoutMs: 1500000,
          models:
            [
              { id: DeepSeek-V4.1-Flash, name: DeepSeek-V4.1-Flash, contextWindow: 768000,
                input: [text, image],
                reasoningEfforts: { "off": none, minimal: low, low: low, medium: high,
                                    high: high, xhigh: xhigh, max: max } }
            ]
        }
    }
```

**0.2 与旧格式的差别(踩过 ✓)**:
| 项 | 旧 | **0.2** |
|---|---|---|
| 多模态字段 | `inputModalities` | **`input`** ✓(π-ai 插件用 `input`;`inputModalities` 是 DeepSeek 插件用的)|
| 推理能力 | `reasoning: true` | **改用 `reasoningEfforts`** ✓(省略=沿用目录;`false`=非推理)|
| 选择器等级来源 | — | **`reasoningEfforts` 的【键】** = 选择器提供的等级 ✓ |
| 送出的拼写 | — | **`reasoningEfforts` 的【值】** = 上线拼写 ✓ |

**思考强度映射(关键 ✓)**:DSH 的等级键 = `off/minimal/low/medium/high/xhigh/max` ✓;
而 **V4.1 只认 `low/high/xhigh/max` / `none` / 整数 1–100** ✓ —— **没有 `medium`** ✗
(发 `medium` 直接 **400**,B160 ✓)⇒ 所以 **`medium` 必须映射成 `high`** ✓(上表已如此 ✓)。

⚠️ **YAML 陷阱** ✓:`off` 作**键名必须加引号** ✗ 否则 YAML 1.1 会解析成**布尔 `False`** ✓
(`on/yes/no` 同类)⇒ 写成 `"off": none` ✓。

**vLLM 侧**已是默认 ✓(`serve_v41.sh` 的 `TOOL_PARSER` / `REASONING_PARSER` / `DEFAULT_CHAT_KWARGS` ✓);
**改完 settings.yaml 后 DSH 会热加载 ✓**(`watch` ✓)。

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

### 10.2 GPU 预填的真实瓶颈是 **PCIe H2D**,不是 GPU 算力 ✗

```
GPU prefill ACTIVE(首次 1468 tokens ≥ 阈值 384 ✓)
   preflight: staging ~10.73 GiB, required ≥ 13.30 GiB, had 16.96 GiB ⇒ slack 【+3.66 GiB】(正)✓
nvidia-smi: GPU0/1 = 100%,37,979 / 40,960 MiB(92.7%)
Avg prompt throughput ≈ 104–160 tokens/s   ⇒ 20 分钟 ≈ 16 万 token
```
* 专家权重常驻 **CPU 内存** ✓;GPU 预填时**逐层把权重 H2D 搬进 GPU staging** ✓(日志里的 `staging ~10.73 GiB` ✓)
* 通道 = **PCIe,无 NVLink**,本机上限 **≈26.86 GB/s** ✗ ⇒ 台账实测 `dma = 599.2 ms/层`,占单层 **98.9%** ✗
* ⇒ **GPU 的 100% 绝大部分是 DMA 搬运/等待** ✓ —— 所以"看起来像 CPU 预填性能",其实是 **PCIe 受限的 GPU 预填** ✓
* 同一原因解释了:**为什么显存占到 92.7%** ✓ 以及 **为什么 1M + GPU 预填必 OOM** ✗

**判据(本仓纪律 ✓)**:`GPU prefill ACTIVE` **不等于**已启用 ⇒ **必须看 slack 正负** ✓(本次 +3.66 GiB 为正 ⇒ 真启用 ✓)。

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
