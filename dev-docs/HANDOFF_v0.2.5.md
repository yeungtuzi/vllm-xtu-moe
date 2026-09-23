# HANDOFF —— 从 0.2.4 之后继续(vLLM-XTU-MoE / `xiaotu_moe` 引擎)

> 面向**下一个会话**的交接。写于 **2026-09-23**,本会话主题:**让 staging 仪表可信 → 归因 → 定位 GPU 预填的真实门槛 → 1M 上下文下把 GPU 预填做出来**。
> 内部文档(`dev-docs/`,`.gitignore`)。对外只在 `docs/` 与 `RELEASE_NOTES_*.md`。

---

## 0. 三十秒速览

| 项 | 状态 |
|---|---|
| **🔴 当前生产(8700)是一个"极限显存"配置** | `1M + GPU 预填(384) + FULL_DECODE_ONLY+compile + KV 5.6 GiB + 激活预留 1.5 + dspark k=5 + seqs 2`;**显存 ≈40.1/41.0 GiB(98%)**;用户正在**持续观察其稳定性** ⇒ **未经允许不要重启/挪动它** |
| **显存风险** | staging 常驻显存;**长 prompt 的激活峰值**是 OOM 触发点。不稳就按 §4.3 回退到**固化默认 A**(CPU 预填 + 全 eager + KV 6 GiB,低风险) |
| **1M + GPU 预填=本会话成果** | 预检 slack **+0.29 GiB** ✅(此前被判定"放不下" ✗)。关键三步:①`FULL_DECODE_ONLY` 省 **9.6 GiB/卡**;②激活预留 `3.0→1.5`(需求 14.80→**13.30**);③KV `6.0→5.6 GiB`(空闲 13.19→**13.59**) |
| **1M 的 CPU 预填实测量级** | **≈3.3 ms/token**(512→1.76 s、4096→13.8 s、16384→53.9 s、32768→~107 s) |
| **GPU 预填的判据(用户给的,很好用)** | 小请求应出现**与尺寸无关的固定开销**、大请求应**反超 CPU**;否则就是**没走 GPU**。256K 档实测:L=512 **9.03 vs 1.76 s(+7.27 固定)**;L=8192 **24.70 vs 27.11**;L=16384 **49.85 vs 53.85**;交叉点 **≈10K token** |
| **图模式结论(经用户纠正)** | `FULL_DECODE_ONLY+compile` 的聚合吞吐比全 eager **低 30%** ✗,但**这是误导**:p50 反而更好(TTFT 923 vs 954 ms、E2EL 3951 vs 5462 ms),劣势全在 p99(E2EL 82.8 s vs 11.1 s)⇒ **一次性编译/捕获代价,长跑会摊掉** ✓ |
| **README 性能表** | **本次未改** ✓(新配置在标准四格上不是更优;且协议用 `random` 数据集,**对投机是最坏情况**,与基线"投机关"不可比) |
| **两个我犯过的错(已记录)** | ①"B 慢 30%"—— 停在聚合均值上,分位数就在手里 ✗;②`RAW2_SHARE` 只存在于 **FP8 路径**,对 V4.1(MXFP4)**无效** ✗ |
| **YaRN factor** | 决定**不动**(改 factor 的 dict 覆盖**不会传播到投机草稿** ⇒ dspark 风险)。详见 §3.7 |
| **发布** | **v0.2.4 已发行** ✓;**0.2.5 未发布**,待收:同步化仪表 + KV 常量根因 + `serve_v41` 修复与固化默认(§6) |

---

## 1. 本会话产物(插件仓库 `/home/user/lvllm/vllm-xiaotu-moe`)

### 1.1 代码/脚本(已提交并推送,`main`)

| commit | 内容 |
|---|---|
| `b1cf8a3`/`7aae250` | **`gpu_prefill._stage_mark` 改为无 sync 的 CUDA event**(原实现 `perf_counter + torch.cuda.synchronize()` 把 `[gpf-stage] dma` 报成 **537 ms/层**,真值 **165–205 ms**,**虚高 3.2×** ✗);与 `[fp8-asm]` 交叉验证 **±2%**(164.9↔166.8、203.0↔205.0 ms)✓ |
| `0a3d3ab`→`052b140`→`c4c75d2` | 阈值默认值的一段弯路(384↔4096),最终**保持 4096** 并写明依据;`c4c75d2` 修掉我在这段改动里弄坏的缩进 |
| `51fd931` | **`vram_policy.KV_GIB_PER_MTOKEN` 1.290154 → 5.04**(实测);KV 下限/夹取逻辑重写 ⇒ 131072/524288/1048576 分别输出 0.66/2.64/**5.28 GiB** ✓ |
| `90775bd` | `serve_v41.sh` 新增 **`HF_OVERRIDES`** 透传(脚本头写明"dict 不传播到草稿 ⇒ 改 RoPE 前必须验接受率") |
| `1bb79b7` | **固化"本机 + V4.1-Flash 默认配置"**进 `serve_v41.sh` 默认值(见 §4.2)+ `MODEL_GUIDES §0.1b-2` |
| `77fb29d` ⭐ | **修掉 `1bb79b7` 弄坏的续行链** —— 详情见 §5.2 第 1 条(**任何启动都会失败**,而 `bash -n` 查不出) |
| `044d712` | 图模式 A/B(EXPERIMENTS **B156**) |
| `af7b965` | 1M + GPU 预填的成因与"README 不改"的理由(EXPERIMENTS **B157**) |

### 1.2 台账与文档

* `docs/EXPERIMENTS.md`:**B142**(无 sync 仪表交叉验证)、**B144**(停滞归因:stall = 一次携带 prefill 的 pass)、**B145**(KV 根因修复实证)、**B146/B147**(MBT 杠杆:2 chunk→1 chunk **−33%** prefill)、**B148**、**B149/B150**(阈值 2860 的出处与"两臂打平"的**无效**结论)、**B151**(用户判据 + 1M 显存账 + 第三卡拓扑)、**B152**、**B153**(YaRN 调查)、**B154**(512K 放不下 GPU 预填)、**B155**(固化默认)、**B156**(图模式 A/B + 两条流程纪律)、**B157**(1M+GPU 预填做成了 / README 不改 / 两个错)
* `docs/MODEL_GUIDES.md` **§0.1b-2**:本机 + V4.1 默认配置表 + **三个易错点**
* `scripts/serve_v41.sh`:默认值 + `HF_OVERRIDES`

### 1.3 未合并分支(供参考,未采用)

`diag/gpf-events`(通用路径的 event 计时 + 队列收割)、`diag/h2d-real-time`(读预取槽 event)、`fix/vram-kv-floor`(已被 `51fd931` 取代,stale)。

---

## 2. 当前生产环境(★ 用户正在用/观察,**默认不动**)

### 2.1 8700 上正在跑什么(2026-09-23 起)

| 项 | 值 |
|---|---|
| 模型 / 端口 | `dsv41`(DeepSeek-V4.1-Flash)/ **8700** |
| MAXLEN / seqs / TP | **1,048,576** / 2 / 2(GPU 0,1) |
| **GPU 预填** | **开(门槛 384)** —— 预检 `slack +0.29 GiB` ⇒ ACTIVE(不是 DISABLED)✓ |
| 图模式 | **`FULL_DECODE_ONLY` + `VLLM_COMPILE`**(即 `COMPILE=1 EAGER=0`) |
| KV | `--kv-cache-memory 6012954214`(5.6 GiB)⇒ 池 **2,962,508 token** |
| 激活预留 | `XIAOTU_GP_ACT_RESERVE_GIB=1.5`(默认 3.0) |
| 投机 | **dspark k=5** ✓ |
| MoE 常驻层 | **无**(§3.6 说明为什么 1M 下不值得) |
| **显存** | **≈40.1 / 41.0 GiB(≈98%)** ⚠️ |

启动命令(如需重建,**同一时刻只能有一个服务占 GPU0/1**):

```bash
cd /home/user/lvllm/vllm-xiaotu-moe
setsid env MAXLEN=1048576 MBT=4096 MAXSEQS=2 GPUS=0,1 TP=2 GPU_UTIL=0.90 LOAD=auto \
  COMPILE=1 EAGER=0 SPEC=1 KV_CACHE_BYTES=6012954214 \
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384 \
  EXTRA_ENV="XIAOTU_GP_ACT_RESERVE_GIB=1.5" \
  WARMUP=1 PORT=8700 TAG=dsv41_prod_extreme \
  bash scripts/serve_v41.sh > /tmp/extreme_launch.log 2>&1 < /dev/null &
# 就绪判定:curl -sf http://127.0.0.1:8700/v1/models(约 5 分钟)
# 日志:dev-docs/report/tuning/logs/dsv41_prod_extreme.log
```

### 2.2 观察它的稳定性(用户当前任务)

| 信号 | 判据 |
|---|---|
| **OOM** | 日志出现 `OutOfMemoryError` / `CUDA out of memory`(典型触发:**长 prompt 的激活峰值**)|
| **引擎死** | `EngineDeadError`、`Executor failed`、或**进程/端口消失** |
| **堆积** | 周期统计 `Running:` 长期 >2、或 `Waiting:` 持续 >0 |
| **显存** | `nvidia-smi` 已 ≈98% ⇒ 任何额外分配都危险 |

日志:当前服务的 `[v41] starting ... log=<路径>` 一行会打印路径(**别猜**)✓

### 2.3 回退(不稳就执行;这是"固化默认 A",低风险)

```bash
cd /home/user/lvllm/vllm-xiaotu-moe
P=$(ss -ltnp | grep ':8700' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2); [ -n "$P" ] && kill -9 "$P"
sleep 10; for x in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$x"; done
sleep 8; bash scripts/serve_v41.sh          # 固化默认 = 1M + CPU 预填 + 全 eager + KV 6 GiB + dspark
```

---

## 3. 本会话关键实测(全部可复现)

### 3.1 staging 的真实成本(仪表修好之后)

| 项 | 值 |
|---|---|
| 每层 staging | **165–205 ms**(w13 107–134、w2 53–65、tr ~6) |
| 等效带宽 | **17.7–22.0 GB/s ≈ 本机 PCIe 峰值 26.18 GB/s 的 70–85%** |
| ⇒ 含义 | **"重叠/加缓冲"这条路余量很小**(≤15–30%);真正的大杠杆是**减少 chunk 重复** |
| 反例归档 | 旧仪表报 **537 ms/层** ⇒ 据此算出的"3.8× 差距、每步省 17.5 s"**全部作废** ✗ |

### 3.2 MBT 杠杆(B147)

8000-token prompt、预热 + 每请求唯一 nonce:MBT 4096(2 chunk)**51.7 s** → MBT 8192(1 chunk)**34.7 s** = **−33%** ✓
(16384/32768 在 GLM 131072 上下文下 OOM ⇒ 该杠杆有天花板)

### 3.3 KV 根因(B145)

`KV_GIB_PER_MTOKEN = 5.04`(实测)⇒ 131072/524288/1048576 分别 **0.66 / 2.64 / 5.28 GiB** ✓
⚠️ **显式 `--kv-cache-memory` 时 vLLM 会额外要一块同尺寸的临时 buffer**(`v1/worker/utils.py` 的 `torch.zeros`)⇒ 1M 且 seqs 大时必须**显式给预算**,否则 OOM。
⚠️ **同一条路径下 `--gpu-memory-utilization` 完全不起作用**(vLLM 日志:`skipped memory profiling`)⇒ 想腾显存只能真减 KV/staging ✓

### 3.4 GPU 预填判据(用户给的)与 V4.1 曲线

| L | 256K + GPU 预填 | 1M + CPU 预填 | 判据 |
|---|---|---|---|
| **512** | **9.03 s** | 1.76 s | ✅ **+7.27 s 固定开销** |
| 2048 | 10.41 | 7.10 | — |
| 8192 | **24.70** | 27.11 | ✅ 反超 |
| 16384 | **49.85** | 53.85 | ✅ 反超 |

* 交叉点 **≈10K token**;GPU 每-token 斜率 2.57 ms vs CPU 3.28 ms。
* **1M + GPU 预填(本次做成)**:门槛 384 生效,预检 `slack +0.29 GiB`,长探针通过,**0 DISABLED、0 崩溃** ✓

### 3.5 1M 的显存账(为什么差一点点、又是怎么凑齐的)

单卡 40.5 GiB;非专家权重 **19.6** + KV(6.0)

| 配置 | 预检需求 | 实测空闲 | slack |
|---|---|---|---|
| A(全 eager)+384 | 14.80 | 10.1 | **−4.70** ✗ |
| B(FULL_DECODE_ONLY)+384 | 14.80 | 13.19 | **−1.61** ✗ |
| B + 激活预留 1.5 | **13.30** | 13.19 | −0.11 |
| **B + 预留 1.5 + KV 5.6** | 13.30 | **13.59** | **+0.29** ✅ |

⚠️ **`RAW2_SHARE` 不可用于 V4.1**:它只在 **FP8** 路径(`gpu_prefill_fp8.py`,6 处);V4.1 走的 **MXFP4** `gpu_prefill.py` **0 处**(实测 staging 仍 10.73 印证)✗

### 3.6 MoE 常驻层:1M 下**不值得**(用户已决定不放)

* 每层 **3.36 GiB / 层 / rank**(**每个 rank 持整层副本**,不是 TP 分片的一半)
* 1M 配置**稳态**空闲仅 **4.32 GiB** ⇒ 按 20% 安全余量最多 **1 层**(且只剩 ~1 GiB)
* ⚠️ **算驻留必须用稳态空闲**;预检时刻报的 10.66 GiB 是**瞬时值** ✗
* 插件**自己**保留少量层给 dspark 草稿(日志里的 `GPU-resident` 行,约 8 个层×rank)⇒ 属正常,勿误认为"常驻没关"

### 3.7 YaRN / 上下文扩展(B153)

* V4.1 `config.json`:`rope_scaling={rope_type: yarn, factor: **16**, beta_fast: **32**, beta_slow: **1**, original_max_position_embeddings: **65536**}`;`max_position_embeddings=1048576` ⇒ **原生 64K + YaRN×16 = 1M**,而 **4/8/16 ↔ 256K/512K/1M** ✓
* 我们的树里**没有** `--rope-scaling`;`--hf-overrides '{"rope_parameters":{...}}'` 是正确机制 ✓
* ⚠️ **dict 覆盖不会传播到投机草稿**(代码原文 "Dict overrides … are not applied to the draft")⇒ 上游 [#37435](https://github.com/vllm-project/vllm/issues/37435)、[#58080](https://github.com/vllm-project/vllm/issues/58080)(后者:draft 超长度后 **dummy-0 草案静默杀死接受率**);修复 PR #37443/#58094 **不在我们树里** ✓
* ⚠️ 覆盖时**必须带上 `beta_fast`/`beta_slow`**,否则退回默认值、YaRN 插值形状改变 ✗
* **决定:不动 factor**(dspark 风险 > 未证实的精度收益);要动就必须**同步改草稿的 config**,或改 factor 时**关投机** ✓

### 3.8 第三张卡(GPU2)

A100-**PCIE**,**无 NVLink**,三卡各在不同 NUMA:主机→GPU0 **19.54 GB/s**;**GPU2→GPU0 17.96 GB/s**;TP 对之间 **14.24 GB/s** ⇒ **跨卡不比主机快** ⇒ staging/KV/草稿放 GPU2 都不成立 ✗
⇒ **GPU2 的正解:单独跑"256K + GPU 预填"快档**(正是 `serve_prod_8070.sh` 的 `MODE=fast`,`GPUS=2 TP=1` 的设计)✓

### 3.9 标准四格复测(用"极限配置",只跑完短两格)

| 格 | 新配置(1M+GPU预填+dspark) | README 基线(256K·投机关) |
|---|---|---|
| 短 C=1 | prefill **239.5** / decode **15.4** | **254.3 / 19.8** |
| 短 C=2 | prefill **348.2** / decode **19.2** | **350.4 / 28.0** |
| 长 C=1 / C=2 | **未测完**(用户叫停) | 903.9/19.3、1204.2/15.4 |

⇒ **不是更优 ⇒ README 不改** ✓;且协议用 `random` 数据集**对投机最坏**(仓库注释早有此论断)⇒ 与"投机关"基线不可比 ✓
⇒ **公平做法(若将来要测)**:同样四格再跑一遍 **`SPEC=0`**,才与 README 口径一致 ✓

---

## 4. 操作实务

### 4.1 脚本与端口

| 脚本 | 用途 | 默认端口 |
|---|---|---|
| `scripts/serve_v41.sh` | V4.1-Flash(本机默认档) | **8700** |
| `scripts/serve_mimo26.sh` | MiMo-V2.6-Flash-RL | — |
| `scripts/serve_prod_8070.sh` | GLM 生产 / `MODE=fast(256K)/1m(1M)` | 8070 |
| `dev-docs/report/tuning/probes/p5_unified.sh` | 标准性能四格驱动 | — |

### 4.2 固化默认(裸跑 `bash scripts/serve_v41.sh` 即这套)

`PORT 8700 · GPUS 0,1 · TP 2 · MAXLEN 1048576 · MAXSEQS 2 · MBT 4096 · LOAD auto · GPU_UTIL 0.90 ·
EAGER 1(全 eager)· SPEC 1(dspark k=5)· KV 6442450944 · GPU 预填 **0(显式关)** · 常驻层不设`

* **要 256K + GPU 预填**:
  `MAXLEN=262144 KV_CACHE_BYTES=2147483648 VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384 bash scripts/serve_v41.sh`
* **要改 YaRN factor**:`HF_OVERRIDES='{"rope_parameters":{...,"beta_fast":32,"beta_slow":1,...}}'`(并先读 §3.7)⚠️

### 4.3 bench 协议(易错,照抄 P5)

```bash
PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python     # ⚠️ 裸 shell 里 vllm 不在 PATH
$PY -m vllm.entrypoints.cli.main bench serve \
  --backend openai --endpoint /v1/completions --host 127.0.0.1 --port 8700 \
  --model /home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master \
  --served-model-name dsv41 \
  --dataset-name random --random-input-len L --random-output-len 128 \
  --num-prompts 8 --max-concurrency C --seed $((L*7+C*131+17)) --skip-chat-template \
  --save-result --result-dir <dir> --result-filename <name>.json
```

* **口径**:`prefill = C × prompt_tokens / TTFT`、`decode = C × 1000 / median(ITL)` ✓
* **`--model` 给 checkpoint 路径(load tokenizer)**、**`--served-model-name` 给服务名** —— 只给服务名会 `OSError: Can't load the configuration of 'dsv41'` ✗
* 服务端取样:`curl -s http://127.0.0.1:8700/metrics`(679 行);配合日志周期统计(`Avg prompt/generation throughput`、`Running:`、`GPU KV cache usage`、`SpecDecoding metrics`)
* ⚠️ **一个服务同一时刻只跑一个 bench 循环**(`Running > max-concurrency` 说明重叠,数据作废)

---

## 5. 铁律与陷阱(本会话新增/复验)

1. **续行链里绝不能放注释** —— `cmd \` + 换行 + `# 注释` ⇒ 逻辑行被 `#` 吃掉 ⇒ **后续参数变成新命令**(报 `OMP_NUM_THREADS=1: command not found`)**而 `bash -n` 查不出** ✗。改脚本的**结构性校验**:链内无注释、除末行外每行以 `\` 结尾 ✓(`77fb29d` 的教训)
2. **启动必须同步读日志确认**,不能"发脚本→等结果" —— 本次多次因此误判(B 臂没起来却以为在跑)✓
3. **`GPU prefill ACTIVE` 不等于"已启用"**:它在**显存预检之前**打印,且**每 (层,rank) 只打一次** ⇒ **只看它尾巴的 `slack` 正负**(负值随后就是 `DISABLED ... staying on CPU`)✓
4. **预检的"空闲"是那一刻的瞬时值**,与**稳态**空闲可以差很多(本次 10.66 vs 4.32 GiB)⇒ 算容量/驻留必须说清用哪个 ✓
5. **裸跑脚本时 `TAG` 默认是 `v41`** ⇒ 日志在 `logs/v41.log`,不是你以为的名字 ✓(启动日志里那行 `log=...` 是权威)
6. **杀掉 API server 后 EngineCore worker 会残留并占着显存**(本次残留 11 分钟、占 ~32 GiB)⇒ 清理**按 PID + 核对 `nvidia-smi` 进程列表** ✓
7. **绝不要按名字匹配 PID**(`grep -E "...sweep"` 会匹配到自己的命令行 ⇒ 自杀,本次犯过 3 次)⇒ 只用**端口派生 PID 或 PID 文件** ✓
8. **显式 KV 时 `--gpu-memory-utilization` 无效**(vLLM 跳过显存剖析)✓
9. **改投机/图/RoPE 前先看接受率**:`Mean acceptance length` 是最灵敏的健康指标(1M 档实测 **3.50/3.67**;sharegpt 上 **2.16–2.54**;random 数据集上会很差 —— 属正常)✓
10. **一次只跑一个 bench 循环;预热 + 每请求唯一 nonce**(否则前缀缓存把结论毁掉)✓

---

## 6. 未完成 / 下一步(按优先级)

| # | 事项 | 依据/起点 |
|---|---|---|
| 1 | **观察当前极限配置的稳定性**(用户正在做);若不稳 → §2.3 回退 | §2.2 |
| 2 | **发布 0.2.5**:把本会话的修复打包(无 sync 仪表 + KV 常量 5.04 + `serve_v41` 的 `HF_OVERRIDES`/固化默认/续行修复 + 文档),写 RELEASE_NOTES、更新 README/CHANGELOG、打 tag | B142/B145/B155/B157 |
| 3 | **V4.1 的 GPU 预填门槛定值**:已验 256K 可行、512K 不可行,**256K–512K 之间的精确上限未测**(此前两次扫描因自身缺陷失败) | B150/B151/B154 |
| 4 | **长上下文下的 dspark 接受率**:现有数据来自 ~5K token 请求;**上游 #58080 的"draft 长度门"在我们栈上未验证** | B153 |
| 5 | **`FULL_DECODE_ONLY` 的真实收益**:冷 bench 对"前端编译成本"不公平 ⇒ 需要**长跑(形状遍历充分)后再比** | B156/B157 |
| 6 | **标准四格补测**:跑 **`SPEC=0`** 的四格(与 README 口径一致)才谈得上更新性能表 | §3.9 |
| 7 | (可选)预填 `pre` 阶段削减优化;GLM/MiMo 的 KV 标定(它们不走 policy) | 目标 2 剩余项 |

---

## 7. 现场关键路径

| 路径 | 内容 |
|---|---|
| `/home/user/lvllm/vllm-xiaotu-moe` | 插件仓库(本会话全部改动);`pip show vllm-xtu-moe` ⇒ Editable 指向此树 |
| `dev-docs/report/tuning/logs/` | 各次服务日志(`dsv41_1m_gpf_B3.log`、`dsv41_full_decode.log`、`v41.log` …) |
| `dev-docs/report/tuning/raw/` | bench 结果(`v41_new/`、`sg_c2_n100/`、`yarn_ab/` …) |
| `dev-docs/report/tuning/NOTES.md` | 长年笔记(**§1.2 CUDA graph 的 use-after-free 已修** 即出自此) |
| `docs/EXPERIMENTS.md` | 台账(B1xx,本会话 B142–B157) |
| `docs/MODEL_GUIDES.md §0.1b-2` | 本机 + V4.1 默认配置 + 三个易错点 |
| `/home/user/lvllm/vllm-mainline`(环境里的 `vllm`) | 上游主树(bench 客户端也用它) |
| 模型 | V4.1:`/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master` |
| 硬件 | 2×A100-40GB(0,1 供生产;2 空闲可用)+ 3 NUMA 域、PCIe、无 NVLink |

**本会话结束时的 git 状态**:`main` 干净、HEAD **`af7b965`**、已 push ✓;目标 2(`goal-d00b4ee0-5541-4cb4-b5f1-d2ccfe6bd946`,100 轮)仍 active。
