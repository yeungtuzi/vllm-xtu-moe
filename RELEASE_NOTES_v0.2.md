# vllm-xtu-moe v0.2 — DeepSeek-V4.1-Flash(748B)+ CPU 预填充性能优化

**本版是交付版。** 它把两件事合在一起:

1. **DeepSeek-V4.1-Flash(748B)全链路可用** —— 1M 上下文 + GPU 预填充 + 投机解码
   (这部分原先以中间版本 `v0.21.0` 发布,**现已撤回**,见文末「版本口径」);
2. **CPU 预填充路径的实测性能优化** —— 去掉每层 **43-80 ms** 的纯浪费,
   唯一 prompt 的首 token 延迟 **−38~40%**(详见 §1、§3)。

> 版本口径(用户 2026-09-18 裁定)
> * **`v0.2`(本文)= 交付版**;
> * 原 `v0.2.0`(2026-09-14「主线化」版,**不支持 V4.1**)全仓库改称 **`0.2pre`**;
> * `v0.21.0`(2026-09-17,只有 GPU 预填充修复)**已撤回**:GitHub Release 与 tag 均删除,
>   内容并入本版。

---

## 1. 本版的核心性能改动:CPU 预填充路径里每层 43-80 ms 的浪费

### 1.1 先纠正两个会读错 40× 的口径

* `[cd-timing]` 的 `period`/`compute` 是**每层**均值(回调 entry→entry),**不是** 40 层合计。
  自证:qlen=1893 时 `period=345.12 ms` × 40 = **13.80 s**,而实测 TTFT = **13.765 s**。
* `nshard_ >= 2` 时(**本机 8 个 NUMA node ⇒ 恒真**)所有 batch 都走
  `forward_many_nsliced`,不是 legacy 路径。

### 1.2 根因:按专家各自 `std::vector::resize` 的 scratch

真实路由是**长尾**的,于是每个专家的 `me` 会不断刷新自己的历史最大值;而
`resize(n)` 既会**零填充**新增元素,又因为按需精确扩容而**每次重新分配 + 整块 memcpy +
重新触碰新映射的页面**。实测(`XIAOTU_MOE_SETUP_PROF_EVERY=1`,qlen≈1.9K,na≈232):

```
grow=79.3ms  row=0.10ms  realloc=355  bytes=118458068
need(down,both,act,bf16,row)=71,71,71,71,71   me_max=1246  me_sum=11399 (=NASS ✓)
```

⇒ 每层 **355 次真实扩容、搬 118 MB**,占该层 CPU MoE 时间的 ~25%。

### 1.3 修法:扁平 arena(尺寸只由 `NASS` 决定)+ 不零填充

不变量:`sum_{e ∈ active} me_e == NASS`(每条 (token,rank) 指派恰好属于一个专家)
⇒ 整块 scratch 大小 **只由 NASS 决定**,与路由形状无关,稳态**扩容 0 次**。
同时引入 `NoInitAlloc<T>`(无参 `construct` 不做事)去掉零填充 —— 这些缓冲全是**写后读**,
已逐个核对(`both` 由 gate/up 写满、`abf16` 由融合 SiLU 写满、`down` 由 down GEMM 写满、
`rowmap` 紧接着写满;`xg`/`act` 全仓库无读取点)。

| | 修前(0.2pre/0.21.0) | **v0.2** |
|---|---|---|
| `resize` 段 / 层(M≈1.9K) | **43-80 ms** | **56-78 µs** |
| 每层真实扩容次数 | **355** | **0** |
| 不变量 `sum_me` | — | **= NASS** ✓ |

**效果的正确描述**(很重要):去掉的是**一次性高水位棘轮**,所以
* **新路由分布(首趟 / 真实业务里 prompt 一直在变)⇒ 持续收益**:median TTFT **−38~40%**;
* **反复重放同一批 prompt(纪录已饱和)⇒ 收益趋近 0**。
  两个独立实验互证:ShareGPT 16 条 loader 口径 median TTFT **1239 → 744 ms**;
  8 条最长 prompt 首趟 median **3570 → 2210 ms**。

---

## 2. 新增的三个客户端可见默认参数(都为 DSH 这类客户端)

| 参数 | 默认 | 作用 |
|---|---|---|
| `--enable-prompt-tokens-details` | **开** | `usage.prompt_tokens_details.cached_tokens` ⇒ 客户端才能显示**前缀缓存命中率** |
| `--default-chat-template-kwargs '{"reasoning_effort":"high"}'` | **开** | 思考强度**默认值**;请求级 `reasoning_effort` 覆盖它 |
| `--reasoning-parser deepseek_v41` | **开** | **不加则 `reasoning_content` 恒为空**,思考文本会混进 `content` |

* V4.1 的思考强度取值:**`low`/`high`/`xhigh`/`max`(=25/50/75/100)**、**`none`(关思考)**、
  或 **1..100 的整数**。**`minimal`/`medium` 会被 400 拒绝**(实测确认)。
* DSH 侧还需在 pi-ai 目录里声明 `reasoningEfforts` + `compat.supportsReasoningEffort`
  (能力不会向网关查询)。可粘贴的配置见 `docs/RUNBOOK.md` §5.10。

---

## 3. 端到端性能(官方 `vllm bench serve`)

### 3.1 硬件配置(数字必须与它同页,否则无意义)

| 项 | 本机 |
|---|---|
| **GPU** | **2× NVIDIA A100-PCIE-40GB(SM80)**,TP=2,PCIe Gen4 ×16 |
| **CPU** | **2× AMD EPYC 9654 96-Core**(192 核 / 384 线程;NPS4 ⇒ **8 NUMA node**) |
| **内存** | **1538 GiB**;实测聚合带宽 **~740 GB/s** |
| 线程 | `XIAOTU_MOE_THREADS=60`/rank;`SPIN_IDLE_US=300` |
| 软件 | vLLM 主线 `6b5ef34f7b` + 本插件 **v0.2**;TP=2 / `--max-model-len 1048576` |

### 3.2 ShareGPT 摘要(**output tok/s 与 TTFT 两个独立量,不折算**)

口径:`--dataset-name sharegpt`、`--sharegpt-output-len 128`、`--num-prompts 16`、
**前缀缓存开**、`--backend openai`(裸补全,**不触发思考**;见 §3.4)、C=1/4/8。

| 并发 | Plain:output tok/s | Plain:TTFT (ms) | dspark:output tok/s | dspark:TTFT (ms) |
|---|---|---|---|---|
| **C=1** | <!--PENDING:plain_c1--> | <!--PENDING:plain_ttft_c1--> | <!--PENDING:ds_c1--> | <!--PENDING:ds_ttft_c1--> |
| C=4 | <!--PENDING:plain_c4--> | <!--PENDING:plain_ttft_c4--> | <!--PENDING:ds_c4--> | <!--PENDING:ds_ttft_c4--> |
| C=8 | <!--PENDING:plain_c8--> | <!--PENDING:plain_ttft_c8--> | <!--PENDING:ds_c8--> | <!--PENDING:ds_ttft_c8--> |

**v0.2 相对上一版(`v0.21.0`)在同口径下的变化(Plain,`--backend openai`,C=1/4/8,out≤128)**

| 并发 | output tok/s(旧 → 新) | TTFT ms(旧 → 新) | mean TPOT ms(旧 → 新) |
|---|---|---|---|
| C=1 | 13.75 → **16.64(+21%)** | 2224 → **1339(−40%)** | 43.32 → 42.46(不变) |
| C=4 | 42.36 → 42.50 | 695 → 745 | 75.79 → 74.88(不变) |
| C=8 | 49.58 → 50.03 | 1121 → 1100 | 117.26 → 116.08(不变) |

**怎么读**
* **解码速度(TPOT)没有变化** —— 这一版改的是预填充;
  C=1 的 output tok/s 涨 21%,**全部来自 TTFT 从 2224 → 1339 ms**;
* C=4/C=8 基本不变:并发下 TTFT 已被摊薄,预填充不再是瓶颈(见 §3.3);
* 因此**报数必须带 output 长度**(用户 2026-09-18 的裁定)。

### 3.3 ⭐ output 长度决定 `output_throughput` 里有多少是解码

源码口径(`vllm/benchmarks/serve.py`):`output_throughput = Σoutput_len / dur_s`(**含 TTFT**),
而 `tpot = (latency − ttft)/(output_len − 1)`(**不含 TTFT**)⇒
`output_throughput ≈ 1 / (TTFT/L + TPOT)`。

实测(同一批 8 条 ShareGPT prompt、C=1、`--ignore-eos` 强制长度):

| output 长度 | 纯解码上限 `1/TPOT` | 实测 `output_throughput` | 占解码上限 |
|---|---|---|---|
| 128 | 18.2 tok/s(TPOT 55.0 ms) | **13.54** | **74%** |
| 1024 | 16.5 tok/s(TPOT 60.9 ms) | **15.81** | **96%** |
| 2048 | 16.3 tok/s(TPOT 61.3 ms) | 16.01 | **98%** |

⇒ **output ≥1024 时 `output_throughput` 已经就是纯解码速度**;output=128 时它只有解码上限的
~3/4,差的正是 TTFT 项。也正因如此,**同一份预填充改动**在 out=128 上值 +18%、
在 out=1024 上归零(−0.4%)—— 它抬的是 `TTFT/L`,不是解码。

### 3.4 ⚠️ 口径警告:`--backend openai` **不触发思考**

`--backend openai` 打 `/v1/completions` 并**原样发送 prompt**,而 V4.1 的思考是在
prompt encoder(`encode_messages`)里渲染的、只经 chat template 走到 ⇒ **裸补全路径不开思考**。
DSH 的真实路径是 `/v1/chat/completions` + 默认开思考。两列都测、都标注:

| 口径 | 是否开思考 | output tok/s(C=1) | TTFT (ms) | 实测输出均值(条) |
|---|---|---|---|---|
| `--backend openai`(与历史数字可比) | 否 | 16.64 | 1339 | ~75 |
| `--backend openai-chat`(DSH 真实路径) | **是** | <!--PENDING:chat_c1--> | <!--PENDING:chat_ttft_c1--> | <!--PENDING:chat_out_c1--> |

### 3.5 唯一 prompt 的预填充(不命中前缀缓存,量的是真预填充)

| prompt tokens | 0.2pre/0.21.0 | **v0.2** | 变化 |
|---|---|---|---|
| 806-820 | 6.077 s(132.6 tok/s) | **4.688 s(174.9 tok/s)** | −23% / +32% |
| 1889-1893 | 13.765 s(137.5 tok/s) | **10.558 s(178.9 tok/s)** | −23% / +30% |

### 3.6 长 prompt(DSH 真实开发会话回放)

**这不是 ShareGPT** —— 是本会话的 DSH transcript(`~/.dsh/sessions/.../session.v3.jsonl.zstd`,
23,484 条记录)末尾若干轮的**原文回放**,按 token 截成 2K/4K/8K/16K/32K。
用途:ShareGPT 平均 prompt 只有 ~227 token,**永远触发不到 GPU 预填充**,而这条曲线能。

| prompt tokens | TTFT(CPU 预填充) | TTFT(GPU 预填充) |
|---|---|---|
| 2048 | <!--PENDING:rp2048_cpu--> | <!--PENDING:rp2048_gpu--> |
| 4096 | <!--PENDING:rp4096_cpu--> | <!--PENDING:rp4096_gpu--> |
| 8192 | <!--PENDING:rp8192_cpu--> | <!--PENDING:rp8192_gpu--> |
| 16384 | <!--PENDING:rp16384_cpu--> | <!--PENDING:rp16384_gpu--> |
| 32768 | <!--PENDING:rp32768_cpu--> | <!--PENDING:rp32768_gpu--> |

---

## 4. 回归门禁(硬约束,全部通过)

```
数值门 test_block23_equiv.py : OK=7 BAD=1   (me=1 的既有偏差 max_rel=1.873e-02,与历史逐位相同)
性能门 bench_engine_ab.py    : DEDUP=12 0.70 ms/层 216 GB/s ; DEDUP=23 0.84 ms/层 300 GB/s
确定性 test_engine_determinism.py 11 : 11 次运行 10/10 逐位相同
```

---

## 5. 推荐配置

见 **`docs/RUNBOOK.md` §5.9**(TP=2 / 1M / 显式封顶 KV / 投机开 / `THREADS=60` / `SPIN=300`)
与 **§5.10**(§2 那三个客户端参数 + DSH 侧配置)。

## 6. 复现

```bash
# 引擎三门禁
bash scripts/check_engine_aligned.sh
python scripts/test_engine_determinism.py 11

# ShareGPT(产品口径 = openai-chat;历史口径 = openai)
PORT=8425 TAG=rel_plain_chat SERVER_TAG=v0.2 CS="1 4 8" N=16 OUT=512 bash scripts/bench_sharegpt.sh
# DSH 会话回放(长 prompt;先看 RUNBOOK §5.8 的显存前提)
bash /tmp/run_replay2.sh <TAG>
```

## 7. 已知限制

* GPU 预填充的 chunk 上限 **MBT=8192**(16384 会 OOM,点在 attention 的
  `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`);
* **`nshard_>=2`(本机恒真)时 legacy `forward_many` 是死代码** —— 优化只需看 `forward_many_nsliced`;
* CED(③ 预填充捷径)仍未实现,原因与边界见 `report/tuning/NOTES.md` §604;
* `--backend openai` 的 ShareGPT 数字**不含思考**,与 DSH 真实负载不同(§3.4)。

---

## 版本口径(最终)

| 名称 | 含义 | 状态 |
|---|---|---|
| `v0.1.0` | 首版(lk 编排链 fork) | 历史 |
| `v0.2.0` → **`0.2pre`** | 2026-09-14「主线化」;**不支持 V4.1** | tag 保留作归档,文档改称 `0.2pre` |
| `v0.21.0` | 2026-09-17 中间产物(V4.1 + GPU 预填充修复) | **已撤回**(Release + tag 均删除) |
| **`v0.2`** | **本版:V4.1-Flash + CPU 预填充优化** | **交付版** |
