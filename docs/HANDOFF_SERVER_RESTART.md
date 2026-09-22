# 交接：服务器重启前状态（2026-09-21 晚）

> 用途：**服务器即将重启**，本文件固化「重启前必须知道的一切」。
> 台账全文见 `docs/EXPERIMENTS.md`（B1–B83）；已知问题见 `docs/PREFILL_KNOWN_ISSUES.md`。

---

## 1. 一句话状态

**GLM-5.3-Flash 与 DeepSeek-V4.1-Flash 在 256K 下的 prefill 都已显著提升，均为「只调 `MBT`、零代码改动」：**

| 模型 | 256K prefill | 相对起点 | 配置 | 判据 |
|---|---|---|---|---|
| **GLM-5.3-Flash** | **266.3 tok/s** | 233.6 → **+14.0%** | `MBT=12288`（起点 8192） | `[fp8-asm]=168`、`DISABLED=0`、`illegal=0`、`OOM=0` |
| **DeepSeek-V4.1-Flash** | **355.0 tok/s** | 115.8 → **+207%（3.07×）** | `MBT=8192`（起点 4096） | `DISABLED=0`、`aten::new_empty=0`、`illegal=0`、`OOM=0` |

**共同根因与教训**：两模型的 `MBT` 都**只测过端点的跳变、从未二分中间档**。
`MBT` 越小 ⇒ chunk 数越多 ⇒ **每 chunk 重流全部专家权重** ⇒ 越慢。
**（GLM 每次重流 141.8 GiB；V4.1 每次 253.1 GiB。）**

**提交与推送**：`origin/main` = 已同步（写本文件后再 commit + push 一次）。
**工作区干净、`rebase` 树（`vllm-up-133b71e0b`）干净。**

---

## 1b. 2026-09-21 深夜续做（重启之后）——**CED 已验收，V4.1 长两行已换数**

**CED（decoder-side SWA bounded replay，上游 PR #56752）验收完成**，被测树 `/home/user/lvllm/vllm-ced`
（= `eddc6d0eb7` **+3 commits / 15 文件 +1500−92**；两臂之间唯一差异就是 CED）：

* **32/32 格零失败**（`vllm bench serve`，L=16384，N=8，C=1/C=2 × 两臂，两臂同 seed ⇒ 配对比较）；
* **prefill**：C=1 **434.5 → 903.9 tok/s（2.08×）**、C=2 **334.6 → 602.1 tok/s（1.80×）**；
* **decode**：C=1 持平（median TPOT 51.91 → 51.84 ms）、C=2 快 **1.99×**（257.3 → 129.4 ms）；
* **生成等价性**：3 条自然长 prompt（2160/5193/10071 token）**首 token 全一致**，
  且 5193 token 上**全文逐字节相同**。⚠️ 「逐字节全文」在 ≥5K **不能当判据**：
  同一臂（CED 关）两次运行自己就不一致（前缀缓存路径差异）——详见 `docs/EXPERIMENTS.md` B84/B85。
* README（中/英）的 V4.1「长」两行已换成 CED 开的数据，并补上原「待测」的「长/C=2」。

**两件工具链结论（都写进 B86）：**

1. **`nsys` 在本机不可用**：容器里只有 2023.1.2，对 driver 580 + CUDA 12.1 采不到 GPU 核事件
   （`nsys stats` 报 `does not contain CUDA kernel data`，sqlite 里没有 KERNEL 表）
   ⇒ **§3.4 第 3 条「nsys 一次运行给出完整时间线」作废**。
2. ③ 的下一步改为**插件自带的逐层计时**：`[layer-timing]` 已加**墙钟时间戳**（只改打印），
   配 `XIAOTU_LAYER_TIMING=1 XIAOTU_LAYER_TIMING_EVERY=1` 可逐层反解 `pre/eng/post` + 层间墙钟，
   解析器 `dev-docs/report/tuning/probes/attrib_layer_timing.py`。**这是 ③ 现在最便宜的一步。**

**③ 已按此做完(见 §3 顶部、EXPERIMENTS B87/B88):缺口 = 稀疏 MLA 注意力被记进 MoE 的 `pre`;闭合账无残差。**

**pr4 已实测(B89)**:独立树 `/home/user/lvllm/vllm-pr4`(生产树 `git worktree` + 编译产物,生产树未动)
打上 `patches/upstream/pr4-sm8x-sparse-mla-2d-tile.patch` ⇒ **TTFT 47.44 → 31.18 s(1.52×)**、
稀疏 MLA 核 **24.80 → 8.60 s(2.89×)**,其余各项不变。**是否打进生产树待用户定。**

**⭐ 生产口径已改并实测通过(B90):`MBT=4096` + GLM-5.3 **512K** + DeepSeek-V4.1 **1M**。**
* GLM:默认 `MAXLEN=524288`/`MBT=4096`(实测 `/v1/models=524288`、KV 池 952,107 token)。
* V4.1:默认 `MBT=4096`;1M 用 `GPUS=0,1 TP=2 MAXLEN=1048576 MBT=4096 SEQS=64 LOAD=auto`
  (实测 `/v1/models=1048576`)。
* ⚠️ GLM 的 KV 自动封顶**不再乘 SEQS** —— 512K+SEQS=2 时它会一次要 23.56 GiB ⇒ CUDA OOM(差 0.6 GiB)。

---

## 2. 重启会打断的任务

| 任务 | 状态 | 重启后如何处理 |
|---|---|---|
| ~~**V4.1 长/C=2**（`/tmp/v81c2.sh`，PORT 8701）~~ | **已由我主动停掉**（未出结果）。原因：它只为一格 README 数据，而 CED 子代理要用 GPU0/1 验证 1500 行实现 —— 优先级明确 | **需重跑**（若仍要那格数据）。命令见 §4 |
| 所有 256K 测量 | ✅ 已完成并写入 README/CHANGELOG | 无需重跑 |

**README 里「长/C=2」两格** —— **V4.1 那一格已在 §1b 取得**（CED 开：prefill 602.1 / decode 7.7）；
**GLM 那一格仍未取得**（`待测 †`）。

---

## 3. 当前主攻项：③「~14 s 未归属时间」（用户指定）

> ## ✅ 2026-09-21 深夜：**③ 已结案** —— 详见 `docs/EXPERIMENTS.md` **B87 / B88**。结论三句话：
>
> 1. **缺口不在 MoE 之外**：逐层计时显示 `pre`（apply 内）= **98.9%**、`other`（apply 之外，含注意力/层间）= **1.1%（11.3 ms/层）**。
>    注意力时间是被 **MoE `apply()` 的 `pre` 吸收**了（host 在 apply 内阻塞等 GPU）。
> 2. **慢层精确等于 DSA 层**：逐层「3 快 1 慢」，慢层名就是 `layers.7/11/15/…`，与
>    `layer_types` 的周期 4（`deepseek_sparse_attention`）一一对应。
> 3. **闭合账（chrome trace，无残差）**：`_sparse_mla_fwd_with_sink_kernel` **24.80 s = 59.8%**、
>    MoE GEMM 5.80 s、NCCL 2.08 s、其余核 2.11 s、**非核空隙（H2D DMA）6.71 s**，合计 **41.49 s**。
>    ⭐ **最大的一笔现成收益**：交付树跑的是**未优化**的 1-D 稀疏 MLA 核；
>    `patches/upstream/pr4-sm8x-sparse-mla-2d-tile.patch`（自测**核 2.88× / 端到端 1.52×**）
>    被标为 "Tree reverted; patch is standalone"，**没有打进树**。
>    打回去预期注意力 24.8→~8.6 s、TTFT 47.4→~30 s 量级。**未动手（生产树）。**

### 3.1 已确立的事实（实测）

**(a) 账目（32K / MBT=16384 / TP=2 / 单条 16384 / C=1，基线 TTFT 36.86 s）**

| 类别 | 时间 | 备注 |
|---|---|---|
| `moe_forward_shared`（**父行，含 H2D**） | **11.45 s** | 内含 `down` 2.885 + `gate_up` 2.566 + **Memcpy(HtoD) ~5.67** |
| `_sparse_mla_fwd_with_sink_kernel_hb` | **7.74 s** | 稀疏 MLA 注意力 |
| `ncclDevKernel_AllReduce` | 2.11 s | 已证明**不在关键路径**（关掉 TTFT 只差 9 ms） |
| KDA 四核合计 | 0.12 s | 线性注意力**很便宜** |
| 其余叶子（转置/mhc/aten…） | ~1.5 s | |
| **已归因合计** | **22.91 s** | |
| **实测 TTFT** | **36.86 s** | |
| **⇒ 缺口** | **13.95 s** | **每层约 332 ms**（墙钟 878 ms/层，已归因 546 ms/层） |

**(b) 窗口覆盖度：排除了「漏采」**

profile 的调用次数 vs 应有层数：

| 核 | 捕获/应有 | 覆盖 |
|---|---|---|
| 稀疏 MLA 注意力 | 9 / 11 | 82% |
| MoE `gate_up`/`down` | 39 / 42 | 93% |
| KDA 四核 | 30 / 34 | 88% |

⇒ **窗口基本完整，缺口不是「profile 只采了一部分」造成的。**

**(c) GPU 是满忙的**

外部轮询（`nvidia-smi utilization.gpu`，100 ms 一次，423 点）：**median 98%、max 100%**
（mean 74% 是因采样窗口含加载与收尾）。⇒ **缺口不是「GPU 空闲」。**

**(d) `Memcpy HtoD` 行自身记 `0.000us`**（所有历史日志一致）——
因为该 DMA 由插件 C++ 的 `cudaMemcpy2DAsync` 发出，profiler 的 CUDA-kernel 视图**归因不到它**；
**但它的时间已计入父行 `moe_forward_shared`**（B18 已拆解：11.52 ≈ 11.45）。
⚠️ **我一度据此误判「缺口含 DMA」（B82），已更正（B83）—— DMA 不是缺口。**

### 3.2 已排除

| 假设 | 证据 |
|---|---|
| profile 漏采 | 覆盖度 82–93%，见 (b) |
| GPU 空闲 | median 98% 忙，见 (c) |
| 缺口 = H2D DMA | 已在父行内，见 (d) |
| 核本身低效（GEMM） | GEMM 8.1% 达峰属正常；且**优化 −14.4% 只让 TTFT 动 −0.3%**（B36） |
| `all_reduce` 占关键路径 | 关掉 2.11 s ⇒ TTFT −9 ms（B23） |

### 3.3 四次「完整 trace」尝试均失败（**不要再走**）

| 尝试 | 结果 |
|---|---|
| `TRITON_KERNEL_DUMP` ×2 | 两次都只抓到第二个变体 / 都抓不到 |
| chrome trace | 只有 64 个 event、58 个 metadata、**没有 kernel 时间线** |
| `XIAOTU_TORCH_PROFILE_CALLS=1000000` | **profiler 从不 flush ⇒ 0 行输出**（该值必须 ≤ 实际调用数才会输出） |

⇒ ~~**结论：现有工具链拿不到完整 GPU 时间线。** 缺口是「工具盲区」而非「实验没跑成」。~~

> ❌ **这一条已被 B88 推翻**：`export_chrome_trace` 的 JSON **里有**带 `ts/dur` 的 `cat=kernel` 事件
> （本次一个 rank **8633 个核事件**）。之前"只有 64 个 event、没有 kernel 时间线"是**读法/配置**的问题,
> 不是工具链的限制。解析器已写好:`probes/attrib_trace.py`（算 GPU 忙占比 + 最大空隙 + top 核）。
> ⇒ **不要再花时间在 TRITON_KERNEL_DUMP / nsys 上,直接用 chrome trace。**

### 3.4 下一步（**2026-09-21 深夜更新：第 2 条已做，第 3 条已作废**）

1. **最便宜（仍在）**：开 `XIAOTU_LAYER_TIMING=1 XIAOTU_LAYER_TIMING_EVERY=1` 跑一次 32K 基线请求。
   ⚠️ **口径先修正**：`_lt_t0` 在 `mixed_experts.py:1368`（GPU 流式分支**之前**）、
   `_lt_record` 在 `:1919`（CPU 引擎调用**之后**）⇒ **GPU 预填充时 `pre` 覆盖整段 GPU MoE（H2D+核）、
   `eng`≈0**；CPU 路径才 `eng`>0。历史上把两种运行的 `pre/eng` 混着比是**错的**
   （B17 的 `eng` 21.9–33.3 ms 说明那次是 CPU 路径）。
2. **已做**：`[layer-timing]` 加了**墙钟时间戳**（只改打印，默认不变）。
   现在可直接从日志得到**逐层时间线**：相邻行 Δt = 层间墙钟，逐层 `pre/eng/post` 由累计平均反解
   ⇒ 一跑就能把每层拆成「apply 内 MoE」vs「apply 之外（注意力/indexer/层间）」。
   解析器：`dev-docs/report/tuning/probes/attrib_layer_timing.py`。
3. **❌ 已作废**：`nsys` **在本机采不到 GPU 核事件**（只有 2023.1.2 套件 vs driver 580；
   `nsys stats` → `does not contain CUDA kernel data`，sqlite 无 KERNEL 表）。见 B86。
4. **结构性怀疑对象**（未验证）：KDA 侧的**非核**工作、CUDA graph 的 replay 开销、
   或 vLLM 侧未进 profile 的融合核。**注意 KDA 四核只 0.12 s，所以不是 KDA 的核。**
5. **另一条更硬的判据（未做）**：插件内 `XIAOTU_TORCH_PROFILE` 会 `export_chrome_trace`，
   而 chrome trace 里**有**带 `ts/dur` 的 GPU kernel 事件 ⇒ 可直接算「窗口内 GPU 忙/空隙」，
   判断那 332 ms/层是「真空隙」还是「叶子求和漏项」。**这是与第 1 条互补的一步。**

---

## 4. 环境与复现命令（**含多个必须知道的坑**）

### 4.1 服务启动

**GLM-5.3-Flash：**
```bash
cd /home/user/lvllm/vllm-xiaotu-moe
setsid env PYTHONPATH=/home/user/lvllm/vllm-up-133b71e0b TAG=<tag> PORT=<port> \
  GPU_UTIL=0.85 SPEC_K=0 SEQS=1 MAXLEN=262144 MBT=12288 \
  KV_CACHE_BYTES=5113807360 XIAOTU_GPF_STAGE=1 \
  bash scripts/serve_glm53_mainline.sh > /tmp/<tag>_w.log 2>&1 &
# served-model-name 是 GLM-5.3-Flash
```

**DeepSeek-V4.1-Flash：**
```bash
setsid env PYTHONPATH=/home/user/lvllm/vllm-up-133b71e0b TAG=<tag> PORT=<port> \
  GPUS=0,1 TP=2 MAXSEQS=1 GPU_UTIL=0.85 \
  MAXLEN=262144 MBT=8192 KV_CACHE_BYTES=8031830016 \
  bash scripts/serve_v41.sh > /tmp/<tag>_w.log 2>&1 &
# ⚠️ 必须显式传 GPUS=0,1（脚本默认 GPUS=0）
# served-model-name 是 dsv41（不是 DeepSeek-V4.1-Flash）
# 日志在 dev-docs/report/tuning/logs/<tag>.log，不是 logs/
```

### 4.2 五个必踩的坑（每个都让我失败过）

1. **`GPUS` 默认是 `0`** ⇒ `TP=2` 不传 `GPUS=0,1` 会 pydantic 报
   「World size (2) larger than available GPUs (1)」；
2. **`--served-model-name` 猜错** ⇒ `NotFound`，**请求根本没送达**（V4.1 真名是 `dsv41`）；
3. **V4.1 日志路径**是 `dev-docs/report/tuning/logs/`，GLM 是 `logs/`；
4. **`KV_CACHE_BYTES` 必须按引擎反算的真实需求、不加乘性余量**
   （GLM 19,505 B/token；V4.1 30,639 B/token）—— 256K 下 10% 余量就 OOM；
5. **V4.1 的 `MBT=16384` 会撞装配期 `aten::new_empty` 失败** ⇒ 用 8192（或更小）。

### 4.3 清场（**验收判据是 `nvidia-smi` 读到 0**）

```bash
for p in $(pgrep -f '[V]LLM::EngineCore'); do pkill -9 -P "$p"; kill -9 "$p"; done
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done
```
**⚠️ `EngineCore` 子进程不在任何 pidfile 里，必须按名字杀（用 `[V]` 括号技巧避免自匹配）。**

---

## 5. 过程纪律（本会话用多次事故换来的硬规则）

| # | 规则 | 代价 |
|---|---|---|
| 1 | **`pgrep -f`/`pkill -f` 绝不用会出现在自己命令行里的字符串**（用 `[V]LLM` 括号技巧） | 4 次自杀 |
| 2 | **绝不 `kill -9 -PGID`**（会波及自己的进程组） | 1 次杀死自己的组 |
| 3 | **改 `rebase` 树必须挂 `trap ... EXIT` 回滚**，不能把回滚写在流程末尾 | 2 次脏树（各约 10 分钟） |
| 4 | **「失败即退出」的检查必须留启动宽限期**（服务启动需 ~340 s，EngineCore 前 60 s 可能还不存在） | 1 次误杀健康实验 |
| 5 | **改树前记 md5、改后三重校验**（md5 / 与备份逐字节 / `git status`） | — |
| 6 | **测量优先选「不动被测对象」的手段**（外部轮询一次成功；改源码注入探针写坏了类） | — |
| 7 | **多 rank 日志做任何计数前先按 `Worker_TP*` 分组** | 1 次把 42 层数成 70 |
| 8 | **先穷举已有的中间选项，再追更大的目标** | **GLM 79 轮 + 本会话大量运行** |

---

## 6. 未解项（**不要假装已解决**）

| 项 | 状态 |
|---|---|
| **③ ~13.95 s 未归属** | ✅ **已结案（B87/B88）**：它是稀疏 MLA 注意力被记进 MoE 的 `pre`；闭合账见 §3 顶部 |
| `h` 为何逐层不释放（结构上应释放） | ⚠️ 未查 |
| pr4 的**端到端数值**与 `head_mask` 分支 | ⚠️ 未验证（核级 `max|diff|=2.819e-05` 已做） |
| 布局转换（为何二维 `q` 快 2.8×）的解释 | ⚠️ 未证实 |
| README 的「长/C=2」两格 | **V4.1 已取得（CED 开：602.1 tok/s / decode 7.7）；GLM 仍待测**（§1b） |
| CED 逐层 SWA KV 的 **Golden 对拍** | ⚠️ 未做（`CED_PR_PLAN.md` §4；注意带 `prompt_logprobs` 时 CED 会自动不裁剪，见 B85c） |
| README 的「短」三行 | 来自**早期 32K 预算**的测量，**未在 256K 配置下重测** |

---

## 7. 其他工作线（**由人类手工开启的子代理负责，我不接管**）

**CED（decoder-side SWA bounded replay）已在另一个树上实现：**

* 工作树 **`/home/user/lvllm/vllm-ced`**，分支 `ced/pr56752`，HEAD `f8a142e168`（基线 `eddc6d0eb7`）。
* 3 个 commit，**15 files / +1500−92**；上游 PR #56752 `CLOSED`（非评审否决），故自行承接。
* **⚠️ 子代理报告称「V4.1-Flash 检查点本机已删」，此说法不成立** ——
  检查点实际存在（`/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master`，**476 GB**），
  且我在 22:25 用它跑通了 256K 请求。**我已在会话中发出更正。**
* 其验证三件套（生成等价性 / 加速比 / 端到端不崩）**已于 §1b 完成**（2026-09-21 深夜，真权重）：
  加速比 2.08×/1.80×、32/32 格零失败、首 token 等价。
  ⚠️ 仍**未做**：逐层 SWA KV 的 **Golden 对拍**（`CED_PR_PLAN.md` §4），
  以及 pr4 的端到端数值与 `head_mask` 分支。
* **子代理补充的两条事实（有用，已采纳）**：
  1. **CED 复用 `CacheConfig.swa_bounded_replay`（默认 `True`），没有单独 flag** ⇒ A/B 要用 `--no-swa-bounded-replay`；
  2. `serve_v41.sh` 的 **`--load-format` 默认 `dummy`** ⇒ `dummy` 只能证明「kernel/服务不崩」与 A/B 确定性，
     **要评生成等价性需 `LOAD=auto`**。
* **资源状态（写本文件时）**：三张卡**全空**（`0/0/0 MiB`），主机内存 `available 1493 GB`。
  ⇒ CED 的 `GPUS=2 TP=1` 或 `GPU0,1 TP=2` 都能起。
  ⚠️ **但服务器重启会杀掉任何在跑的服务** ⇒ 子代理的验证宜在**重启后**进行，或只跑「短时间能出结论且可立刻重来」的项。

---

## 8. 重启后建议的第一步

```bash
# 1) 确认环境干净
cd /home/user/lvllm/vllm-xiaotu-moe && git status --porcelain
git -C /home/user/lvllm/vllm-up-133b71e0b status --porcelain
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 2) 补 README 的「长/C=2」两格（可选，各约 10 分钟）
bash /tmp/v81c2.sh          # V4.1（重启前未出结果，需重跑）

# 3) 回到 ③：开逐层计时，把缺口定位到「哪一相」
#    见 §3.4 第 1 条
```
