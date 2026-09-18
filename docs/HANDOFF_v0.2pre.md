# v0.2.0 交接文档(HANDOFF)

> 面向:**开新会话继续这条线的人**。
> 目标:v0.2.0 已发布,交接**已验证的事实 / 未验证的假设 / 已知陷阱 / 下一步入口**。
> 配套阅读顺序:`README.md` → `docs/PLUGIN_INTERFACE.md` → 本文 → `report/tuning/NOTES.md`(按 §号跳读)。

---

> ### ⚠️ 本文是 v0.2.0 的交接文档(历史);**已被 v0.21.0 取代**
>
> 若你是新会话:请从 `docs/RUNBOOK.md` 5.9 节(最新推荐配置)与
> `RELEASE_NOTES_v0.21.0.md`(最新交付)开始,再看 `report/tuning/NOTES.md` 的 594-604 节
> (GPU 预填充与 CED 的完整证据链)。
> 本文的「已验证事实 / 陷阱」大多仍成立,但凡涉及 **GPU 预填充**的结论
> (当时是「默认关、更慢/OOM」)**已作废** —— v0.21.0 修好后实测**快 2.0-2.8x**,阈值推荐 **4096**。

## 0. 一句话现状

**v0.2.0 已发布**(2026-09-14):vllm-xtu-moe 可以在 **vLLM 主线**上跑起来,一条命令安装,
解码性能为 fork 的 **1.34×**,内存 190 GB/worker(fork 105 GB)。
**主线路径已端到端验证并通过数值门禁**;Qwen3.8-Flash-Next 等其它模型**未在当前代码上复验**。

| | |
|---|---|
| tag | `v0.2.0` |
| Release | https://github.com/yeungtuzi/vllm-xtu-moe/releases/tag/v0.2.0 |
| 发行物 | `dist/vllm_xtu_moe-0.2.0-*.whl`(2.06 MB)、`dist/vllm_xtu_moe-0.2.0.tar.gz`(2.05 MB) |
| 最新提交 | `30c8c5d` |
| 远端 | `https://github.com/yeungtuzi/vllm-xtu-moe.git`(branch `main`) |

---

## 1. 本机环境(所有数字的基准)

- **2×A100-40GB**(PCIe,NVLink 无)+ 2×EPYC 9654(192 核)+ **NPS=4 ⇒ 8 个 NUMA node,每 node 189 GB**
- 1.5 TB DDR5-4800
- CUDA 12.1(注意:**FlashInfer 需要 ≥12.8** ⇒ 必须 `VLLM_USE_FLASHINFER_SAMPLER=0`)
- 三个关键环境:
  | 变量 | 用途 |
  |---|---|
  | `ENV=/home/user/anaconda3/envs/vllm-xiaotu-moe` | **主线**(tree B = `process_data/ref/repos/vllm-mainline` @ `6c73b08dec` + 32 个改动文件)+ 我们的插件 |
  | `ENV=/home/user/anaconda3/envs/lkxtu` | **fork 树**(`Lvllmds4-x`)+ **我们的引擎**(它是"只换引擎"对照) |
  | `ENV=/home/user/anaconda3/envs/lvllmds4-x` | fork + **原版 lk_moe** —— **⚠️ 从未跑过对照,见 §5** |
- 模型:`/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master`

---

## 2. v0.2.0 交付了什么(都可验证)

| 条目 | 产物 | 怎么验 |
|---|---|---|
| 漂移审计 | `docs/UPSTREAM_DRIFT.md` | 读 |
| 最小补丁集 | `patches/upstream/pr0..pr3` + `scripts/apply_xtu_patches.sh` | `DRY=1 LEVEL=3 bash scripts/apply_xtu_patches.sh <tree>`(0 failed hunks) |
| 一条命令安装 | `scripts/install_mainline.sh`(`LEVEL=0/1/2/3`) | `DRY=1 LEVEL=0 bash scripts/install_mainline.sh` |
| 启动自检 | `scripts/check_mainline_env.sh`(9 条断言) | `bash scripts/check_mainline_env.sh` |
| 接口契约 | `docs/PLUGIN_INTERFACE.md`(10 条耦合表) | 读 |
| 同机 A/B 基线 | `scripts/ab_mainline_vs_fork.sh` | `CS="1" bash scripts/ab_mainline_vs_fork.sh` |
| 主线起服务 | `scripts/serve_mainline.sh` | 见 §3.1 |
| GPU 预填充设计 | `docs/GPU_PREFILL_MAINLINE.md` | 读 |

**硬约束(继承,不要破)**:不改 vLLM 上游核心文件(只走 `patches/` + 插件入口)· 复用优先上游→lvllm→自研 · **draft model 永远在 GPU** · 每次改动 git commit + NOTES + before/after · 优先无模型加载微基准。

---

## 3. 已验证的事实(可以依赖)

### 3.1 主线起服务(唯一验证过的配置)

```bash
cd /home/user/lvllm/vllm-xiaotu-moe
ENV=/home/user/anaconda3/envs/vllm-xiaotu-moe TAG=mytag PORT=8071 \
  MBT=256 GP_MIN=0 TP=2 GPUS=0,1 RESIDENT=0-11 EAGER=0 THREADS=60 \
  bash scripts/serve_mainline.sh
# 等 "Application startup complete"(~300 s);KV ≈ 113,271 token
```
只测 512-in/128-out 用 `scripts/bench_lat.sh`:
```bash
PORT=8071 TAG=mytag SERVER_TAG=mytag CS="1 2 4" bash scripts/bench_lat.sh
```

### 3.2 性能与数值(同机同协议)

| | C=1 TPOT | C=1 agg | C=2 agg | C=4 agg |
|---|---|---|---|---|
| **主线 + 插件** | **35.97 ms** | 16.95 t/s | 29.32 t/s | 38.55 t/s |
| fork + 同一个引擎 | 26.81 ms | 22.42 t/s | — | — |

- **数值门禁**:`XIAOTU_LAYER1_NPZ=fixtures/real_layer1_model.npz python scripts/test_block23_equiv.py`
  ⇒ **`OK=7 BAD=1`,`me=1 max_rel = 1.873e-02`**(与 0.1.0 基线逐位一致)
- **引擎逐位确定**:`python scripts/test_engine_determinism.py` ⇒ 11/11 一致
- **GPU 预填充(主线)**:`PREFILL=1`(⇒ `MBT=8192/GP_MIN=1024/只捕获解码尺寸`)
  实测 1750 token → TTFT **3.379 s(606 t/s)**,是纯 CPU 基线的 **2.6×**
- **内存**:190 GB/worker(改前 260)

### 3.3 三个**必需**的开关(缺了不会报错,只会静默变慢)

`serve_mainline.sh` 已默认带上,但**自己拼命令行时必须带**:

| 开关 | 值 | 缺了会怎样 |
|---|---|---|
| `XIAOTU_MOE_ASYNC` | `0` | 异步握手在主线下每层多等 ~28 ms ⇒ **11.96 s → 1.42 s(8.4×)** |
| `XIAOTU_MOE_NSLICE_SMALL` | `0` | `wlimit=59/60` 门闸失效,60 个 worker 全参与每一相 |
| `XIAOTU_MOE_SPIN_IDLE_US` | `0` | worker 每次调用后自旋 5 ms ⇒ 池几乎不停转(load 119) |

⇒ **`bash scripts/check_mainline_env.sh` 会在起服务前断言这些**(已接入 `serve_mainline.sh`)。

---

## 4. 六个陷阱(每个都真实踩过,别再踩)

1. **计时口径**:`wall / out_tokens` **含预填充**。必须用 `(wall-ttft)/(n_out-1)` 或 `bench_lat` 的 **TPOT**。
   (我因此把 fork 上"512-token prompt 168 ms/token"当成解码,实际解码只有 ~38 ms。)
2. **`[cd-timing/async]` 的 `period`/`compute` 只反映"投递"**,不反映异步 worker 的算完时间。
   异步路径下 `period × 43 = 42 ms/token` 与端到端 `1225 ms/token` **可以同时成立**。
   量真实算力要用 `XIAOTU_MOE_FAKE_CPU`(分离"算力 vs 搬运")或诊断性 `sync`。
3. **隔离微基准覆盖不到异步/池行为**:`bench_cd_plumbing.py` 给 0.32 ms/层,服务里曾 28 ms/层
   (单 rank、背靠背调用 ⇒ worker 永不 park)。**"微基准正常" ≠ "服务正常"。**
4. **`nsys`/`perf` 等一切重工具在模型加载期间不要跑**;一次整机加载 ~5 min,用微基准迭代。
5. **杀进程不要用 `pkill -f`**:会自匹配。用 `report/tuning/logs/<tag>.pid` 里的 PID,
   或 `nvidia-smi --query-compute-apps=pid`。
6. **启动"只加载到一半就没了"且无 Traceback** ⇒ 去 `/var/log/kern.log` 找 OOM 行,
   看 `nodemask=N` —— 是**单 NUMA 节点耗尽**,不是总量不足。见 §5.4。

---

## 5. 未验证 / 未解决(按优先级)

### 5.1 🔴 其它模型在**当前代码上**未复验(最该先做)

v0.2 这一程改了**所有模型都会走的路径**:`XIAOTU_MOE_ASYNC=0`(执行模型)、
`NSLICE_SMALL=0`(小 batch 路径)、**EP 存储分片**(专家分配/加载/映射)。
**⇒ 只有 DeepSeek-V4-Flash 被复验过。**

| 模型 | 历史状态 | 需要做什么 |
|---|---|---|
| **Qwen3.8-Flash-Next-FP8** | `NOTES §5` **已在 mainline 跑通**(单卡 A100-40GB,`XIAOTU_PLE_CPU=1`:VRAM 14.7 GiB、KV 17.83 GiB/728,851 token、答案正确、ShareGPT C=8 TPOT 195 ms)。checkpoint 在 `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8`,日志在 `report/tuning/logs/qwen38_tp1_pleoff*.log` | **用 v0.2 代码重跑 `scripts/fp8_moe_smoke.py`**(见 `MODEL_GUIDES.md §2.2`),尤其验 EP 存储分片没破坏它的 PLE offload |
| Qwen3-30B-A3B-FP8 | 单卡跑通(0.1.0 时期) | 重跑 |
| Qwen1.5-MoE-A2.7B-GPTQ-Int4 | INT4 路径跑通(0.1.0 时期) | 重跑 |
| GLM-5.3-Flash | **需要 SM90+**,本机(A100)不行 | 无需 |
| **DeepSeek-V4.1-Flash**(748B) | 只有资源账分析(`docs/V41_FLASH_ANALYSIS.md`) | **未跑过**;需主线先支持 `deepseek_v41` |

⚠️ **README 第 39/86/92 行把这些历史实测写成了当前能力** —— 应补上"验证时点/路径"标注。

### 5.2 🔴 原版 `lk_moe`(`lvllmds4-x`)对照**从未跑过**

目标里明确列了它。目前的对照只有"**fork + 我们的引擎**"。
成本:**一次服务启动**。
```bash
ENV=/home/user/anaconda3/envs/lvllmds4-x TAG=ref PORT=8070 TP=2 ... bash scripts/serve_lk_port.sh
```
跑完就能得到真正的三方对照:原版 lk_moe / fork+我们引擎 / 主线+插件。

### 5.3 🟡 内存仍比 fork 高 1.8×(190 vs 105 GB/worker)

C3(EP 存储分片)已把 260 → 190。**剩余 ~85 GB 未定位**。
参考阶梯:lk_moe@fork **80 GB** / 我们@fork **105 GB** / 我们@主线 **190 GB**。
可能的位置:引擎自身(~74 GB 分片)+ pinned 缓存 + vLLM 侧其它副本。
**注意**:`gpu_prefill._pin_key()` 会把源 `untyped_storage()` 存进 `_PIN_CACHE`
——这是"权重缩容无效"的旧因(NOTES §346c)。现在分配已分片,值得复查它是否还在多留一份。
(用户曾观察 **lk_moe@fork 稳定 80 GB/worker**。)

### 5.4 🟡 NUMA 相关(机制已查清)

- **OOM 机制**:权重**切得很好**(`SHARD_DIAG`:每层 8 分片,每 node 仅 ~18.5 GB,`mbind` 全成功);
  OOM 来自 **vLLM 每 worker 138 GB 的 first-touch、未绑定**分配。
  **是"粒度 × 未绑定",不是总量**:8 node × 189 GB 时堆到 185 GB 就撞顶;2 node × 756 GB 有 4× 余量。
- **现状**:C3 之后 **`INTERLEAVE=0` 也能起**(默认 NSHARD=8 实测无 OOM)
  ⇒ `numactl --interleave=all` 从"必需"降级为"**可选优化**"(带它 1.47 s / 不带 1.58 s)。
- **`XIAOTU_MOE_NSHARD=2`(按 socket 切两片)已否决**:能免 interleave 但**慢 32%**(1.94 s)。
- 用户建议的 **NPS=4 → NPS=1** 改动未在本机验证(需改 BIOS)。

### 5.5 🟡 TP=2 下非逐位确定(语义等价)

同一配置、同一 seed 连续两次 greedy 只有 **3/5** 逐字符一致(引擎单 rank 11/11 确定,fork 栈 5/5 确定)。
分歧来自近似并列候选的 FP 次序,**最终答案一致**。
⇒ 已写入 `RELEASE_NOTES_v0.2pre.md` §6 与验收口径:数值以门禁为准,端到端以语义抽样为准。
**若要修**:先定位是 vLLM 侧 all-reduce 还是我们 EP 归约的顺序;`XIAOTU_MOE_EP_SHM=0` 可切到 NCCL 对比。

### 5.6 🟢 深水区(有入口,优先级低)

| 项 | 入口 |
|---|---|
| 大 `MBT` 的宿主内存 | `cfg.group_max_len = max(4096,MBT)+128`;MBT=8192 曾 OOM(现已因 C3 减轻,值得重测) |
| `parallel_for_limited` 边界竞态 | 强制小 `wlimit` 会**挂死**(NOTES §356);修好才能精确控制 worker 参与度 |
| `small_batch_workers()` 标定 | `single_us` 用标称 MAC 率,高估 ~50× ⇒ `wlimit=59/60`;依赖上一条 |
| GPU 预填充 P2/P3 | `docs/GPU_PREFILL_MAINLINE.md`(全层接入 / 阈值 T / 与解码共存) |
| **投机解码未在主线验证** | 硬约束"draft model 永远在 GPU"在主线路径上**未实测**;所有测试 `SPEC=0` |
| fresh-venv 安装走查 | `install_mainline.sh` 只 `DRY=1` 测过 |

---

## 6. 已实测**否定**的假设(不要重走)

| 假设 | 结果 |
|---|---|
| `numactl --interleave=all` 拖慢了引擎 | ❌ fork 加上它反而 1.88 → 1.18 s |
| `SPIN_IDLE_US=0` 走 condvar 是主因 | ❌ fork 设 0 与默认逐位相同 |
| `THREADS` 过大 | ❌ 12 比 60 **更慢**(50.3 vs 37.7 ms/token) |
| `RANK_SPLIT=2`(fork 式 NUMA 布局) | ❌ 请求**全部挂死** |
| `NSHARD=2` 更好 | ❌ 慢 32% |
| 每层新建 ids/weights 张量是主因 | ❌ 只 7%(12.86→11.96) |
| "无限重做预填充"(`qlen=257`) | ❌ qlen 谱只有 `{1,2,4,8,16}` |
| chunked-prefill 把预填充混进解码步 | ❌ 同上 |
| 引擎算力/隔离微基准能代表服务 | ❌ 见 §4.2/§4.3 |

---

## 7. 下一步建议(给新会话的起手式)

**如果你想先"保底"**(推荐,半天):
1. §5.1 **用 v0.2 代码复验 Qwen3.8-Flash-Next-FP8**(`scripts/fp8_moe_smoke.py`)——
   它是最可能被 EP 存储分片影响的模型;顺带修 README 的时点标注。
2. §5.2 **跑原版 lk_moe 对照**(一次服务启动),补齐三方对照表。

**如果你想直奔 DS-V4.1-Flash**:
- 先读 `docs/V41_FLASH_ANALYSIS.md`(资源账已做)。
- **前置依赖**:vLLM 主线必须先支持 `deepseek_v41` —— 插件只覆盖 MoE 层,
  **无法独自提供模型支持**。先确认主线/所需 patch 的现状,再动。
  > ✅ **2026-09-17 更新:该项已完成** —— 主线已支持 `deepseek_v41`,本项目在 **v0.21.0**
  > 上把 V4.1-Flash 全链路(1M 上下文 / GPU 预填充 / 投机)跑通并验收,见
  > `RELEASE_NOTES_v0.21.0.md` 第 1 节与 `docs/RUNBOOK.md` 5.9 节。
- 起手式:先建 `docs/` 里的 V4.1 支持清单(架构差异 → 路由/量化/PLE 是否变化),
  再决定是"复用 V4 路径"还是"新增 model override"。

**每次改动的纪律**:`git commit` + `report/tuning/NOTES.md` 追加 §号(带 before/after)
+ 否定结果进 `TRIED_AND_REVERTED.md` + 整机加载只在里程碑。

---

## 8. 快速命令卡

```bash
R=/home/user/lvllm/vllm-xiaotu-moe; cd $R

# 自检(起服务前)
bash scripts/check_mainline_env.sh

# 微基准(无模型加载,秒级)
XIAOTU_MOE_NO_AUTO_EP=1 XIAOTU_LAYER1_NPZ=<ckpt> NENGINES=8 REP=15 THREADS=60 \
  CFG_WORLD=2 CFG_RANK=0 MODE=both \
  /home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python scripts/bench_cd_plumbing.py

# 数值门禁(无服务)
XIAOTU_LAYER1_NPZ=fixtures/real_layer1_model.npz \
  /home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python scripts/test_block23_equiv.py
# ⇒ 期望 OK=7 BAD=1, me=1 max_rel=1.873e-02

# 引擎确定性(无服务)
/home/user/anaconda3/envs/lkxtu/bin/python scripts/test_engine_determinism.py 12

# 起服务 + 基准
ENV=/home/user/anaconda3/envs/vllm-xiaotu-moe TAG=t1 PORT=8071 MBT=256 GP_MIN=0 \
  TP=2 GPUS=0,1 RESIDENT=0-11 EAGER=0 THREADS=60 bash scripts/serve_mainline.sh
PORT=8071 TAG=t1 SERVER_TAG=t1 CS="1 2 4" bash scripts/bench_lat.sh

# GPU 预填充(GPU 假地板)
#   EXTRA_ENV="XIAOTU_MOE_FAKE_ALL=1"   ⇒ 纯 GPU 地板(实测 33.6 ms/token)
#   EXTRA_ENV="XIAOTU_MOE_FAKE_CPU=1"   ⇒ 保留拷贝、跳 CPU 计算(分离"算力 vs 搬运")

# 同机 A/B
CS="1" bash scripts/ab_mainline_vs_fork.sh
```
