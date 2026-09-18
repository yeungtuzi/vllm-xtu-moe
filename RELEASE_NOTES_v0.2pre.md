# vllm-xtu-moe v0.2pre — 主线化(历史版本;原 `v0.2.0`)

**主题:让 vllm-xtu-moe 在 vLLM 主线最新版上跑起来,并提供一条命令的安装方式。**

v0.1.0 的性能数字全部来自 **lk 编排链 fork**(`Lvllmds4-x`);用户要装两套东西,
而且 fork 与主线已经漂移、无法随主线 rebase。v0.2 把这条链**搬回主线**。

---

## ⚠️ 重切说明(2026-09-14 晚,同名 `v0.2.0`)

**同日重建发行物**:初版 v0.2.0 针对主线 `6c73b08dec`(2026-09-08);当晚主线已前进
**346 commits**,初版 wheel **在新主线上不能工作**。本次把发行物重建到新基线
**`dabc4362b`(2026-09-14)**,版本号仍为 0.2.0(tag/release 覆写)。

| | 初版 v0.2.0 | **重切 v0.2.0** |
|---|---|---|
| 针对的主线 commit | `6c73b08dec`(2026-09-08) | **`dabc4362b`(2026-09-14,+346 commits)** |
| 插件在新主线上 | ❌ `select_experts` ImportError / `num_hash_layers` TypeError / PLE 静默失效 | ✅ 已适配 4 类 API 漂移 |
| `patches/upstream/pr0..pr3` | 只适用旧基线(pr1 有 1 个 hunk 失败) | ✅ **重新生成,在 `dabc4362b` 上 0 failed hunks** |
| C=1 TPOT / 相对 fork | 35.97 ms / 1.34× | **33.46 ms / 1.25×** |
| 每 worker 内存 | 190 GB | **192.7 GB** |
| 数值门禁 | `OK=7 BAD=1`,1.873e-02 | **逐位一致** |

> 如果你已经装了初版 v0.2.0 的 wheel 且**停留在旧主线 `6c73b08dec`**,可以继续用;
> 一旦升级主线,请换用本次重切的 wheel。完整过程见
> [`report/tuning/NOTES.md`](report/tuning/NOTES.md) §368 与
> [`report/tuning/IRON_RULES.md`](report/tuning/IRON_RULES.md) R10。

---

## 摘要

| | v0.1.0 | **v0.2.0** |
|---|---|---|
| 运行方式 | lvllm fork + lk 编排链 + 插件 | **vLLM 主线 + 插件**(零编排补丁 / 极少数补丁) |
| 安装 | 两套(fork + 插件) | **一条命令** `install_mainline.sh`(纯插件 / +补丁 分档) |
| 解码(同机同协议 C=1 TPOT) | 26.81 ms(fork) | **33.46 ms** ⇒ 相对 fork **1.25×** |
| 内存(每 worker) | 105 GB(fork) | **192.7 GB** |
| 数值门禁 | `OK=7 BAD=1` | **`OK=7 BAD=1`**(me=1 max_rel **1.873e-02**,逐位保持) |
| 启动自检 | 无 | **10 条断言**,把"静默变慢 30×"的配置错误变成启动时失败 |

---

## 1. 主线支持(目标第 1/2/3 条)

### 1.1 漂移审计 → [`docs/UPSTREAM_DRIFT.md`](docs/UPSTREAM_DRIFT.md)

三棵树(fork / 主线+SM 补丁 / 干净主线)的完整 diff 审计,量化"只保留必要 hunk 后
补丁有多少行、涉及多少文件"。

### 1.2 最小补丁集 → [`patches/upstream/`](patches/upstream/)

按级别选装,**逐块给出"为什么主线做不到"**:

| 级别 | 补丁 | 文件数 | 为什么必须打 |
|---|---|---|---|
| **L1**(默认) | `pr0-handshake-timeout` | 1 | 上游把 `HANDSHAKE_TIMEOUT_MINS` 硬编码 5 分钟;CPU 引擎要逐层构造 43 层 ⇒ 必然超时。**只把常量变成可配置,不改默认行为** |
| | `pr1-experts-load-device` | 3 | 提供 `VLLM_EXPERTS_LOAD_DEVICE=cpu` —— 没有它,138 GB 专家权重无处安放。主线没有等价公开开关 |
| L2 | `+pr2-fp8-sm80-o-proj` | 4 | A100(SM80)的 FP8 `o_proj`(非 A100 不需要) |
| L3 | `+pr3-sm80-port` | 21 | SM80 上的 DS-V4 完整移植(非 A100 不需要) |

**净结果:L1 只需 4 个文件**,且都能随主线 rebase。

### 1.3 端到端验证(同机同协议,`scripts/bench_lat.sh`)

| C | out_tput | mean_tpot | mean_ttft |
|---|---|---|---|
| 1 | 16.95 t/s | **35.97 ms** | 2983 ms |
| 2 | 29.32 t/s | 53.90 ms | 1873 ms |
| 4 | 38.55 t/s | 72.80 ms | 3995 ms |

同机对照(`scripts/ab_mainline_vs_fork.sh`,fork + **同一个引擎**):C=1 **26.81 ms**。
⇒ 主线相对 fork 的开销是 **1.34×**,剩余部分已量化、属可接受的移植税。

**数值门禁**:`scripts/test_block23_equiv.py` = **`OK=7 BAD=1`**,`me=1 max_rel = 1.873e-02`
—— 与 0.1.0 基线**完全一致**。

---

## 2. 本次最贵的一个 bug(值得单列)

主线移植后解码慢到 **1225 ms/token**(相对 fork **47×**)。根因**不在任何代码 diff 里**:

> 引擎的**异步握手**路径(`XIAOTU_MOE_ASYNC=1`,引擎默认)在 fork 编排下正常,
> 在**主线下每层多等 ~28 ms**(43 层 × 10 pass ≈ 11 s)。

修法是一行:`XIAOTU_MOE_ASYNC=0` ⇒ **11.96 s → 1.42 s(8.4×)**,端到端口径 **47× → 1.34×**。
已设为 `serve_mainline.sh` 默认并加入启动自检。

**这条同时暴露了一类方法论问题**:异步/同步这种"执行模型"开关**必须在真实服务负载下 A/B** ——
分层计时(`period` 只反映投递,不反映异步算完)与隔离微基准(单 rank、背靠背调用,
worker 永不 park)都**覆盖不到它**。已写成 [`docs/PLUGIN_INTERFACE.md`](docs/PLUGIN_INTERFACE.md) §3 的"测量陷阱"。

---

## 3. 内存:每 worker 260 → 190 GB

主线原先**每个 rank 都分配全部 256 个专家**(138 GB),而 fork 按
`moe_config.num_local_experts`(=256/TP)只分配本 rank 的分片。照此修复后:

| | 改前 | **改后** | 参照 |
|---|---|---|---|
| 每 worker `RssAnon` | 260.0 GB | **190.0 GB** | lk_moe@fork 80 GB / 我们@fork 105 GB |

`XIAOTU_MOE_EP_SHARD_STORAGE=0` 可回退。**附带好处**:未绑定部分减半后,
`numactl --interleave=all` 从"**必需**"降级为"**可选优化**"(带它 1.47 s / 不带 1.58 s)。

---

## 4. 简化安装(目标第 4 条)

```bash
LEVEL=1 bash scripts/install_mainline.sh        # 默认:最小补丁 + 插件 + 自检
LEVEL=0 bash scripts/install_mainline.sh        # 纯插件:一个补丁都不打(能力受限,见 §6)
LEVEL=3 bash scripts/install_mainline.sh        # + A100/SM80 完整移植
DRY=1  bash scripts/install_mainline.sh         # 只打印将要做什么
```

三种自动模式:已装官方 wheel(补丁打到 site-packages)/ 已有源码树 / clone 主线。
结束前自动跑**启动自检**,并打印四个最常见的失败回退。

### 4.1 新增:启动自检 `scripts/check_mainline_env.sh`

把那些**缺失时不会报错、只会静默变慢(实测可达 30×)**的开关变成显式断言:

```bash
bash scripts/check_mainline_env.sh              # 当前 shell
TAG=myrun bash scripts/check_mainline_env.sh    # 某个已启动实例的 .env 记录
```

已接入 `serve_mainline.sh` 启动前(`CHECK=0` 跳过,`CHECK_STRICT=1` 不通过即拒绝启动)。

### 4.2 新增:接口契约 [`docs/PLUGIN_INTERFACE.md`](docs/PLUGIN_INTERFACE.md)

把**引擎哪些默认值由宿主节奏决定**写成可查的 10 条契约表(每条附证据与后果),
并记录两条会得出错误结论的**测量陷阱**。

### 4.3 新增:同机 A/B 回归基线 `scripts/ab_mainline_vs_fork.sh`

fork+我们引擎 vs 主线+插件,按 **TPOT** 口径出对照表(>1.5× 标退化)。

---

## 5. GPU 预填充(主线)

`--cudagraph-capture-sizes` 只列解码尺寸 + 阈值 + `MBT ≥ 阈值` **三个开关缺一不可**
(缺任一都会让 GPU 预填充**静默失效**)。主线实测:**1750 token → TTFT 3.379 s(606 t/s)**,
是纯 CPU 基线的 **2.6×**。详见 [`docs/GPU_PREFILL_MAINLINE.md`](docs/GPU_PREFILL_MAINLINE.md)。

---

## 6. 已知限制(诚实记录)

1. **TP=2 下不是逐位确定**:同一配置、同一 seed 连续两次 greedy,约 3/5 逐字符一致。
   经查**引擎本身逐位确定**(单 rank 11/11 一致),fork 栈 5/5 一致;
   分歧来自近似并列候选的 FP 次序,**语义等价**(最终答案一致)。
   ⇒ **验收口径**:数值以门禁(`OK=7 BAD=1`)为准,端到端以语义级抽样为准。
2. **内存仍比 fork 高 1.8×**(190 vs 105 GB/worker):C3 关掉了一半,
   剩余 ~85 GB 在引擎自身与缓存,未定位完。
3. **主线相对 fork 有 1.34× 的解码开销**,已量化,不再追。
4. **纯插件路径(`LEVEL=0`)不能用于真实服务**:缺 `pr1`(138 GB 专家无处安放)与 `pr0`
   (逐层构造会撞 5 分钟握手超时)。它只用于 import/注册冒烟。
5. **投机解码未在主线验证**:所有测试 `SPEC=0`;"draft model 永远在 GPU"这条硬约束
   在主线路径上尚未实测。
6. **原版 `lk_moe`(`lvllmds4-x`)对照未跑**:目前的对照是"fork + 我们的引擎"。

---

## 7. 升级建议

```bash
pip install -U vllm_xtu_moe-0.2.0-*.whl
bash scripts/check_mainline_env.sh      # 先过自检
ENV=<你的环境> PREFILL=1 RESIDENT=0-11 THREADS=60 bash scripts/serve_mainline.sh
```

**注意**:`serve_mainline.sh` 现在默认带上 `XIAOTU_MOE_ASYNC=0` / `NSLICE_SMALL=0` / `SPIN_IDLE_US=0`
—— 这三个缺失会让解码静默慢到不可用。自己拼命令行时务必带上(或用自检脚本确认)。

---

## 8. 验证环境

- 2×A100-40GB(PCIe)+ 2×EPYC 9654(192 核,NPS=4 / 8 NUMA node)+ 1.5 TB DDR5
- vLLM 主线 `6c73b08dec`;对照 fork `Lvllmds4-x` + 同一个 `xiaotu_moe` 引擎
- 全部数字:同机、同协议、同 `scripts/bench_lat.sh`
