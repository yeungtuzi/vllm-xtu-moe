# 铁律(固定规则 · 常驻文档 · 只增不改)

> **用途**:这个仓库**反复踩过**的坑和用户明确给出的规则,一律写在这里。
> 每次得出"架构级/物理级"结论(带宽、NUMA、线程绑定、布局、通信),**立刻**追加到本文件,
> 并在 `report/tuning/NOTES.md` 对应章节里链回。**不要靠记忆,先读本文件。**

---

## R1. 解码线程数:**每 CCD 4–5 核**,不要加核(用户 2026-09-11 明确指示)

* 本机 24 CCD ⇒ **96–120 线程**(不是 192,不是 168,不是 128)。
* 更多核**不会带来更多带宽**,只会抢 L3 与抢 power/散热预算,性能反而变差。
* 4 核/CCD = 96;5 核/CCD = 120。引擎默认 `n_ccd × 5`。
* **推论(重要)**:在"每 CCD 4–5 核"这个固定前提下,单核的**每字节指令数**就是性能的第二杠杆 ——
  因此"减少解码指令数"(如 LUT 展开/`vpermps`/VNNI)与线程规则**不冲突,而是互补**。

## R2. 内存架构:按 NUMA node 做 **TP 式切片**,每个 node **只读写本地内存**,跨 node 用类似 all-reduce 的方式合并

* **权重布局**:每个 NUMA node 持有 1/N 的专家行;**`mbind` 到本 node**;该 node 的 worker 只读本 node 的片。
* **线程**:worker 绑定到**本 node 的 CCD 核心**(slot-major/ccd-minor 顺序,保证一核一 CCD 轮流)。
* **通信**:各 node 的部分和在**引擎内部**合并(共享内存/同地址空间),**不是**每层一次 GPU all-reduce。
* ❌ **绝对不要再回到"单份连续拷贝(single-copy)"**:实测解码慢 ~30 倍(7/8 的读跨 node)。
* ❌ 不要把"切核"和"切 node"拆成两套不同划分(见 R5):会让跨 socket 流量爆炸。

## R3. 本机内存带宽的**正确**数字(2026-09-14 用户实测校正)

| 口径 | 带宽 |
|---|---|
| 理论峰值(24ch DDR5-4800 × 2 socket) | **> 900 GB/s** |
| **用户实测** | **~740 GB/s** |
| 我方 `numactl --interleave=all` 复测(48/96/192 线程) | 447 / 420 / 465 GB/s |
| ⚠️ 历史报告 `process_data/decomp/NUMA_BANDWIDTH_CCD.md`(2026-09-05) | 24 CCD **73.7 GB/s** |
| ⚠️ `mb/bandwidth_ccd` 空载复测(2026-09-14) | 12 CCD **178 GB/s**、24 CCD **238 GB/s** ← **同样是错的** |
| 我 2026-09-14 自写单节点探针 | 83–129 GB/s |

**⚠️ 上面两个"低数字"都是测量方式的错,不许再引用它们当上限**:

* 历史报告的测量环境写明 **`loadavg=155`,`机器共享,prod GPU0/1 正在跑 CPU-MoE 竞争`**
  ⇒ 那个 73.7 GB/s 是**被抢空后的**结果;
* 我自写探针在**单节点**上 `malloc+memset` 单线程首次触碰 ⇒ 所有页落在 1 个 node,读全跨 node;
* **`mb/bandwidth_ccd` 也有同样的缺陷**(用户 2026-09-14 指出):它的缓冲区由**主线程**分配,
  页面全部落在主线程所在 node,worker 只是"读"⇒ 量到的是**跨 socket 流量**,
  所以 178/238 GB/s 是"单 node 被跨 socket 读"的结果,**不是机器上限**。
  **正确做法:每个线程自己分配 + 自己首次触碰(写一遍)自己的缓冲,页才会落在本地 node。**

**正确测法**:`numactl --interleave=all`(或按 node 显式分片 + 线程绑到本 node),
然后对比 `4-5 核/CCD` 与"按 node 切片"两种配置。

## R4. 引擎当前**已满足** R1/R2(代码证据,便于回归检查)

| 规则 | 代码位置 |
|---|---|
| 按 node 切权重 + `mbind` | `csrc/moe/moe_v2.hpp::shard_region()`(`syscall(SYS_mbind, …, MPOL_BIND, 1<<node, …)`) |
| 每 node 的 worker 只拉本 node 的活 | `csrc/moe/numa_pool.hpp` 的 `worker_node_` + `node_ticket_[]`(分片路径 `parallel_for_sharded`) |
| 线程按 CCD 交错绑定(4/CCD) | `numa_pool.hpp::start_workers()` 的 slot-major/ccd-minor `cores_` + `pin_to()` |
| 跨 node/跨 rank 部分和合并 | `cpu_decode` 内的 EP 块(`/dev/shm` + 自旋 barrier);TP=1 时是引擎内部的 node 分片归约 |
| 多 rank 同机不抢核 | `XIAOTU_MOE_RANK_SPLIT=1`:按 **node 子集**切核(核互不重叠) |

## R5. 多 rank 同机的三种切法,**只有一种是对的**

| 模式 | 结果(实测) |
|---|---|
| `RANK_SPLIT=0`(两 rank 共用核表) | ❌ compute **9.80 ms/层**(TPOT 244 ms):两进程 pin 同 48 核 ⇒ 2× 超订 |
| `RANK_SPLIT=1`(按 node 子集切,**默认**) | ✅ compute **0.41 ms/层**;每 rank 覆盖 4 node ⇒ `nshard=4` |
| `RANK_SPLIT=2`(CCD 交错 + `nshard=8`) | ❌ compute **33.3 ms/层**:每 rank 铺满 8 node(含对端 socket)⇒ 跨 socket 爆炸 |

⇒ **内存放置与线程放置必须同构**;"切核"与"切 node"必须是同一套划分。

## R6. 度量纪律(否则会得出 R3 里那类错误结论)

1. 任何"带宽/上限"结论必须**在空载机器上**测,并写明当时的 `loadavg`;
2. 测内存必须**已经按 node 分片/交错**;单点分配测出来的数字只反映**一个 node**;
3. 结论要么进 NOTES(章节号)+ 本文件,要么不进 —— **不许只存在于上下文里**;
4. 每次"回退/失败"必须进 `report/tuning/TRIED_AND_REVERTED.md`。

---

## R7. **NPS=4 是本机 BIOS 选择,不是局部性边界;真正的边界是 socket**(用户 2026-09-14 提醒)

* 本机 8 个 NUMA node 来自 **NPS=4**;ACPI 距离矩阵:**同 socket 内 10/12/12/12,跨 socket 32**。
* NPS=4 相对 NPS=1 只是把**同 socket 内的本地距离**从 12 降到 10(收益很小),
  却把"同 socket 的 3 个 CCD 分成 4 个 node" ⇒ **增加了节点间(其实是同 socket 内)的通信/协调开销**。
  ⇒ **NPS=1 很可能更优**;而我们在不能重启改 BIOS 的前提下应当**用配置等价实现 NPS=1**:
  **按 socket 切片(2 份),每份内存交织在自己 socket 的 4 个 node 上**,而不是按 8 个 node 切。
* ⚠️ **注意一个曾经的误用**:`XIAOTU_MOE_NSHARD=2` 在 NPS=4 下**并不等于 NPS=1** ——
  现有实现是 `mask = 1UL << node`,即"shard n → 单个 node n";NSHARD=2 只会用 node0/node1
  (只用掉 1/4 内存通道)⇒ harness 实测 0.90 ms/层(比 NSHARD=8 的 0.38 慢 2.4×)。
  **正确做法**:shard 的内存用 **nodemask = 本 socket 的 4 个 node + MPOL_INTERLEAVE**,
  worker 用该 socket 的 12 个 CCD(48-60 线程,4-5 核/CCD)。
* 代码待办:`shard_region(total, node)` → 支持"node 集合 + 交织策略"(`XIAOTU_MOE_SOCKET_SHARD=1`),
  然后 A/B:`socket 2 片交织` vs `node 8 片各自绑定`。

---

## R8. **decode 的带宽账与本机真实上限**(来自 `process_data/ref/fork_vs_mainline_plugin_decision.md`,2026-09-08)

* DS-V4-Flash 单层全专家 fp4 = **3.221 GB**;单层激活(topk6)= **75.5 MB**;
  **每 token decode 读激活权重 ≈ 3.246 GB**(43 层)。
* 需要的 DRAM 读带宽:**21.5 tok/s ≈ 70 GB/s;50 tok/s ≈ 162 GB/s;100 tok/s ≈ 325 GB/s**。
* 本机上限:理论 921 GB/s;**现实 ~600-690 GB/s**;用户补充:**只读实测约 800 GB/s**。
  ⇒ **100 tok/s 只需要现实峰值的 ~50%,是完全可行的**。
* **重要区分(当年就写明了,别再搞混)**:
  - **原理上** decode 是"带宽受限 + 是否批处理"决定的;
  - **但当前实现并不是带宽受限**:实测只用到 ~45-200 GB/s,真正卡住的是
    **每层一次调用的延迟**(当年 2.3 ms/层 × 43 ≈ 100 ms/step;rank 切分修好后是 0.41 ms/层)
    加上 GPU 侧的 dense/attention。
  ⇒ **优化方向**:(a) 降每层关键路径延迟(MLP/指令数/pinned 往返);
  (b) 批处理(并发 = 每次调用吃更多字节,摊薄延迟);(c) 提高每线程 MLP。
* 任何"我们已经到带宽上限"的结论**必须先算出用了多少 GB/s 并与 600-800 GB/s 对比**,
  低于 400 GB/s 就不要说"到顶了"。

### 8.1 微基准工具(本仓/参考仓自带,别再自己造)

`process_data/ref/mb/`: `bandwidth_ccd`(⚠️ 主线程分配 ⇒ 量的是跨 node,**不可作上限**)、
`micro_mt_dram`、`micro_ilp`/`micro_ilp2`(ILP/MLP)、`micro_parallel`、`micro_multi`、
`micro_dram`(prefetch 距离)、`micro_ntile`、`micro_fanout`、`micro_gateup`、`test_sharded`。
**先跑这些,再谈结论。**

### 8.2 我自己写的探针也有同类陷阱

`malloc` + 单线程首次触碰 ⇒ 单 node;依赖链累加 ⇒ 只有 1 个在途 load ⇒ 量到的是**延迟**不是带宽。
写带宽探针必须:**每线程自己分配并首次触碰 + ≥8 条独立累加链**(`/tmp/bw6.c` 是正确写法)。

---

### R8 补充(2026-09-14,第 218 轮实测修正)

* **旧表述**:"引擎每层只吃到 200–250 GB/s = 机器上限(740)的 27–34% ⇒ 不是带宽受限"。
* **新表述(更准确)**:那 200–250 GB/s 是**摊薄后的表观值**。
  用 `DEDUP` 扫描(固定 FLOPs、只变字节)测出的**边际带宽 = 692–768 GB/s**,
  即 **权重流式读取已经在机器上限的 93–100%**。
  ⇒ **"让每字节搬得更快"这条线已经到顶,不要再去优化它。**
* **时间结构**(qlen=1,实测):`compute ≈ A + B×rows + bytes/740GB/s`
  * **A ≈ 0.124 ms/层(与字节、行数都无关的纯固定成本)** ← **当前唯一的大靶子**
  * B ≈ 0.0215 ms/行
  * 字节项:(DEDUP 1→6)67 MB / 0.091 ms = 736 GB/s
* ⇒ `A × 31 层 ≈ 3.8 ms/token`,几乎等于与参考的全部 C=1 差距(5.0 ms)。
* **线程伸缩也一样**:24/48/96/192 线程 → 0.645/0.419/0.344/0.333 ms(DEDUP=6)
  ⇒ **4 核/CCD 饱和**(R1 的干净曲线);多开线程只增加唤醒与竞争。


## R9. **分片绑定已被完整验证(2026-09-14)**:128/128 个 shard `mbind` 全部 `rc=0`

* harness:`NENGINES=8 × nshard=8 = 64` 个 shard,`XIAOTU_MOE_SHARD_DIAG=1` 输出 **`rc=0` × 64**;
  另一次 8 引擎 × 8 shard 的统计为 **128 行 `rc=0`、0 失败** ⇒ **每个 shard 都真的绑到了目标 node**。
* 活体服务(`/proc/<worker>/numa_maps`)也证实:`bind:0/1/2/3` 各 3 GiB(**严格本地**)。
* **lk 的 `LVLLM_ENABLE_NUMA_INTERLEAVE=1` 只作用于"非 shard 的分配"**(实测 `interleave:0-7` 15 GiB,
  是**重复的源权重副本**,不在热路径上)。⇒ **按用户指示:忽略该环境变量**,我们的 `mbind` 优先。
* **内存重复**:引擎会把权重复制进分片区(`COPY the weight blocks`),而源张量仍然驻留
  ⇒ 单层实测 RSS 3.67(源)+ 3.29(分片)= **6.96 GiB ≈ 2×**;服务里 worker RSS **114 GB/rank ≈ 2.2×**。
  这是**主机内存浪费**(不影响热路径带宽,因为热路径只读绑定的分片),
  但省下来的内存**换不到显存**,所以对"常驻层"没有直接帮助 —— 记下来避免重复误判。

---

## R10. **上游主线漂移:周期性检查 + 及时跟进**(用户 2026-09-14 定为开发原则)

> 用户原话:「你要周期性检查上游主线是否升级,并及时跟进,把这条作为开发原则」。
> 入口脚本:**`scripts/check_upstream_drift.sh`**(每次开工先跑,`--full` 看逐文件热度)。

### 10.1 为什么必须周期做(实测,不是担心)

* vLLM 主线移动**极快**:我们上次的基线 `6c73b08dec`(2026-09-08)→ `dabc4362b`(**2026-09-14**)
  只有 6 天,却差了 **346 个 commit / 300 个文件 / 1307 个文件被上游改过**。
* **DS-V4.1-Flash 的支持就是在这 6 天里落地的**(`deepseek_v41` 包 + registry +
  `DSparkV41DraftModel`,PR #56503 起)。**不跟进 = 白写别人已经写完的东西。**
* 典型漂移形态(两条都真实打断过服务):
  1. **符号搬家**:`cpu_moe.select_experts` → `router/cpu_router.select_experts`
     (模块还在,名字没了 ⇒ `ImportError`);
  2. **构造函数加参数**:`DeepseekV4MoE.__init__` 新增 `num_hash_layers=`
     ⇒ `TypeError: CpuXiaotuMoE.__init__() got an unexpected keyword argument`。
  ⇒ 插件侧的固定写法:**能兼容两版就兼容两版**(`try/except ImportError`、
  `*` 关键字参数带 `None` 默认并回退到 config),而不是绑死一个版本。

### 10.2 ⚠️ 本机最硬的约束:**不能从源码编译**,"上游支持 X" ≠ "我们能跑 X"

* 本机 `nvcc` 是 **CUDA 12.1**,而 torch 是 **2.13.0+cu130**(CUDA 13.0)⇒ **版本不匹配**;
  且**没有 Rust 工具链**(`cargo`/`rustc` 缺失),而主线有上百个 rust 文件。
* 当前 env 的 vLLM 是 **editable + precompiled wheel** 安装(`.so` 由官方 nightly 轮子提供)。
* ⇒ **升级目标必须是"已发布 precompiled wheel 的 commit"**,不是"上游 HEAD"。
  查询方式:
  ```bash
  curl -s https://wheels.vllm.ai/nightly/cu130/vllm/metadata.json | grep -o '+g[0-9a-f]*'
  # 某个具体 commit 有没有轮子:
  curl -s -o /dev/null -w '%{http_code}\n' https://wheels.vllm.ai/<full-sha>/cu130/vllm/metadata.json
  ```
  实测:`bdad63c9`(当时 HEAD)= **404 无轮子**;`dabc4362b`(nightly)= **有 cu130 轮子**。

### 10.3 标准跟进流程(每次升级照做)

```bash
R=/home/user/lvllm/vllm-xiaotu-moe; M=/home/user/lvllm/process_data/ref/repos/vllm-mainline

# 0) 先跑漂移检查(会告诉你 HEAD 能不能装、补丁还打不打得动、API 面是否完好)
bash $R/scripts/check_upstream_drift.sh

# 1) **先做安全快照**(本地补丁是未提交的工作树改动,极易丢!)
cd $M && git diff > $R/backup/<date>/xtu-working-tree-<base>.diff
git checkout -b xtu/snapshot-<base> && git add -u && git commit -m "snapshot: <base>"

# 2) 切到目标 commit(必须是有轮子的那个)
git fetch origin main && git checkout -b xtu/upgrade-<target> <target-full-sha>

# 3) 用官方轮子刷新二进制(不改依赖;缺 setuptools-rust 就补装,它是官方 build 依赖)
VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm-<target>.whl \
  pip install -e . --no-deps --no-build-isolation

# 4) 复验(顺序:便宜 → 贵)
python -m vllm_xiaotu_moe.mainline_shims    # 垫片是否全部命中
python scripts/probe_oracle.py             # oracle 是否仍选中我们的 CPU 后端
bash scripts/check_mainline_env.sh         # 9 条宿主契约断言
python scripts/test_block23_equiv.py       # 数值门禁 OK=7 BAD=1
ENV=... TAG=... bash scripts/serve_mainline.sh   # 端到端(最贵,最后做)
```

### 10.4 升级后必须写下的三件事

1. `report/tuning/NOTES.md` 追加章节:**旧基线 → 新基线、差多少 commit、打断了什么、
   怎么修的、before/after 数字**;
2. `docs/UPSTREAM_DRIFT.md` 的 **BASE 与"三棵树"表**跟着更新(否则下次审计读到过期基线);
3. 本文件 R10.2 的**轮子可用性**结论(哪个 commit 能装、哪个不能)。

### 10.5 补丁集随漂移"减法优先"

上游每往前一步,我们自己要背的补丁就应当**变少而不是变多**。实测例子:

* `pr0`(握手超时)在 `dabc4362b` 上**仍然必需** —— 上游还是硬编码
  `HANDSHAKE_TIMEOUT_MINS = 5`,而 CPU 引擎逐层构造要 ~6 min;
* `pr1` 的 **mxfp4 oracle 分派**已被上游自己吸收(native `Mxfp4MoeBackend.CPU`
  分支 + `prepare_mxfp4_moe_layer_for_cpu`),只剩 plug in 侧的 shim 还需要;
* `pr2`/`pr3`(SM80 移植)是否还必需,**取决于主线对 A100 的支持现状**,每次升级都要重问。

⇒ **不要默认"补丁越多越安全";每升一次就问一次"这条上游做了吗"。**


### 10.6 首次执行的实测记录(2026-09-14,可直接照抄)

| 步骤 | 结果 |
|---|---|
| 漂移检查 | `6c73b08dec → dabc4362b` = **346 commits / 300 文件**;上游 HEAD `bdad63c9` **无轮子** |
| 升级目标 | `dabc4362b`(nightly,有 cu130 轮子);装完 = `vllm 0.29.1rc1.dev95+gdabc4362b` |
| 冲突面 | 31 个自研文件里 **21 个与上游冲突**;**7 个** `cherry-pick` 冲突 |
| 真被打断的 API | 4 类(符号搬家 / 构造函数加参 / kernel 重构 / **上游自己实现了同一功能**) |
| 补丁净效果 | `pr1` 少了 1 个 hunk(上游吸收 mxfp4);mHC 冲突 43 行 → **~20 行门闸**;`pr3` 整体仍 **7512 行**(含新增 SM80 内核文件) |
| 复验 | 自检 10/10 · `OK=7 BAD=1 max_rel=1.873e-02`(**逐位一致**) · 确定性 11/11 · C=1/2/4 8/8 完成 |
| 性能 | C=1 TPOT **35.97 → 33.46 ms(−7%)**;相对 fork 的移植税 **1.34× → 1.25×** |
| 内存 | 每 worker **192.7 GB**(旧 190)⇒ EP 存储分片未退化 |

**下一次跟进的起点**:最新有轮子的 commit 已是 `d392ac836`(比 `dabc4362b` 又新 4 个 commit);
`upstream main HEAD` 仍**无轮子**。跑 `scripts/check_upstream_drift.sh` 即可看到当前值。

## R11. **ngram/Engram 一律"最后加载";大权重必须"逐层加载、切片、释放"**(用户 2026-09-15 定为开发原则)

用户原话要求,分两条:

1. **凡是带 ngram(Engram)的模型,ngram 必须放到**最后**才加载进内存。**
   理由(本机实测,`cellK`,见 NOTES §394):`ParallelEngramEmbedding.__init__` 在
   **模块构造期**就分配了 `94.42 GiB × 2 = 188.8 GiB` 的 **pinned**(不可回收、不可换出)表,
   而专家层是在那之后才逐层处理的:
   ```
   行44/46  engram.py:255  Engram table offloaded ... 94.42 GiB ×2   (10:08:40 / 10:10:17)
   行53…147 released 6.59 GiB × 40 layers (layers.0 … layers.39)      (专家阶段在其后)
   ```
   ⇒ 整个**内存最吃紧的专家阶段**,白白多扛 188.8 GiB 不可回收内存。
   把 ngram 挪到最后,峰值直接少 188.8 GiB。

2. **大权重不允许"一次性全量加载"。** 必须**逐层** load → slice → release。
   用户明确:如果 vLLM 一定要一次性把全部权重读进来(对我们不合适),
   就**给 vLLM 打 patch**:识别到我们标记的参数时,按层做 加载/切片/释放,
   并**作为单独的 PR 提交**。

配套(已具备的基础):
* 标记机制:`create_weights` shim 会给混合模式下建在 CPU 的**大**参数(≥64 MiB)打上
  `_xiaotu_cpu_expert`(见 `mainline_shims.py`),这正是 patch 里识别"我们的参数"的依据;
* `device_loading_context` shim 已经阻止了主线把 CPU 专家搬上 GPU(NOTES §390);
* 逐层释放已在 `apply()`/`process_weights_after_loading` 两处生效,实测 40/40 层
  各释放 6.59 GiB(NOTES §391/§392)。

## R-VRAM(2026-09-16). **显存分配的优先级顺序**(用户明确指示,以后一律照此办理)

在**任何**显存不足/需要取舍的场合,按下面这个**固定顺序**分配,前面的必须先满足:

| 优先级 | 项目 | 不满足时的 fallback |
|---|---|---|
| **1** | **保证 1M 上下文**(KV cache 必须先按 `max_model_len` 留够) | —— **不可降级**;它不满足就没有下一项 |
| **2** | **GPU 预填充** | 退回 **CPU 预填充** |
| **3** | **GPU 投机解码**(DSpark drafter 放 GPU) | **关闭投机解码** |
| **4** | **尽可能多的专家层常驻 GPU**(放收益最高的那些) | **全部放主机内存**(常驻 = 0) |

### 两条硬约束(与优先级同等重要)
* **总约束:不允许额外多占系统内存** —— 任何"为了 GPU 功能而在主机侧多留一份"的做法都不允许。
  **⚠️ 本条曾被我用错,已更正(见 R-VERIFY)**:我一度据此判定"GPU 预填充不可用",
  理由是"它会 +253 GiB/rank"。**那是错的** —— GPU 预填充的活跃路径
  (`kmajor_from_engine_shards → copy_hostbuf_to_device`)直接从**引擎分片**填 K-major,
  主机侧代价 **0**;显存侧是 **ping/pong 双槽 ≈ 两层权重**。
  那 253 GiB/rank 是我自己把 `XIAOTU_GPUPREFILL_WCOPY` 与"GPU 预填充开启"绑在一起造成的,
  现已把该默认改回**恒 0**(只在兜底路径才建副本,并打告警)。
* **优先级 4 的"收益最高的层"(已按报告原文更正)**:
  * CED 报告关于激活量的原话是"**16B/token during decode** but only **8B during prefill**"
    ⇒ **预填充**只跑前一半(编码器 20 层,见下条);**解码每 token 都要跑全部 40 层**
    (解码要算新 token 的 `h_{L/2}` 才能把解码器各层的 KV 投影出来)。
    所以**解码阶段各层收益基本均匀**,"只放解码器"没有依据 —— 我先前那句判断是错的,已更正;
  * 若 GPU 预填充开启,预填充只覆盖**编码器(0-19)**(+ 解码器 128 token 的 SWA 重放),
    编码器那半在"预填充 + 解码"里都出现 ⇒ 此时**编码器(0-19)略微优先**;
  * 若 GPU 预填充关闭(当前策略下的默认),预填充走 CPU 覆盖全部层 ⇒ 收益完全均匀,
    **放哪 N 层都一样**。
* **CED(只属于 V4.1)** 的事实与是否实现,见 §506;V4.0(0731)没有 CED ⇒ 预填充必须跑全部 43 层。

### 实测的账本(TP=2,单卡 40 GB,用来执行本规则)
| 项 | 单卡代价 |
|---|---|
| 模型权重 | ~7.4 GB |
| **1M 上下文 KV** | **~17.5 GiB**(实测:`17.45 GiB → 1,116,005 tokens`,≈16.4 KiB/token) |
| GPU 预填充(`MBT=8192`) | 激活数 GB(**未知精确值**,待测) |
| 每层专家常驻 | **3.36 GiB/层**(= 6.72/TP) |
| DSpark drafter(`mtp`) | 7.39 GiB 全量 ⇒ **~3.7 GiB/rank**(TP=2) |

⇒ 按本规则执行时,**先把 17.5 GiB 的 KV 锁死**,再依次试 预填充 / 投机 / 常驻;
上限算式:`7.4 + 17.5 + 预填充激活 + 投机 + 3.36×N ≤ 39.5`。

## R-VERIFY(2026-09-16). **判断"一项能力的代价/可行性"必须读活跃调用路径** —— 不许从 env 开关或我自己的默认值反推

**踩的坑(严重,用户当场纠正)**:我写了"GPU 预填充今天必须判为不可用,因为它会给 K-major 缓存
再复制一份权重(+253 GiB/rank)"。**错。** 真相:
* GPU 预填充的**活跃路径**是 `mixed_experts.py:1278 → gpu_prefill.kmajor_from_engine_shards()`
  → `engine.copy_hostbuf_to_device()`:K-major 目标缓冲**直接从引擎自己的分片**填,
  **主机侧代价 0**;
* 配合 `gpu_prefill.prefetch_layer()` 的 **ping/pong 双槽**(`nslots=max(2, XIAOTU_MOE_PREFETCH_SLOTS)`,
  每槽 ≈ 一层权重)与 `_prefetch_stream` 的异步 H2D ⇒ **显存代价就是"两层"**;
* 那份 Python 权重副本(`XIAOTU_GPUPREFILL_WCOPY`)**只服务兜底路径**(引擎没有分片时用 checkpoint 源张量),
  而且**是我上一轮自己加的条件默认**把它和"GPU 预填充开启"绑在一起的
  ⇒ **那 253 GiB/rank 是我自己造出来的代价,然后我把自己的产物当成了系统的必要条件。**

### 两条强制做法
1. **要判断"某功能贵不贵/能不能开",去读它被调用时真正走的那条路**(`grep` 调用点 → 逐层展开),
   而不是看"有哪些 env 开关"、更不是看"我上一轮设的默认值"。
2. **不许把"压缩后的会话摘要 / NOTES 里的旧结论"当作现状**。摘要里写着"ping/pong 未实现",
   而代码里 `prefetch_layer` 的双槽环 + 异步 H2D 一直都在 ⇒ **结论必须重新落回当前代码验证**,
   尤其当你要用它去**关掉**一个用户明确要的功能时。

### 推论(立即生效)
* R-VRAM 的优先级 2(GPU 预填充)**不再被"不得额外占系统内存"挡住** —— 它本来就是零主机代价;
* `XIAOTU_GPUPREFILL_WCOPY` 默认**恒 0**;真走到兜底路径时打**显式告警**;
* 规划器 `GPU_PREFILL_HOST_GIB = 0`;1M 上下文下的规划变为
  **GPU 预填充 ✅ + GPU 投机 ✅ + 常驻 2 层(20-21)**。

### 第三条强制做法(2026-09-17 追加):**复刻一次历史测量,必须连 env 一起复刻;不许用"日志里没有 env 回显"推断"没设 env"**
* 踩的坑:复刻 cellV(24.94 t/s)时只复刻了命令行旗标、没设 `VLLM_EXPERTS_LOAD_DEVICE=cpu`
  ⇒ 专家权重建到 GPU 上,装载期 OOM(`Tried to allocate 4.22 GiB`),**看起来像代码退化**。
* 更隐蔽的一次:cellV 打印 `[cd-timing]`(sync 路径)而我复刻出来的是 `[cd-timing/async]`
  ⇒ **cellV 的 env 里一定有 `XIAOTU_MOE_ASYNC=0`**。插件只回显它从 `XIAOTU_ENV_FILE`
  **补进来**的键 ⇒ **没有回显 ≠ 没有设置**(直接放在进程 env 里的键不会被回显)。
* 结论:任何"历史数字 vs 今天数字"的对照,开跑前先列**三张清单**:
  ①命令行旗标 ②**进程 env(含隐含默认值** —— 如 `xiaotu_async_enabled()` 未设即 true**)** ③机器状态(NPS/NUMA、是否有别的服务占核)。
* **反面教材(同一天)**:`binding.cpp:68-72` 注释写着"async 默认开启且是最大单点收益
  (6.53→1.80 ms/token)",而 2026-09-17 在 TP=1/qlen=1 实测 **async 2.05 ms/层 vs sync 1.05 ms/层**
  ⇒ **注释不是证据,实测才是**;发现这种矛盾要立刻改注释或改默认值,别让它再骗下一个人。

## R12/JIT(2026-09-17,用户要求"把 JIT 的固定缓冲目录加上,不然每次都重新 JIT 太可怕了"). **编译缓存必须钉在固定目录;JIT 开销必须与"改了旗标/改了代码"解耦**

**机制(在安装好的 vLLM 源码里逐行核实,不是推测)**:
1. torch.compile 缓存目录 = `$VLLM_CACHE_ROOT/torch_compile_cache/<hash10>`,
   `hash10 = sha256([env_hash, config_hash, code_hash, compiler_hash])[:10]`
   (`vllm/compilation/backends.py:1028-1067`):
   * `env_hash`  = **每一个已知的 `VLLM_*` 环境变量**(`vllm/envs.py:compile_factors()`,白名单极小);
   * `config_hash` = `vllm_config.compute_hash()`(maxlen / MBT / seqs / cudagraph 模式 …);
   * `code_hash` = **被 trace 的 Python 源码内容**,**含我们插件替换的模型类**。
   ⇒ 改**一个**旗标、或改**一行**我们的代码 ⇒ 目录换新 ⇒ **逐形状重新 JIT**
   (`jit_monitor` 报的 20-60 s/形状 就是这个,生产环境首次遇到新长度会卡住)。
2. 更隐蔽:`CompilerInterface.initialize_cache()`
   (`vllm/compilation/compiler_interface.py:470-481`)会**把 `TRITON_CACHE_DIR` 重定向**到
   `<cache_dir>/triton_cache`,即那个 hash 目录 ⇒ **`~/.triton/cache` 里攒下的几千个内核根本用不上**。
   (`mode=NONE` 时不走这一步,所以只有**编译路径**坏掉 —— 这也是本机 `~/.triton/cache` 有 2814 个
   内核却仍反复重编的原因。)

**做法**:`scripts/lib_jitcache.sh`。`JITCACHE=1`(默认)时给编译缓存钉一个稳定目录
`…/torch_compile_cache/xtu-<模型>-tp<TP>-<编译模式>-<vLLM commit>-<我们源码 sha1 前 8 位>`,
并把 Triton/Inductor/TileLang 指到它下面:
* **旗标扫描、重复启动 ⇒ 命中同一目录**(这是要解决的问题);
* **我们自己的源码变了、或上游 vLLM 变了 ⇒ 目录自动换新**(不会读到陈旧计算图);
* `JITCACHE=0` 回到 vLLM 默认的 hash 行为;`JITCACHE_STAMP=<串>` 强制重建;
* 只在**编译路径**(`JITCACHE_COMPILING=1`)上改那三个 env —— `mode=NONE` 时不要动
  Triton 的默认目录(它本来就固定持久,动它只会白丢一份缓存)。

**纪律**:任何"启动慢/首个请求卡住"的归因,**先看这一条**;报告 JIT 开销时必须写明
缓存目录与它是否命中,不许把"每次都要重编"当成模型或引擎的固有代价。


## R13/NUMA(2026-09-17,用户提出"修正后的原则"). **切片按内存域、核心分组不出组** —— 两条都被实测钉死,但措辞必须精确,否则会重犯 §505 的错

### 原则一(切片):**主判据是"分片单位 = 内存域(NUMA node)",不是"访存距离"**
原文"按 numactl 设置、按访存距离、尊重 NPS 决定切片方案,且必须确保本地计算的权重切到本地内存"
—— 结论**同意**,但三处必须精确化:

1. **不要把距离当主判据**。§505 的根因正是拿距离当判据("socket 内 10/12 很小 ⇒ 只到 socket 就够")。
   §520 直接反证:把 socket 分组**正确地**实现出来(每次读距离都 ≤12)后,仍比 node 分片慢 **2.12×**。
   距离只配当 tiebreaker;真正的判据是 **`nshard_ == 该 rank 拥有的域数`,且每个 worker
   **只处理自己所在域的那一片**"(= 原话里"本地计算的权重切到本地内存")。**NPS 决定域数**
   ⇒ "尊重 NPS"成立,§512 的"改 NPS 无效"已被 §519 的自适应默认推翻。
2. **"按 numactl 的设置"要限定**:权重缓冲**不许**依赖进程级 `numactl --interleave=all`
   (我们的 vLLM 父进程就是它)—— 引擎必须对每个分片**显式 `mbind`**,否则域数与局部性
   都被 interleave 隐式抹平。当前实现是显式 mbind ✓。
3. **必须有回退层级**(因为"一片=一域"并非总可满足,如 NPS=1+TP=2 ⇒ 域/rank = 1):
   * ①首选:`1 分片 = 1 域`,`worker 只读本域`(实测最快);
   * ②若分片必然比域粗:该分片用 **`MPOL_INTERLEAVE` 铺到它所辖的全部域**
     (实测比"粗分片绑单个域"快 **2.22×**:8.63→3.88 ms/层);
   * ③**禁止**:"粗分片绑单域 + worker 跨域" —— §505 恰好是这一档,也是它最坏的一档。

### 原则二(核心分组):**"核心不出组"必须与分片同单位;交错要铺满"本组拥有的全部 CCD"**
原文"按 TP 数量做核心分组,组内按启用核数交错分配(至少留 2 核空闲),确保核心不出组 &
均匀分布到所有 CCD" —— 前半句**同意且必需**(单位不配对的代价实测 **9.02 ms/层**,比错的默认还慢 9×),
但后半句必须改:

* **"均匀分布到所有 CCD" 与 "核心不出组" 会冲突**:分片按 node 子集切时,一个 rank 只拥有
  该子集的 CCD。要写成 **"组内交错、均匀铺满该组拥有的全部 CCD"**。
  现状(RANK_SPLIT=1,rank0 = node 0-3 = **12 个 CCD**)60 线程 = 12×5 ⇒ 正好铺满本组 12 个 CCD,
  每 CCD 5 核、`cores_` 是 CCD-first 排序 ⇒ 天然均匀;强行铺到全机 24 CCD 就是"出组"。
* 每 CCD 4-5 核、至少留 2 核空闲:2 rank × 60 = 120 / 192 ✓ 满足。
* (注:单进程微基准确在 144-192 线程更好,那是"一台机一个引擎"的场景,不能搬到 TP=2 服务。)

### 本规则的实测依据(全部同机同模型同日,只动一个旋钮)
| 切片 | 核分组 | B=8 ms/层 | 服务 C=1 t/s |
|---|---|---|---|
| socket 分片(单 node 绑定) | CCD 交错 | 8.63 | 14.17 |
| socket 分片 + 组内交织到整个 socket | CCD 交错 | 3.88 | — |
| **node 分片** | **node 子集** | **1.83** | **18.62** |
| node 分片 | CCD 交错(出组) | — | 2.14(9.02 ms/层) |

### R6 补充(2026-09-17,轮 53). **性能判据一律用"同日、同参、同机的对照比值",不许用跨会话的绝对毫秒**
* 事实链:§505 的分片回归(引擎每层 0.44→0.96 ms)**通过了**当时的性能门 —— 因为①门禁用
  `DEDUP` 压缩工作集,把"分片只落在 2 个 node 上"这类效应量不出来(§518);②2026-09-16 又把
  绝对阈值从 0.70 放宽到 1.15,于是"绝对 1.16"过关,而**同日比值其实是 1.43×**(§525)。
* 另一面:跨会话比绝对毫秒也会冤枉代码 —— 2026-09-17 实测**同日 lk_moe 自己就慢了 1.14×**
  (环境漂移),若只看我们的 0.66→0.93 会得出"我们退了 1.4×"的错误结论(§522)。
* ⇒ **做法**(已落进 `scripts/check_engine_aligned.sh`):每个 `DEDUP` 档**同时**跑同日 lk 对照,
  按 **xiaotu/lk 比值**判门(`RATIO_MAX` 默认 **1.20**,取自 §119 记录的 1.16-1.23×);
  绝对 ms 仍打印,只用于和 §119 等历史数字对照,**不再单独作为判据**。
* 推论:**任何"我们对 lk/lvllm 慢了 N×"的结论,必须给出同机同日同参的对照**;
  跨会话/跨机器/跨 prompt 集的数字只能当线索(§511/§521 各栽过一次)。
