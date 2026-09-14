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
| 补丁净效果 | `pr3` 7001 行 ⇒ 真正必需的只有 **~20 行**(mHC broadcast 的 DeepGEMM 门闸) |
| 复验 | 自检 10/10 · `OK=7 BAD=1 max_rel=1.873e-02`(**逐位一致**) · 确定性 11/11 · C=1/2/4 8/8 完成 |
| 性能 | C=1 TPOT **35.97 → 33.46 ms(−7%)**;相对 fork 的移植税 **1.34× → 1.25×** |
| 内存 | 每 worker **192.7 GB**(旧 190)⇒ EP 存储分片未退化 |

**下一次跟进的起点**:最新有轮子的 commit 已是 `d392ac836`(比 `dabc4362b` 又新 4 个 commit);
`upstream main HEAD` 仍**无轮子**。跑 `scripts/check_upstream_drift.sh` 即可看到当前值。
