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
| 我 2026-09-14 自写单节点探针 | 83–129 GB/s |

**⚠️ 上面两个"低数字"都是测量方式的错,不许再引用它们当上限**:

* 历史报告的测量环境写明 **`loadavg=155`,`机器共享,prod GPU0/1 正在跑 CPU-MoE 竞争`**
  ⇒ 那个 73.7 GB/s 是**被抢空后的**结果;
* 我自写探针在**单节点**上 `malloc+memset` 单线程首次触碰 ⇒ 所有页落在 1 个 node,读全跨 node。

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
