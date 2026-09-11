# 试过并回退的尝试登记簿(TRIED & REVERTED)

> **用途(用户 2026-09-11 明确要求)**:每一次「试了 → 发现不好 → 回退」都必须在这里
> 追加一条记录,写清楚:**试了什么 / 当时为什么觉得可能有用 / 实测数据 / 结论 /
> 什么条件下才允许再试**。下次会话先读本文件,不要再重复已经被否掉的尝试。
>
> 协议:
> 1. 只记录**已经回退或已删除**的做法;仍在用的写在 `report/tuning/NOTES.md`。
> 2. 条目编号 `R<n>`,只增不改号;结论变了就新增一条,并注明取代了哪条。
> 3. 必须带**实测数字**(同一时间窗口交错 A/B,或 min-of-N),不许写"感觉更慢"。
> 4. 回退时把"陷阱注释"也写进代码原处,指向本条编号(如 `见 TRIED_AND_REVERTED R1`)。
> 5. 相关的待办状态同步到 `notes/internal-docs/BACKLOG.md`(那里是唯一总账)。

关联文档:`report/tuning/NOTES.md`(已验证的做法与实验记录)、
`docs/PERFORMANCE_OPTIMIZATION.md`(优化史)、`notes/internal-docs/BACKLOG.md`(待办总账)。

---

## R1. 权重「单份连续拷贝 / 关闭 NUMA 分片」(`XIAOTU_MOE_SINGLECOPY`)— 已删除,永远不要再加

| 项 | 内容 |
|---|---|
| 试了什么 | 让引擎只保留一份连续权重(`nshard_=0`),不分片、不做 socket 副本;"省内存" |
| 当时理由 | 旧注释称"分片会多一份 ~136G 拷贝、MoE 读权重不是 conc-4 瓶颈" |
| 实测(同窗口交错,`bench_cpu_engine.py`,B=6/DEDUP=12/THREADS=120) | 单拷贝 **2.01 ms/层** vs 分片 **1.19 ms/层** = 分片快 **1.67×**;A(读权重)单拷贝冷启 20.3 ms/次 vs 分片 0.65 ms/次;B=18 时 1.15× |
| 机制 | 单拷贝 ⇒ 全部 120 线程读**同一份物理内存**,7/8 的访问跨 node(distance 32);分片 ⇒ 每个 node 只读 `MPOL_BIND` 到自己那份,全部 page-local |
| 回退动作 | 引擎里整段分支**删除**;`tune_serve.sh` / `serve_prod_8070.sh` / `tune_nsys.sh` / `fp8_*` / `tiny_moe_equiv.py` / `README*.md` / `docs/*` 的全部引用清除 |
| 再试条件 | **没有**。内存不够时用 R2(THP)或自动安全网(每 socket 副本),不许恢复单拷贝 |

**教训**:这个开关曾被 `tune_serve.sh` 默认设为 `1`,让交付配置悄悄退化成最慢布局。
不要用"省内存"作为推翻 NUMA 分片的理由——分片本身就是 1 份内存(见 R2)。

## R2. 分片区域用透明大页(`MADV_HUGEPAGE`)— 默认已改为关

| 项 | 内容 |
|---|---|
| 试了什么 | 对每个分片 region `madvise(MADV_HUGEPAGE)`,理由是"不落大页会 thrash 4KB TLB" |
| 实测(同窗口交错,B=6/THREADS=120) | ms/层:**关** 1.20 / 1.20 / 1.23 vs **开** 1.20 / 1.28 / 1.32;单层 RSS:**关 6.9 GB vs 开 18.9 GB(2.7×)**;逐 node:`node7=5.0GB,node2/6≈3.0GB,其余≈1.5GB`(名义 402MB/node) |
| 机制 | 每个 node 只拥有每个专家的一段**稀疏跨度**(w13:8.4MB stride 里 2×512KB;w2:4.2MB stride 里 1×512KB),2MB 大页把跨度覆盖到的整页都落地 ⇒ 放大 5-6× 且各 node 不均 ⇒ 29 层就把单 node 的 193GB 吃光(`CONSTRAINT_MEMORY_POLICY` OOM) |
| 回退动作 | 默认改为**不调用** `madvise`;仅 `XIAOTU_MOE_SHARD_HUGEPAGE=1` 可复现旧行为(调试用) |
| 再试条件 | 只有当分片跨度变成 **2MB 对齐且连续**(例如每个 node 拥有整专家)时,才值得重测;否则不要 |

## R3. 分片绑定用 `set_mempolicy()` — 必须用 `mbind()`(已修)

| 项 | 内容 |
|---|---|
| 试了什么 | `set_mempolicy(MPOL_BIND, node n)` + 拷完后 `MPOL_DEFAULT` 复位 |
| 实测 | 启动到第 29-30 层被内核杀:`oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=7, task=VLLM::EngineCor, anon-rss 593GB`(整机只用了 600GB,是**单 node** 被打爆) |
| 机制 | `set_mempolicy` 是**线程**策略,会被之后新建的线程**继承**;引擎构建期间 vLLM 仍在起线程(如 pinned 权重缓存),它们在窗口内的分配被绑到单个 node |
| 回退动作 | `shard_region()` 与 `numa_socket_alloc()` 全部改成 `mbind(addr, len, ...)`(只作用于这段映射) |
| 再试条件 | 没有。任何时候都不要用 `set_mempolicy` 做权重布局 |

## R4. 后台 pinned 预构建撞上 CUDA graph 捕获 — 已改为"转置/锁页分离 + 等捕获静止"

| 项 | 内容 |
|---|---|
| 试了什么 | 首次大 batch warmup forward 时启动后台线程,并行(6 worker)做 K-major 转置 + `cudaHostRegister` |
| 实测(3 次启动失败) | ① 捕获期间调用 ⇒ `cudaErrorStreamCaptureInvalidated` ⇒ `RuntimeError: Engine core initialization failed`;② 同一调用还会让 **libcuda segfault**(`cuMemHostRegister_v2` → SIGSEGV,EngineCore 被杀);③ 改成"等捕获结束再开始"后,90s 宽限期在 vLLM 的 warmup→捕获(~100s)时间线之前到期,仍然撞上 |
| 机制 | 捕获窗口由 vLLM 在 warmup 之后约 100s 打开(主模型 + DSpark 投机器各一次);已经在驱动内部执行的注册调用无法撤回 |
| 回退动作 | 两阶段:阶段 1 = 纯 CPU 的 K-major 转置(任何时刻都安全)在后台并行预建;阶段 2 = 等"捕获静止 `quiet_s=10s`"后再并行 `cudaHostRegister`;服务路径命中未锁页的缓存项时就地锁页(服务期没有捕获) |
| 结果 | 两阶段拆分后启动 **285s 就绪**(此前 wait 版本 8 分钟以上且仍撞捕获),日志无 `prebuild * failed` / `StreamCaptureInvalidated` / `SIGSEGV` |
| 再试条件 | 不要把 `cudaHostRegister`/`pin_memory` 放回"捕获可能正在进行的窗口";纯 CPU 部分可以 |

## R5. `XIAOTU_MOE_DPBF16`(bf16 点积累加路径)— 保持默认关

| 项 | 内容 |
|---|---|
| 实测 | 只快 **1.10×**;数值门槛不过(me=2 时 max_rel **8.4e-3** > 门槛 2e-3) |
| 结论 | 代码保留、默认关;不要为了 1.1× 拿数值精度冒险 |

## R6. `XIAOTU_MOE_NOSHARD=1`(每 socket 一份完整副本)— 只做自动安全网

| 项 | 内容 |
|---|---|
| 实测/机制 | 内存 ×2,读放大的只是"跨 node"而不是"跨 socket";分片方案严格更优 |
| 结论 | 保留为"分片不可用(维度不整除/分配失败)"时的自动兜底,**不要**手工启用 |

## R7. 让 CPU 引擎做预填充(prefill)— 不可用

| 项 | 内容 |
|---|---|
| 实测 | **19 t/s**(4096 token 用了 214s)vs GPU 逐层流式 **586-671 t/s**;NS-PROF 显示每层 setup **170-340 ms**(gather 201 MB + fp32 转换 340 MB) |
| 结论 | 预填充一律走 GPU 流式(`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`,默认 384);不要把 CPU 路径用于 prefill |

## R8. 把线程数加到 192(全核)— 更慢

| 项 | 内容 |
|---|---|
| 实测 | 96 / 120 / 128 线程基本等价;192 线程 **慢 18%**(L3/热预算/调度) |
| 规则 | **每 CCD 4-5 个核**(本机 24 CCD ⇒ `XIAOTU_MOE_THREADS=120`);不要再加核 |

## R9. 池自旋窗口调大到 5000 µs — 更慢

| 项 | 内容 |
|---|---|
| 实测 | `XIAOTU_MOE_SPIN_IDLE_US` 5000 → `rest` 变差;300 µs 使 rest 降 3.5×、端到端 +28-60% |
| 结论 | 默认 300 µs;不要为了"减少唤醒"调大 |

## R10. 投机 token 数 5 → 3 — 拒绝(k 保持 5)

| 项 | 内容 |
|---|---|
| 实测 | 截断到 3 会让 **pos-0 接受率 0.67 → 0.30**(dspark 是并行起草,训练时 `dspark_block_size=5`);整体接受长度 2.63-3.64 ≥ 生产 3.01 |
| 结论 | `num_speculative_tokens` 保持 **5** |

## R11. 线程池里的层次化 barrier — 隔离环境有效,引擎内 0% 收益

| 项 | 内容 |
|---|---|
| 实测 | 隔离池测试 **1.79×**,放进引擎后 **0%**;空 body 的"每阶段固定 176 µs"**不可加**(阶段重叠) |
| 结论 | 不要再用"空 body 阶段计时"推断固定开销,也不要重复做层次化 barrier |

## R12. TP=2 启动挂死:以下假设**已排除**,不要重测

| 假设 | 实验 | 结论 |
|---|---|---|
| GEMM 内核(第 13 轮改动) | `XIAOTU_MOE_GEMM_NR=0` | 仍挂 ⇒ 排除 |
| 无锁任务发布 | 回退 `numa_pool.hpp` 到改动前 | 仍挂 ⇒ 排除 |
| 自旋窗太小 | `XIAOTU_MOE_SPIN_IDLE_US=5000` | 仍挂 ⇒ 排除 |
| CUDA graph 捕获 | `EAGER=1` | 仍挂 ⇒ 排除 |
| 池死锁 | `WATCHDOG/STALL/SLOW parallel_for` 计数全 0 | ⇒ 不是池 |
| **真因** | 被 kill 的 TP≥2 进程在 `/dev/shm` 留下 46 个 `xiaotu_ep_L*_4096_1024_2.bin`,新进程 attach 到世代错乱的旧文件 ⇒ 两个 rank 永久互等 | `rm -f /dev/shm/xiaotu_ep_*.bin` 后 250s 内就绪(已固化进 `tune_serve.sh`) |

## R13. 单卡 256K 上下文 + GPU 常驻专家层 — 放不下

| 项 | 内容 |
|---|---|
| 实测 | 单卡 256K 时 KV 需求已把显存吃满(KV 检查要求 131072 给 6.71 GiB),再加常驻层无空间 |
| 结论 | 常驻专家层必须走 **TP=2**(或降到更短上下文);不要重复"单卡 + 常驻层"的尝试 |

---

## M. 测量陷阱(**犯过的错**,不要再犯)

| 编号 | 陷阱 | 正确做法 |
|---|---|---|
| M1 | 把 SSE chunk 数当 token 数(投机解码下每步只发一个 chunk) | 从 `usage.completion_tokens` 取;`bench_nat_client.py` 已这么做 |
| M2 | 带宽探针把页面都放在一个 NUMA node(测出 95 GB/s 当成整机带宽) | 真机聚合 783 GB/s(单线程 36,单 node 98.7);探针要跨 node |
| M3 | 服务端 vs 微基准直接比(实际是 `na` 不同),得出"2.1×" | 对齐 `na`(去重后活跃专家数);真实差距是 1.26× |
| M4 | 用"空 body 阶段计时"推算固定开销并相加 | 阶段重叠,不可加(见 R11) |
| M5 | 先后两次测量直接比较(共享主机负载 30-130 漂移) | 只在**同一时间窗口内交错 A/B**,或 min-of-N |
| M6 | `MOE-PROF` 第一段 40 次调用含冷启动 warmup,拿它当稳态 | 读第二段窗口(如 `calls=80`),或看计时循环里的 ms/层 |
| M7 | `perf` 用于性能剖析 | 本机 `perf_event_paranoid=4`,用引擎自带 `XIAOTU_MOE_PROFILE` / `XIAOTU_CD_TIMING` |
| M8 | `pkill -f "关键字"`(模式匹配到自己的命令行,自杀 3 次) | 用不自匹配的模式:`pkill -f "bw[3] "`、按 PID 杀 |
| M9 | kill TP worker 后立刻看显存(残留驱动条目) | 等 ~10s,或显式 kill `Worker_TP*` PID |

---

## 修订记录

| 日期 | 改动 | 依据 |
|---|---|---|
| 2026-09-11 | 建立本文件,回填 R1-R13 + M1-M9 | 用户要求(会话中反复回退同一批做法) |
