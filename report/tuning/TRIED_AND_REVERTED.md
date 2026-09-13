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

## R14. GPU 常驻专家层 + CUDA graph 捕获(EAGER=0)— 当前不兼容

| 项 | 内容 |
|---|---|
| 试了什么 | TP=2 + `XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-5` + `EAGER=0`(CUDA graph) |
| 实测 | 捕获阶段直接失败:`torch.AcceleratorError: CUDA error: dependency created on uncaptured work in another stream` → `operation failed due to a previous error during capture`,`Worker failed`,启动失败 |
| 机制 | 常驻层的 GPU MoE 在 tp>1 时要归约部分和:代码在**捕获区内**调用 `tensor_model_parallel_all_reduce(gpu_out)`(`hybrid_model.py` 常驻分支),而 vLLM 的非 custom all-reduce 用侧流 + event ⇒ 捕获期间非法。非常驻的 GPU 预填充分支有 `not capturing` 门控,所以从来没踩到 |
| 回退动作 | 该轮改用 `EAGER=1`(与之前 tp2_eager 基线单变量可比)先拿常驻层的数据 |
| 再试条件 | **不要**在下面两处捕获不安全点修完之前重复 EAGER=0 + 常驻层 |
| 追查 1(已修) | 失败点其实是 `gpu_prefill.py` 里 `torch.cuda.current_stream().wait_event(slot.ready)`:`cudaErrorStreamCaptureIsolation`(常驻槽位的 ready 事件在捕获外 record)。已改为捕获期间跳过该等待(数据在构建时就写完) |
| 追查 2(**未修,阻塞**) | 再启动一次后失败点上移到 `_build_segmentation()` 的 `ids = ids[ok]`(布尔掩码索引 = 数据相关形状)⇒ `cudaErrorStreamCaptureUnsupported`。**`gpu_moe_layer` 过去只用于"非捕获"的 GPU 预填充分支(有 `not capturing` 门控),所以从未做过捕获安全审计** |
| 修法(下轮) | 去掉数据相关形状:无效 id 映射到"垃圾桶"专家 `E`(`ids_c = where(bad, E, ids)`),`bincount(minlength=E+1)` 排序分段,内核 grid=E 不读垃圾桶段;`A` 用固定 `T*K` 上界。需同时验证非捕获路径数值不变(用 `torch_reference_layer` 对齐) |
| 附带收益 | 修完还能让**非常驻的 GPU 预填充**也能进图(现在每步都在 EAGER 下跑) |
| **结果(已解决)** | 两处都修完后:①`/tmp/test_capture_gpu_moe.py` 证明带常驻 slot 的 `gpu_moe_layer` 可捕获并重放(与 eager 逐位一致);②真实系统 TP=2+6 常驻层+`EAGER=0` **启动成功,0 捕获错误**(tag tp2resG) |
| 教训 | 「修完捕获问题解码就会变快」是错的:TP=2 下 graph 对解码**没用**(TPOT 47.55 vs 47.32 ms),因为瓶颈是 EP 合并与线程减半,不是启动开销 — 见下一条 |

## R15. TP=2(+EP)用于 **CPU 专家解码** —— 净亏,不要再试

| 项 | 内容 |
|---|---|
| 试了什么 | TP=2 + `--enable-expert-parallel`(每 rank 128 专家 / 60 线程),指望"两个 rank 分摊专家权重读取" |
| 实测 | 每层 compute **1.31 ms(TP=1,120 线程)→ 2.45-2.53 ms(TP=2,60 线程/rank)**;解码 C=1 TPOT **28.5 ms(TP=1,35 t/s)→ 47.3-47.6 ms(TP=2+6 常驻层,21 t/s)** |
| 加 CUDA graph 有用吗 | **没用**:TP=2+常驻层 TPOT 47.55 ms(graph)vs 47.32 ms(EAGER)。成本在 EP 合并 + 每 rank 线程减半,不在启动开销 |
| 机制 | EP 让每 rank 少读一半专家 ⇒ 线程数也减半 ⇒ 没有净收益;每层还多一次**跨 rank 部分和合并**(/dev/shm 双 barrier + 等待对端),按 §49.2 估计约 +1.1 ms/层 |
| 结论 | **解码一律 TP=1**。TP=2 只用于两件事:①两条 PCIe 链路做预填充;②提供第二张卡的显存放 GPU 常驻专家层 |
| 再试条件 | 除非能把合并成本压到 <0.1 ms/层(例如"每 rank 持全专家、不做归约"),或每 rank 线程数能翻倍而不损带宽;否则不要再用 TP=2 跑 CPU 专家解码 |

## R16. 让 **draft(speculator)模型**也常驻 GPU 专家层 —— 显存翻倍,已改为默认不常驻

| 项 | 内容 |
|---|---|
| 试了什么 | `XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-5` 对所有 MoE 模块生效 ⇒ 目标模型 **和** DSpark 起草模型各常驻一份 |
| 实测 | `[xiaotu] GPU-resident model.layers.N.ffn: 1.59 GiB` 每层出现 **4 次**(2 rank × 2 模型);6 层就占 **32.25/40 GiB**(约一半是 draft);常驻层数因此上不去 ⇒ 预填充卡在 1117 t/s |
| 机制(**修正后的真因**) | 不是 draft!同 PID、同 prefix 的常驻加载出现**两条**,是插件的 `hybrid_model` 在同一进程里被加载成**两个模块对象**(模块级全局互不相通),加上 `finalize_mega_moe_weights()` 对常驻层可被调用两次(常驻层不设 `self.engine`,原守卫失效)⇒ 同一层建了两份常驻显存。13 层 ×2×1.59 GiB ≈ 41 GiB,把 KV 的 8 GiB 挤掉 ⇒ `memory allocation failed ... 8589934592 bytes`。另:实测 DSpark 起草模型**没有** MoE 层(否则每步会有两遍 43 层 CPU 调用,步时不会是 97 ms) |
| 回退动作 | ①进程级共享状态挂到 `builtins`(`_resident_state()`)——模块被复制也共享,用它记录 prefix 次数与已用字节;②`finalize_mega_moe_weights()` 用 `_resident_slot is not None` 做幂等;③新增**硬旋钮** `XIAOTU_MOE_RESIDENT_BUDGET_GB`(0=不限):超出预算的层自动当普通 CPU 层(并把 CPU 引擎建起来,避免 engine=None 静默返回输入);④`_LAYERS` 只登记第一份,避免 prefetch-ahead 作用到副本 |
| 再试条件 | 只有当显存充裕到"翻倍也无所谓"时才考虑给 draft 常驻;否则不要 |

## R17. 用 **CPU 专家路径做预填充** —— 大批量下是算力瓶颈(261 t/s),不要再试

| 项 | 内容 |
|---|---|
| 曾经的设想 | 预填充时 4096 token 会命中每层**全部 256 个专家**,一遍读 139 GB;DRAM 783 GB/s ⇒ 理论 0.18 s/遍,似乎比 PCIe 流式(25 GB/s ⇒ 5.6 s)快 20× |
| 实测(`bench_cpu_engine.py`,DEDUP=256 全专家,THREADS=120) | B=1024:**113.3 ms/层** ⇒ 43 层 4.87 s ⇒ **210 t/s**;B=4096:**365.0 ms/层** ⇒ 15.7 s ⇒ **261 t/s** |
| 机制 | 每层 3.22 GB / 365 ms = 8.8 GB/s(不是带宽极限);而 4096×6 专家的 MACs(约 3×25.2M×2×4096 = 1.24 TFLOP)÷365 ms = **3.4 TFLOP/s ≈ AVX-512 BF16 峰值的 50%** ⇒ **compute-bound** |
| 与 GPU 路径对比 | GPU 逐层流式 + 预取重叠:3.7 s(1100 t/s),仍是 CPU 路径的 4× |
| 结论 | 预填充一律走 GPU 流式;CPU 路径只服务解码(小批量、带宽受限)。R7 的"19 t/s"是 setup 主导,但即使 setup 归零,大批量也只有 ~261 t/s |
| 再试条件 | 除非 CPU 端算力翻 4 倍(不可能)或只把**部分**层放 CPU 且批量小到算力不是瓶颈 |

## R18. "两个 rank 并发抢 PCIe/主机带宽"导致预填充慢 —— 实测排除

| 项 | 内容 |
|---|---|
| 假设 | TP=2 的两个 rank 同时流 1.59 GB/层,可能互相抢 PCIe 或主机内存带宽(服务里 ~100 ms/层 vs 微基准 64 ms/层)|
| 实测 | 两个进程并发跑 `bench_gpu_moe_prefetch.py`(GPU0/GPU1,各 1.594 GiB/层):两边都还是 **63.9 / 64.0 ms/层 = 1491 tok/s**,与单进程完全一致 |
| 结论 | 并发**不互相干扰**,PCIe 已到 gen4 x16 峰值。服务里多出的 ~36 ms/层是**模型自身每层工作(attention/indexer + 框架)没有和 H2D 重叠**,不是带宽问题 |
| 再试条件 | 不要再把这个差距归因于 PCIe/带宽;要打的是"每层 H2D 与 attention 的重叠" |

## R19. 把 `inter` 缓冲按 T 行分配(想省预填充每层的 201 MB 分配)—— 越界,已回退

| 项 | 内容 |
|---|---|
| 动机 | T=4096 预填充时 `A = T*K = 24576` ⇒ `inter` 每层分配/memset 24576×4096×2B = **201 MB**(主机侧 ~30 ms/层,不在 H2D 重叠之内) |
| 试了什么 | 以为内核按 token id 索引 inter,于是 `inter = torch.empty((T, 2*I))` |
| 实测 | 服务启动即 **`CUDA error: an illegal memory access was encountered`**(金测试在小尺寸下"通过"是因为越界落在了分配器的松弛块里,掩盖了问题) |
| 真因 | gate_up 内核用**排序位置** `g_rows` 索引 inter(`inter_ptr + g_rows*inter_ld`),只有写 `out` 时才换成 token id ⇒ inter 必须有 **A 行** |
| 回退动作 | 恢复 `torch.empty((A, 2*I))`,并在代码原处写清"不许再动行数语义";金测试复验通过 |
| 再试条件 | 想省这块分配只能走"让 A = 有效项数(A 仍是排序行数)"或**持久缓冲复用**(不是改行数语义);改行数语义一律禁止 |

## R20. 加深预取环(`XIAOTU_MOE_PREFETCH_SLOTS=3`)改善预填充 —— 无收益

| 项 | 内容 |
|---|---|
| 动机 | TP=2+6 常驻层实测每层 ~100 ms,而纯流式只需 64 ms ⇒ 怀疑 H2D 发起太晚,想用更深的环给它更多提前量(lk 用 `LVLLM_GPU_PREFETCH_WINDOW=1`)|
| 实测 | 微基准:2 槽 64.1 ms vs 3 槽 64.0 ms(已到 PCIe 上限,无差别);服务:TTFT **3752 / 3763 ms(3 槽)** vs 3666 / 3707 ms(2 槽)⇒ 无收益(略差,在噪声内)|
| 附带 | VRAM 从 32.25 → 34.31 GiB/卡 |
| 结论 | 那 ~31 ms/层**不是**预取提前量不足;环深默认保持 **2**。旋钮保留(`XIAOTU_MOE_PREFETCH_SLOTS`,默认 2 = 原行为)仅供后续调试 |
| 再试条件 | 不要再靠加深环来提预填充;要先用 nsys 把那 31 ms/层量出来 |

## R21. 把 TP=2 预填充多出的 ~31 ms/层归因于**主机侧**开销 —— 实测否定

| 项 | 内容 |
|---|---|
| 动机 | TP=2+6 常驻层每层 ~100 ms,而纯流式只需 64 ms;怀疑 `_build_segmentation`(argsort 24576)+ `zeros((T,H))` + `inter` 分配(201 MB)串在关键路径上 |
| 实测 | 新增 `XIAOTU_GP_TIMING=1` 主机侧分段计时(预热后 T=3074/1335 的窗口):**seg=0.35-0.47 ms,rest=0.11-0.12 ms,host_total≈0.5 ms/层**(两个 rank 一致)|
| 结论 | 主机侧开销**可忽略**(0.5 ms vs 31 ms)⇒ 差额在 **GPU 侧**:候选 ①每层对 (4096,4096) bf16=33 MB 做一次 TP all-reduce(A100-PCIE 无 NVLink,走 PCIe);②H2D 在服务里只跑 ~16 GB/s(微基准 26.8,单卡是 25) |
| 再试条件 | 不要再往"主机侧 Python/分配"方向找;要量 GPU 侧必须用 nsys(nvtx)或把 AR 关掉对照(TP=1 预填充 703 t/s 无 AR,是本机已验证的对照) |

## R22. 把 TP=2 预填充差额归因于**每层 TP 归约**(33MB bf16 过 PCIe)—— 实测只有 1.5-2 ms/层

| 项 | 内容 |
|---|---|
| 动机 | 第 21 轮实测主机侧只 0.5 ms/层,于是怀疑每层一次 `tensor_model_parallel_all_reduce`((4096,4096) bf16=33MB,A100-PCIE 无 NVLink)|
| 试了什么 | 新增计时专用开关 `XIAOTU_SKIP_AR=1`(默认 0,跳过该归约;数值会不完整,只用于计时)|
| 实测 | TTFT **3618 / 3669 ms(跳过归约)** vs 3666 / 3707 / 3742 / 3752 ms(带归约)⇒ 37 层总共只省 ~50-80 ms = **~1.5-2 ms/层** |
| 结论 | 归约不是那 ~31 ms/层的原因(但仍解释 R15 里 TP=2 解码每层 +1.1 ms 的一部分)|
| 再试条件 | 不要再把预填充差额归因于 TP 归约。剩下唯一未解释的就是"H2D 在服务里 ~100 ms/层 vs 微基准 64 ms/层"这 ~1.5× 的传输效率差,要用 nsys 量 |

## R23. 把预取环降到 **1 槽**省 3 GB 显存 —— 是正确性 bug,禁止

| 项 | 内容 |
|---|---|
| 动机(用户 2026-09-11 提议) | GPU 预填充计算远快于权重传输,2 槽可能没必要,省 ~3 GB 显存 |
| 实测(`/tmp/test_1slot.py`,模拟插件顺序:先发 L+1 预取、再算 L) | 2 槽:`slotA is slotB=False`,输出与 L 的参考一致(err_vs_L=0.17 / err_vs_L+1=22.97)⇒ 正确;1 槽:`slotA is slotB=True`,输出与 **L+1** 的参考一致(err_vs_L=17.54 / err_vs_L+1=0.096)⇒ **用下一层的权重算了本层** |
| 机制 | ping-pong 语义要求"一个槽在算、另一个在填"。只有 1 槽时 L+1 的 H2D 必须复用 L 正在读的缓冲 ⇒ 前向读到下一层的权重 |
| 回退动作 | `XIAOTU_MOE_PREFETCH_SLOTS` 下限固定为 **2**(代码里注明原因)|
| 想省显存怎么办 | 用 `XIAOTU_MOE_RESIDENT_BUDGET_GB` 少放 1 层常驻(每层 1.59 GiB @TP=2 / 3.19 GiB @TP=1),不要动预取环 |

## R24. 把预填充差额归因于**服务里 H2D 效率低(17 GB/s)** —— 实测 H2D 其实是 67-74 ms(≈峰值)

| 项 | 内容 |
|---|---|
| 动机 | 每层 100 ms vs 微基准 64 ms;猜服务里 H2D 只跑到 ~17 GB/s |
| 实测 | 新增 CUDA 事件计时(`XIAOTU_GP_TIMING=1` 下 `[gp-h2d]`):**每层 67.4-73.7 ms**(TP0/TP1 一致,≈1.71 GB ⇒ ~24 GB/s,接近峰值);同窗口主机侧 0.44-0.59 ms |
| 结论 | H2D **不是**差额来源。每层 ~102 ms = H2D 70 + **~30 ms 未重叠的 GPU 侧工作**(attention/indexer、MoE 内核、常驻层/DSpark 相关)|
| 再试条件 | 不要再怀疑 H2D 传输效率;要打的是那 30 ms 的 GPU 侧工作(下一步必须用 nsys/nvtx 时间线看它是什么)|

## R25. 用"传-算-传"(串行,不要 ping-pong)省 3.4 GB 显存 —— 代价 1.7× 预填充,不划算

| 项 | 内容 |
|---|---|
| 提问(用户 2026-09-11) | 不要 ping-pong,改成"传完→算→传下一层→算……",是否可行/值得 |
| 这个模式已存在 | 就是 `XIAOTU_GPU_PREFETCH_AHEAD=0`(`slot=None`,H2D 内联在计算流上)|
| 实测(TP=2 + 6 常驻层,`bench_nat_client` L=4096) | 串行 **6.32 / 6.33 / 6.40 s**(645 t/s)vs 重叠 **3.66 / 3.70 s**(1110 t/s);每层 ≈ 70(H2D)+100(算)≈172 ms |
| 显存 | 串行不用环,省 2×1.7 GB ≈ **3.4 GB**(TP=2)|
| 每 GB 效率 | 槽:**3.4 GB 换回 2.65 s ≈ 780 ms/GB**;常驻层:1.59 GiB 换回 ~70 ms ≈ **44 ms/GB** ⇒ 槽的性价比高 ~18× |
| 结论 | **默认保持重叠**(PFA=1、2 槽);显存紧张时优先砍常驻层(`XIAOTU_MOE_RESIDENT_BUDGET_GB`),不要把重叠关掉 |
| 再试条件 | 只有在"显存连 KV 都不够、必须牺牲预填充"时才用 `XIAOTU_GPU_PREFETCH_AHEAD=0` |

## R26. `VLLM_USE_V2_MODEL_RUNNER=1` 改善预填充/解码 —— 本机无收益

| 项 | 内容 |
|---|---|
| 动机 | 目标清单里点名的未试旋钮;怀疑每层那 ~30 ms GPU 侧开销里有 vLLM runner 的框架开销 |
| 实测(TP=2+6 常驻层) | 预填充 TTFT **3733 / 3767 ms** vs 基线 3666 / 3707 / 3742 / 3771 ms ⇒ 无差别;进程 environ 确认该变量已生效 |
| 结论 | 对本模型/本版本无效(可能该 runner 未覆盖这条混合路径)。**保持默认(不设)** |
| 再试条件 | 升级 vLLM 主线版本后再评估;在当前 pin 的 `6c73b08` 上不要再试 |

## R27. 解释 TP=2 的一切反常:每层跨卡归约流量(PCIe 无 NVLink),不是我们的引擎

| 现象(全部实测) | 数值 |
|---|---|
| 单卡每层"非 H2D"开销(H2D 128 ms 之外) | **~12 ms** |
| TP=2 每层"非 H2D"开销(H2D 70 ms 之外) | **~30 ms** |
| 差额 | **~18 ms/层**,且与 T 无关、不在主机侧(R21)、不是我们的 MoE 归约(仅 2 ms,R22)|
| TP=2 预填充收益 | 703 → 1117 t/s(+47%,本应 +100%,两条 PCIe 链路)|
| TP=2 解码 | 每层 +1.1 ms 合并(R15)|

**机制**:TP=2 时 vLLM 的**行并行 attention**(o_proj)每层都要做一次跨卡 all-reduce,
`(T,4096)` bf16 = **33 MB/层**;本机 A100 是 **PCIe 卡(无 NVLink)**,`--disable-custom-all-reduce`
后的兜底 AR 有效带宽只有几 GB/s ⇒ 33-66 MB 双向 ≈ **10-20 ms/层**。这与上面 18 ms 的差额吻合。

⇒ 这不是引擎/编排问题(同机 A/B 我们比 lk 快 1.46×),而是**本机拓扑**(PCIe-only)的代价:
TP=2 省下的 PCIe 权重流式时间,被每层 attention 的跨卡归约吃掉一部分;TP=1 没有这笔开销,
所以**解码一律 TP=1**(R15)、预填充才考虑 TP=2。

**下一步实验(明确、便宜)**:TP=2 + 6 常驻 + EAGER=1,**去掉** `--disable-custom-all-reduce`
(即用 vLLM 的 custom AR,它在 EAGER 下没有 R14 的捕获问题)。若每层那 ~18 ms 缩小
⇒ 归约假设成立,预填充 TTFT 应从 3.70 s 掉向 ~3.0 s(1360 t/s+)。

## R28. 换 vLLM **custom all-reduce** 消除 TP=2 每层那 ~18 ms —— 实测无变化

| 项 | 内容 |
|---|---|
| 动机 | R27 推断 TP=2 每层多出的 ~18 ms 是兜底 AR(33 MB/层过 PCIe);若成立,换 custom AR 应显著变快 |
| 实测 | TP=2 + 6 常驻层 + EAGER=1,去掉 `--disable-custom-all-reduce`:预填充 TTFT **3750 / 3768 ms** vs 兜底 AR 基线 3666-3771 ms ⇒ **无差别**;H2D 仍 67.4 ms/层(与基线一致)|
| 结论 | AR 的实现方式**不是**那 ~30 ms/层的来源(R27 的归约假设未证实)。注意:custom AR 在本配置下能正常启动(不像 R14 的常驻层+图组合)|
| 附带发现 | `VLLM_USE_V2_MODEL_RUNNER=0` 会直接启动失败(`Model Runner V1 does not support: dspark speculative decoding`)⇒ 本模型 + DSpark **必须**走 V2 runner(默认即如此,不要显式设 0)|
| 再试条件 | 不要再在 AR 实现/归约上找这 30 ms;必须用 nsys/nvtx 时间线看 GPU 侧(kernel 级)|

## R29. 用插件自带 `XIAOTU_TORCH_PROFILE` 拿 GPU kernel 时间线 —— 当前拿不到(埋点两个缺陷)

| 项 | 内容 |
|---|---|
| 期望 | `_maybe_profile()` 已带 `activities=[CPU, CUDA]`,以为 `prof.key_averages().table()` 能列出 kernel |
| 实测 | 日志里 `[xiaotu-profile]` 表**只有** `cudaDeviceSynchronize` / `Activity Buffer Request`,**0 条 kernel 行**;两个 rank 还争抢同一路径(`Failed to rename /tmp/pref_prof.json.tmp to /tmp/pref_prof.json`)|
| 结论 | 该埋点当前无法回答"那 ~30 ms/层是哪些 kernel":①窗口从**第一次 MoE forward** 开始(可能落在纯 CPU 的 dummy 阶段)②trace 路径没有 rank 后缀,两个 rank 互相覆盖 |
| 修法(下轮) | ①路径加 rank/pid 后缀;②`XIAOTU_TORCH_PROFILE_CALLS` 从"第一次调用"改为"第一次**真实预填充**"(qlen ≥ 阈值)后再开窗;③表按 `cuda_time_total` 排序并只打印 CUDA 行 |
| 再试条件 | 修完这三点再用它定位 GPU 侧;在那之前不要再解析它的 JSON(§58 已记过一次)|

## R30. 用 profiler 找那 30 ms 时只看了 trace 的 JSON / 只看表的前几列 —— 两次都白费

| 项 | 内容 |
|---|---|
| 错在哪 | ①第 18 轮只解析 chrome trace 的 JSON(该文件在本配置下不含 kernel 事件)却没看**日志里那张表**;②第 27 轮看了表,但用 `cut -c1-120` 把 **CUDA 列**截掉了,只看到 Self CPU 列,误判"0 条 kernel 行" |
| 后果 | 多花 2 轮才拿到"~30 ms/层 = 两个 Triton MoE 内核"这个结论 |
| 正确做法 | 读 `[xiaotu-profile]` 表要**完整行**(CUDA total / CUDA time avg 列在右侧);或直接对日志 `grep -A12 xiaotu-profile` 后按空格规范化再看,不要盲切列 |
| 附带修复 | 埋点已加 rank 后缀(两 rank 不再抢同一 trace)并按 `cuda_time_total` 排序 |

## R31. 在**被污染的窗口**里扫 kernel block 参数 —— 数据无效,不要据此调参

| 项 | 内容 |
|---|---|
| 做了什么 | `bench_gpu_moe_prefetch.py` T=4096 扫 `WARPS=8` / `BM=BN=128,NS=3` / `BK=128,WARPS=8,NS=3` |
| 实测 | 三种配置的 `ovl` 列**全都是 127.7 ms**,基线本是 **63.9 ms**;`seq` 也从 129.5 变成 218-245 —— 连"H2D 主导、与 kernel 参数无关"的那一列都整倍变慢 ⇒ 窗口被外部负载/PCIe 争用污染(共享主机)|
| 结论 | 这轮扫描**不可用**;不能在负载漂移的窗口里比较 kernel 参数(M5 的同窗口交错原则) |
| 正确做法 | ①先把 8070 停掉,②用同一进程内**交替**跑两套参数(或 min-of-N),③看 `seq` 差(因为 `ovl` 被 H2D 顶住,只有 `seq` 反映内核时间)|
| 状态 | 基线仍以安静窗口的 **seq 129.5-130.2 ms / ovl 63.9-64.0 ms**(T=4096)为准 |

## R32. 靠调 Triton MoE 的 block 参数改善预填充内核 —— 安静窗口实测:现有默认已最优

| 项 | 内容 |
|---|---|
| 背景 | §69 证明预填充每层 GPU 侧 ~54 ms 全在 `_down_kernel`(29.4)+ `_gate_up_kernel`(24.3),于是扫 block 参数 |
| 方法(修正 R31 的错)| 停掉 8070 造安静窗口(load 0.85),同窗口连续对照,**只看 `seq` 列**(= H2D + 内核,H2D 常数 ⇒ 差值即内核时间)|
| 实测(E=128 每 rank 分片,T=4096) | 默认 **seq 130.4 ms**;`BM=128` → 134.0;`BM=128,WARPS=8` → 135.9;`BK=128,STAGES=4` → 148.2(且 ovl 也退化到 68.8)⇒ **全部不优于默认** |
| 结论 | 现有默认(`BM=BN=BK=BH=64, NS=2, WARPS=4`)已是这组参数里最好的;预填充内核对这些旋钮**不敏感**。要再快必须换内核结构(预填充向的 grouped-GEMM:更大 N 向 tiling / persistent CTA / 每专家行数少时的合并专家调度),不是调参 |
| 再试条件 | 不要再扫这几个参数;直接上结构改动,并用同一方法(安静窗口 + 只看 `seq`)验收 |

## R33. "我们的引擎比 lk_moe 快"(出自服务器级 A/B)—— 微基准对照推翻

| 项 | 内容 |
|---|---|
| 旧结论 | §44.1:同模式服务器级 A/B,我们 9.53 vs lk 6.52 tok/s ⇒ "快 1.46×" |
| 新实测(同一同步 API/同一权重/同线程数)| xiaotu 1.22/1.32 ms 每层(DEDUP=12/23)vs **lk 0.57/0.67** ⇒ **lk 快 ~2.0×**(带宽 265/376 vs 124/184 GB/s)|
| 为何不同 | 服务器级对照里两边 fork 的调用与同步方式不同(lk 的 fork 走 CUDA host-callback 异步路径),不能代表内核;`bench_vs_lkmoe.py` 的 decode 计时也是异步发射(0.01 ms/层,非物理值)|
| 教训 | 比引擎必须用**同一个同步 API 微基准**;两边各自的环境变量不能混(`LK_THREADS` vs `XIAOTU_MOE_THREADS`,漏设会让 lk 假慢 ~70×)|
| 行动 | 把我们的每线程带宽从 1.0-1.5 往 2.2-3.1 GB/s 提(先看 lk 的 job 划分/内层循环,§2 的反汇编已给出 8 行×64 列的寄存器分块)|

## R34. 把"我们比 lk 慢"归因于 NUMA/线程数/barrier 数 —— 实测都不是,是**标量 FMA**

| 项 | 内容 |
|---|---|
| 排查过的方向 | 线程数(120/192)、NUMA 分片、barrier 数(5 vs 3)、预取、常驻层、归约、主机侧 —— 各自单独测过,都不是那 2× 的来源 |
| 真因(指令级)| `objdump` 直方图:我们库内 **`vfmadd*ss` 标量 FMA 491 条** vs `vfmadd*ps` 仅 243 条,`vdpbf16ps` 仅 18 条;lk 热内核 **158 条全 packed `vfmadd231ps`、0 条标量**,且权重解码提到独立阶段、每 K-block 解一次复用 8 行 |
| 触发条件 | 解码每专家 me≈3 ⇒ 走 `block_23`/小 me 路径 ⇒ 该路径是标量实现 |
| 行动 | 重写小 me 路径为 packed zmm fp32 FMA(+ 权重解码 hoist),验收:同一微基准 BS=6/DEDUP=12 ≤0.7 ms/层、每线程 ≥2.2 GB/s |
| 更正 | 见 NOTES §76:标量 FMA 实际来自阶段 C 归约 lambda 与 flat_thunk,**主 GEMV 是 packed FAST_FP4**;真正根因是 fp4 解码按激活行重复(me 次/权重字节),lk 每 K-block 解一次复用 8 行。结论方向不变,证据更正 |
| 第33轮更正 | §77 的"激活侧"结论**作废**:激活仅占权重流量 0.39%(§78 算术)。累计已排除:解码未 hoist / 激活流量 / 标量 FMA / NUMA / 线程数 / pin / 环深 / 并发 / TP 归约 / H2D。下一步先用 `XIAOTU_MOE_ABL_DECODE` 消融实验区分"解码依赖链延迟" vs "访存/MLP",再改内核 |
| 第32轮更正 | ①解码 hoist **已在 `block_23` 实现**;②两引擎输出一致(±0.5%)⇒ 2× 真实。剩余差异在**激活侧**(我们先把激活整段转 fp32 缓冲再 mul+2fma;lk 用 bf16 直取+寄存器 cvt/bcast,每 k 2 条 FMA)。见 NOTES §77 |
| 再试条件 | 在这条做完之前,不要再去调线程/布局/barrier/常驻层来解释或改善解码 |

## R36. 跨 K 组 2 路软件流水(照 §79 的"解码链"结论)—— 无收益,已回退

| 项 | 内容 |
|---|---|
| 依据 | §79 独立微基准:真实解码链 239 GB/s/线程 vs 去掉链 1002 GB/s/线程(4.19×)⇒ 判断链延迟是限幅器 |
| 试了什么 | `block_23` 组循环按 2 组展开、两条独立解码链(数值逐组顺序不变)|
| 实测 | **1.19 / 1.32 ms/层** vs 基线 1.18 / 1.32 ⇒ **无收益**(OoO 本来就在重叠;源码展开没有增量)|
| 错在哪 | §79 的权重块 128 KB **L2 常驻**,测的是指令吞吐;引擎是 **DRAM 流式**,16 B/43cycle 与链延迟数值巧合吻合但因果不同。真限幅更可能是**访存延迟/MLP** |
| 回退 | 源码已 `git checkout` 还原(当前 build 里的 .so 是展开版,数值一致、性能一致,下次构建自会还原)|
| 再试条件 | 消融必须用 **DRAM 常驻工作集(≥256 MB)** 重做;并测预取距离/多路独立行(MLP),不要再做源码级展开 |

## R37. "内核实现不如 lk_moe"(目标第一条的前提)—— 被 DRAM 消融推翻

| 项 | 内容 |
|---|---|
| 原判定 | 目标第一条把差距定位为"解码内核效率"(先猜标量 FMA,后猜解码摊销/激活侧)|
| 实测(`/tmp/abl2.cpp`,512 MB DRAM 工作集,单线程)| 我们的内层循环(顺序流式 + 真实解码)= **8.46 GB/s/线程**;引擎里只有 **1.1 GB/s/线程** ⇒ 差 **7.7×** |
| 结论 | **内核本身够快**;瓶颈在**引擎结构**(job 粒度/派发开销、分片布局的页表/落点、冷 DRAM),而不是内核指令/解码 |
| 影响 | 目标第一条的"重写内核"方向**不解决**这 7.7×;应先量"引擎内单线程跑同一 job body"的吞吐来二分(结构 vs 布局)|
| 再试条件 | 在"引擎内单线程 job"对照出来之前,不要再改内核内层循环(已有 R34/R36 两次无效尝试)|

## R38. 用"引擎内单线程(THREADS=1)"当布局对照,以及 THP 假设 —— 都已否

| 项 | 内容 |
|---|---|
| 试了什么 | ①`XIAOTU_MOE_THREADS=1` 当"同一 job body 在引擎布局下"的单线程对照;②`XIAOTU_MOE_SHARD_HUGEPAGE=1` 恢复分片大页 |
| 实测 | ①THREADS=1 ⇒ 52.88 ms/层(2.86 GB/s),但**该分支会串行执行全部 8 个分片的 job ⇒ 7/8 是远端读**,数值被 NUMA 距离污染,不能与独立内核 8.46 GB/s 相减;②THP=0 → 1.21/1.45 ms,THP=1 → 1.35/1.36 ms ⇒ **无收益** |
| 结论 | 页表/THP 不是缺口;单线程对照方法无效。正确的每线程对照只能在 **120 线程 + NS=8(本地读)** 下取,现状 1.07 GB/s/线程 |
| 再试条件 | 不要再动 THP 分片(R2 维持);不要再拿 THREADS=1 当布局对照。要判定 (a)行跨步 (b)每job开销 (c)争用,用 SHARDSPLIT 扫描 + 同 na 不同 me 的每字节耗时 |

## R39. 把 7.9× 归因于"每 job / 每阶段开销"(job 粒度太细)—— 被 SHARDSPLIT 扫描否掉

| 项 | 内容 |
|---|---|
| 依据 | §81 算术:96 job/节点 × 64 KB/job,按内核 8.5 GB/s 每 job 只需 7.5 µs,而实测折合 ~117 µs/job |
| 实测 | `XIAOTU_MOE_SHARDSPLIT` = 0/1/2/4/8/16 → **1.32/1.32/1.24/1.20/1.23/1.31 ms/层** ⇒ 平坦(±10%),默认≈5 已近最优 |
| 结论 | 派发/job 粒度最多值 10%,**不是主因**;那 117 µs/job 的算术里混入了内存延迟的真实代价(冷 DRAM + 行跨步) |
| 再试条件 | 不要再调 SHARDSPLIT/线程数来解释 7.9×;下一步做引擎内消融:`XIAOTU_MOE_ABL_SCALE`(跳过 scale 取数)与 `XIAOTU_MOE_ABL_DECODE`(跳过解码),都不改访存字节数 |

## R40. 把 5.8× 归因于"分片布局的行跨步访存" —— 被消融否掉

| 项 | 内容 |
|---|---|
| 依据 | 引擎每 node 只跑到 17 GB/s(理论 191 µs/层,实测 1100 µs),怀疑 512KB+4.2MB 跨步不利于预取 |
| 实测(`/tmp/abl3.cpp`,1 GB 工作集,单线程)| 顺序 8.54 vs 分片形态(512KB+4.2MB)8.52 vs 512KB+8.4MB 8.56 vs 128KB+4.2MB 8.57 GB/s ⇒ **完全无差别** |
| 结论 | 预取器吃得下 128-512KB 连续跨度;布局/跨步不是病根。剩下唯一未被否的是**节点内并行带宽标度**(15 线程/node 只有 17 GB/s,而单 node 上限 98.7 GB/s)|
| 再试条件 | 不要再改访问布局/加预取来救这 5.8×;先做"单 node N 线程读自己分片"的标度曲线,确认是线程↔分片映射/亲和性问题还是引擎外的因素 |

## R41. "线程↔node 映射错误 / 分片落点不对"(导致每 node 只有 16 GB/s)—— 已否

| 项 | 内容 |
|---|---|
| 依据 | 引擎每 node 仅 ~16 GB/s(vs node 能力 98.7),怀疑线程读成了远端 |
| 实测 | ①分片页落点检查(一层引擎):node0-6 各 0.38-0.46 GB(=1/8 ✓),node7 的 4.09 GB 是基准里 numpy 源数组而非分片 ⇒ **页分布正确**;②NSHARD 8/4/2 → 1.20/1.71/2.60 ms ⇒ **细分最快**,证明本地读在生效;③SPIN_IDLE 20/50/100/300/1000 → 1.45/1.50/1.38/**1.27**/1.31 ⇒ 自旋争用不是原因(默认已最优)|
| 结论 | 映射/落点/自旋都正常;引擎是在 **~130 GB/s 硬饱和**(96 线程起不再涨)|
| 再试条件 | 不要再查映射/落点/自旋;下一步做 mini-engine(复刻引擎内层 + 绑 node 分片 + 1/5/15 线程)来二分"编排层 vs 内层每字节成本" |

## R42. 把 A2(setup, 0.28 ms/层)归因于 resize 的零填充 —— 无收益,已回退

| 项 | 内容 |
|---|---|
| 依据 | A2=23%(稳态实测,非冷启动);每专家 5 次 `resize()` 理论零填充 ≈1.4 MB/层 |
| 试了什么 | 把 setup 的 `resize()` 改成 `reserve()`(缓冲只用 `.data()`,size 无意义)|
| 实测 | DEDUP=12 **1.29**(基线 1.20)、DEDUP=23 **1.35**(基线 1.32);**A2 仍 0.28 ms** ⇒ 无收益 |
| 结论 | 零填充不是 A2 主体;A2 的 0.28 ms 仍未解释 |
| 再试条件 | 先在 A2 内部分段打点(fill / bookkeeping / gather)定位,再改;不要再整体替换 resize/reserve |

## R43. 直接按 §89 的"取消 gather(行号索引)"改内核 —— 先被 NOGATHER 诊断挡下

| 项 | 内容 |
|---|---|
| 依据 | §89:gather=267 µs(A2 的 97%、整层 22%),看起来是可摘的串行成本 |
| 实测(计时诊断 `XIAOTU_MOE_NOGATHER=1`,跳过 memcpy、数值无效)| DEDUP=12 **1.27**(基线 1.20)、DEDUP=23 **1.48**(基线 1.32)⇒ **总时间没降** |
| 结论 | 那 267 µs 与池阶段的时间账**可能有重叠**,不是可加串行成本;或摘掉后 A 相读到更冷的 xg 把收益吃回。**在证实"严格串行"之前,不要动内核加行号索引**(否则很可能又是一轮白改)|
| 再试条件 | 先在 A 相入口与 gather 结束处比对时刻,确认是否重叠;确认串行后再改内核 |

## R44. "A2/gather 是 22% 的可摘浪费"(§89/§90)—— 被 NOGATHER+PROFILE 对照否掉

| 项 | 内容 |
|---|---|
| 依据 | §89:gather=267 µs;§90 的 NOGATHER 单测"没降"原因不明 |
| 实测(同窗口,逐相) | baseline A=0.48 A2=0.25;NOGATHER A=**0.84** A2=**0.005** ⇒ gather 被摘掉后 **A 相涨了 +0.36**(预热效应),wall 1.16→1.28 |
| 结论 | gather 是**耦合成本**(预热 xg 目标页/缓存),**不可摘**;A2 不是浪费。靶子改回 **A/B 的访存带宽**(210/152 GB/s)|
| 再试条件 | 不要再试图删除/搬移 gather 或改 slice 内核的行号索引;A/B 的下一步是 **MLP/预取距离**(`_mm_prefetch` 下一行 → 下一 job 前几行 / 软件流水 i+2) |

## R45. 加深预取距离(i+2 行)改善 A/B 相 —— 无收益(1.17/1.32 vs 1.20/1.32)

| 项 | 内容 |
|---|---|
| 依据 | §91 判断剩下的是访存并发度/预取距离(job 间跨 4.2-8.4MB,预取器在边界 restart)|
| 试了什么 | 在 `block_23` 的 3 处 `next_row` 预取点都加上 `far_row`(j+2)提前量(`grep -c` 确认 3 处后 `replace` 全应用)|
| 实测 | DEDUP=12 **1.17**(基线 1.20)、DEDUP=23 **1.32**(基线 1.32)⇒ 无收益,已回退 |
| 结论 | 预取距离不是限幅项。A/B 相的常规杠杆(指令/布局/映射/自旋/job 粒度/分片粒度/线程数/预取距离)**已全部排除** |
| 再试条件 | 只剩"每线程在飞请求数(MSHR/LFB)"方向:内层同时展开 2 个 K 组取数、或跨行交错取数;若仍无收益,则需改数据结构(每线程一次拉 64KB 连续块)而非调参 |

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
| 2026-09-11 | 更正硬件事实:本机为**两路 9654(192 核)**(NOTES §72),此前"单颗"说法作废 | 用户指出 |
| 2026-09-11 | 追加 R14(常驻层+图,已解决)、R15(TP=2 解码净亏)、R16(常驻层重复分配) | 第 15-16 轮实测 |
| 2026-09-11 | 追加 R17(CPU 预填充大批量 compute-bound 261 t/s)、R18(两 rank 并发不抢带宽,已排除) | 第 17 轮实测 |

---

## 第 66 轮追加

| 编号 | 试过的做法 | 结论 / 为什么不要再试 |
|---|---|---|
| R46 | 认为引擎 A 相慢是"每(列,组) `row_scale` 字节取数 + 额外 FMA"造成的,并试图优化它 | **否掉**。同编译单元交替 min-of-3:纯 fp32 = 4.06 GB/s vs 加 scale 取数 = 4.10 GB/s(0.99×,不塌)。该项被流水线完全隐藏。**不要再为它改内核** |
| R47 | 用多线程微基准(`abl10`)测"120 线程聚合带宽",拿它当引擎慢 2.3× 的证据 | **否掉且方法作废**。NT=1/15/60/120 → 4.10/1.94/1.31/0.76 GB/s 每线程;按工作集非单调(29/58/65/82 GB/s),测到的是线程启动/唤醒开销。**多线程微基准不得作为定量证据** |
| R48(=前提本身) | "引擎上下文比同一内核慢 2.3×"这个靶子 | **前提不成立**。单线程 4.06 GB/s 在工作集 537 MB(DRAM)与 16 MB(L3)下**完全相同** ⇒ 单线程是指令路径受限,与 120 线程的聚合访存条件不可比。**不要再拿单线程微基准的绝对数去减引擎的多线程数** |

| 日期 | 改动 | 依据 |
|---|---|---|
| 2026-xx | 追加 R46(M scale 取数 假设否掉)、R47(多线程微基准作废)、R48(2.3× 前提推翻) | 第 66 轮 `/tmp/abl9.cpp` 同 TU 交替 min-of-3 + 双工作集对照 |

| R49 | "gather 的 237 µs 是串行 memcpy 暴露的 DRAM 延迟",把它并行化就赚回来 | **数据结论否掉**。并行化后 `[setup-prof]` 仍报 195-238 µs,且**空 body 的 pfor 就报 177 µs**(x10=1140、x50=5670,严格线性)。那 237 µs 是**并行区固定同步开销**,不是 memcpy。顺带:同类"把串行段并行化"的改动在修好锚定锁之前**都不值得做** |
| R50 | 时钟/计时器伪影导致 setup-prof 数字失真 | 已核验:`steady_clock::now()` = **25.8 ns/次**(/tmp/clk.cpp)⇒ 打点可信,差异是真实的 |
| R51 | 继续深挖解码内核(指令数/字节数/布局/K-major)以追 2× | **暂停**。空并行区 113 µs × 每层 4-5 区 ≈ 0.5 ms,扣掉后 A/B 相已是 263-294 GB/s(= lk 的 265 GB/s)。**内核已与 lk 同级**,先修同步再谈内核 |

| 日期 | 改动 | 依据 |
|---|---|---|
| 2026-xx | 追加 R49(gather 并行化 ≈ 中性但保留)、R50(时钟已核验)、R51(内核方向暂停);新增**硬规则:优化内核前先数每层有几个 `pfor` 并量空 body 成本** | 第 67 轮 NOOP_GATHER 线性响应实验 + `numa_pool.hpp:884` 锁争用 |

| R52 | 想用"降低线程数看 113 µs 是否随线程数缩小"来验证锁争用 | **未取到数据且疑似挂死**:`XIAOTU_MOE_THREADS=8/32` 的 `bench_engine_ab.py`(NOOP_GATHER=1)10 分钟内**无任何输出**,被 SIGTERM 终止。**不要再用低线程数跑该微基准**(120 线程是已知可用点);要用"锁内计数器"或直接改成无锁锚定来验证。另注:本项不影响 §113 的结论(113 µs 的线性响应 + 空 body + 锁在锚定路径上三点已足够) |

| R53 | 直接删掉 `have_work:` 的锚定锁(seqlock 复读 gen) | **已回退(会挂死)**。效果极佳(每层 1.16→**0.83**,空 gather 0.70;空区 177→18.8 µs),但 `test_block23_equiv.py` 两次确定性超时 rc=124。原因:`parallel_for_impl` 464/465 行是**先掀 gen、后读 `start_`**,锁在这一窗口内提供互斥;seqlock 只查 gen 变化,查不到"新 gen + 旧 start_"的撕裂 ⇒ worker 丢票 ⇒ `remaining_` 永不归零。**正确修法见 NOTES §114(奇偶 seqlock / per-worker 槽)**,不要再用"只复读 gen"的朴素 seqlock |
| R54 | 删除 `task_` / `sharded_task_` 的 `std::function` 构造 | **保留(安全)**。全文确认两者只被赋值、从不被读取;删除后对拍通过。 |
| R55 | 用 `test_block23_equiv.py` 作为去锁类改动的门禁 | **采纳为硬规则**:任何动 `numa_pool` 同步结构的改动,必须先过该对拍(它比 bench 更容易暴露挂死:bench REP=20 三次全过,对拍两次全挂) |

| 日期 | 改动 | 依据 |
|---|---|---|
| 2026-xx | 追加 R53(去锁已回退 + 挂死根因)、R54(死代码删除保留)、R55(对拍作同步改动门禁);NOTES §114 记录"1.16→0.83/0.70 但挂死"与下一轮修法 | 第 68 轮二分实验 |

| R56 | 奇偶 seqlock 去锚定锁(奇数=发布中/偶数=就绪) | **成功,已保留**。对拍两次通过(7 OK、无挂死);DEDUP=12 每层 1.14-1.24 → **0.78 ms**(最佳 0.72),DEDUP=23 → 0.88-0.95。这是 §113/§114 那条线的正确修法,不要回退 |
| R57 | gather 只按"每行一个 job"(NASS=36)并行 | **已替换**。36 线程搬 8 KB 冷数据 ⇒ ~110 µs(2.4 GB/s);切成 4×2 KB(144 job)后 37-47 µs,字节级等价。 |

| 日期 | 改动 | 依据 |
|---|---|---|
| 2026-xx | R56(奇偶 seqlock 成功保留)、R57(gather 切段);NOTES §115 | 第 69 轮 |

| R58 | 把 ExpBuf 缓冲区的 `resize(need)` 改成 `capacity()<need ? reserve(need)`(只用 `.data()`) | **已回退(算错)**。对拍 0 OK。虽然 grep 在 `moe_v2.hpp` 里只看到 `.data()` 用法,但确实存在未定位的 `size()` 依赖。**第二次踩同一个坑(R42 首次)** —— 再试前必须先全文定位 `ExpBuf` 各缓冲区的 `size()/end()/begin()` 使用点。当前 `resize` 的 24-34 µs 成本保留待修 |
| R59 | 手设 `XIAOTU_MOE_SHARDSPLIT=32` 想减少锁争用 | **不要用**。0→0.81、16→0.78(自动值)、32→0.96 ms ⇒ 32 明显更差;自动值已近最优 |

| 日期 | 改动 | 依据 |
|---|---|---|
| 2026-xx | R58(reserve 替换已回退)、R59(SHARDSPLIT 大值更差);NOTES §116(提前预取 0.78→0.76-0.77) | 第 70 轮 |

| R60 | 只增不减 resize(`if (size()<need) resize(need)`)替掉每次都 resize | **成功,已保留**。resize 24-34 → 1.4-1.5 µs、对拍 7 OK、最佳 0.75 ms。注意与 R58(reserve)的区别:`size()` 语义不变 ⇒ 不会算错。**R58 的正解就是这一条,reserve 不要再试** |

| R61 | `XIAOTU_MOE_GEMM_NR=16` | **不要用**。交替 min:8→0.75/0.75、16→0.77/0.78、4→0.79 ⇒ 16 明显更差,保持默认 8 |
| R62 | 提高 `XIAOTU_MOE_SPIN_IDLE_US` 到 2e6 以减少 park/唤醒 | **不采纳**。交替 min:默认(min 0.71)优于 2 s(min 0.74)。空区探针里出现的"119 µs→17.1 µs"在真实配置下不可复现 |
| R63 | 用 `XIAOTU_MOE_PROFILE` 的 A/B/C 绝对值做优化依据 | **停止**。同一配置 A 在 18.5-76.1 ms 间摆动(sum 可到 93 ms > 40×层时间)⇒ 该相位计时被污染,只可看 B(稳定)。要分解必须在 worker 内累计 |

| R64 | 用 gather 把每个专家的激活行拷成连续 `xg` | **已被 row-map 取代(永久删除)**。内核只在一处需要连续激活行(FP32 转换),把"行号"交给它即可;省掉 25 µs memcpy + 17 µs 并行区。`XIAOTU_MOE_NOGATHER/SERIAL_GATHER/NOOP_GATHER/GCHUNK` 随之作废,**不要再加回 gather** |

| R65 | `XIAOTU_MOE_DPBF16=1`(vdpbf16ps 点积) | **再次否掉,不要开**。交替 min-of-3(D=12):0.76/0.72/0.72 vs base 0.68/0.68/0.67 ⇒ 始终慢 7-12%。§109 微基准的 1.98× 在引擎里被"权重/激活现转 bf16 对"的开销吃掉 |
| R66 | 假设基准形状下引擎受 DRAM 带宽限制 | **否掉**。线程扫描显示时间近似严格 ∝ 1/N(60→1.70×、30→3.28×)⇒ 是**线程吞吐受限**;每线程 2.2-2.3 GB/s(T≤60)与 lk 同级。所谓"0.65 ms 地板"其实是"每线程吞吐上限 × 固定开销",不是内存墙 |

| R67 | 调 `XIAOTU_MOE_NSHARD` 想再提聚合带宽 | **已到边界**。NS=2/4/8→2.09/1.20/0.73 ms,NS=16 直接崩(50 ms)。**保持 default 8** |
| R68 | 用 `scripts/bwprobe.c` 的输出当"机器带宽上限" | **不可用**。T=120 的 flat 读只有 87.5 GB/s,**低于我们引擎的 226 GB/s** ⇒ 探针自身延迟受限。§35 的 234 GB/s 同类,不要再用它论证"贴上限" |
| R69 | 加大 `XIAOTU_MOE_THREADS` 到 192 | **不要**。120→0.67/0.68,192→0.74。用户"每 CCD 4–5 核"的规则被实测确认最优 |

| R70 | 静态跨步均分(`XIAOTU_MOE_STATIC_PART=1`,零原子领票)替掉动态领票 | **无收益,已回退**。交替 min-of-3:静态 min 0.68/均值 0.687 vs 动态 min 0.65/均值 0.673;对拍 7 OK(数值无误)。⇒ **票号原子争用不是 15% 聚合损失的来源**,不要再往这个方向改 |
| R71 | 认为 D=12 的 226 GB/s 是"贴到内存墙" | **否掉**。同一引擎在 D=48 跑到 350 GB/s 聚合(403 MB/1.15 ms)⇒ 内存系统能给更多。D=12 的低带宽是"固定项 0.383 ms + bytes/525 GB/s"里的固定项造成的,不是硬件墙 |

| R72 | 认为 0.383 ms 固定项 = `a32`(bf16→fp32)每个 job 重转整个 M×K 块 | **否掉**。直接仪器实测:1.25-2.88 µs/调用 × 3072 次/层 = ~5.8 core-ms ⇒ ÷120 = **~49 µs(7.3%)**,远小于 383 µs。仍值得以后收掉,但**不是**固定项的主因 |

| R73 | 把按组 scale 折进权重(`w'=w*sv`,省掉每行第 3 条 FMA) | **不等价,已回退**。对拍 0 OK,max_rel 1.6-2.7(2× 量级错误,非舍入)⇒ `sv` 与该 32-k 组的对应关系不像指令账假设的那样。理论收益仅 5.6%,**不要再试这个方向** |

| R74 | 认为 SHARDSPLIT=32 的 +0.29 ms "大部分"来自 a32 转换 ×4 | **部分否掉(重要更正)**。按 `(2·subA+subB)×na` 重算:subA 8→32 只把转换从 49 µs 抬到 ~121 µs(+72 µs),仅占 +290 µs 的 ~25%。其余来自 job 过小的其他固定成本。**不要把"提转换再提 subA"当主线**(最多几十 µs) |

| R75 | 认为"每 CCD 交付上限 ~9 GB/s"是硬天花板 | **不成立(自查否掉)**。D=48 实测 14.6 GB/s/CCD(350 GB/s 聚合)高于该值 ⇒ 天花板不是绝对的,而是 **L3 驻留口径比 DRAM 流式口径更慢**这一缓存层级现象。不要再把"每 CCD 上限"当硬约束 |

| R76 | "服务端形状(na=32)每线程 2.78-2.89 GB/s 已反超 lk 的 2.2" | **作废(跨口径错误)**。lk 的 2.2 是 **DEDUP=12** 口径;同口径严格对照:我们 D=12 1.94 vs lk 2.21(0.88×)、D=23 2.50 vs lk 3.13(0.80×)。**两个验收形状都落后 12%/20%**,且差距形状一致 ⇒ 不能用"部署形状已达标"来结案,也削弱了"缓存层级硬件差异"的解释 |

| R77 | 按 §135 立项"K-major 列 lane 内核"来补 12-20% | **撤销(自我更正)**。同构 op 账:列 lane 是 `208/M + 48 + 1` 条/512 MAC ≈ 0.23 条/MAC,与现行 k lane 的 0.229 **完全相同**;而若两者真 op-bound,lk 应快 ~1.9×,实测只快 1.14-1.25× ⇒ **都不是 op-bound**。省指令的关键是"解码复用 8 行"(需 me=8,路由决定),不是"摊到列上"。**不要再启动这个重写** |

| R78 | 把 master(调用方)线程绑到 worker 之外的核,避免其自旋偷核 | **无影响**。taskset 128-191 / 0-127 / 不绑交替:0.66/0.68/0.66/0.67 ms,全在噪声内。不要再动 master 绑核 |
| R79 | 认为 harness 把冷启动摊进均值导致虚高 | **否掉**。REP=5/60/300/600 → 0.66/0.65/0.66/0.66 ms ⇒ 启动 <0.1 ms。测量方法本身干净 |
| R80 | 用整库助记符统计直接比较两个 .so 的"指令效率" | **只能看同库内比例**。两库都含未选中的 ISA 回退路径、且 lk 库代码量大得多(vmovaps 7384 vs 862),跨库总量不可比。结论仅取"形态差异"(列 lane 广播 vs K lane;lk 解码无 vpunpck) |

| R81 | "L3 驻留(151 MB)反而比 DRAM 流式(403 MB)慢,故应绕过 L3" | **否掉**。等 na 对照(NENGINES=1 vs 6+ROUNDROBIN,工作集 151 MB vs 906 MB)显示**工作集越大越慢**(0.65-0.69 → 0.85-0.92 ms)。D=48 的高带宽来自 **na 大 ⇒ job 多 ⇒ 并行度高**,与 L3/DRAM 归属无关。**不要再往"绕过 L3 / 非临时载入"方向做** |

| R82 | "na=12 是并行度不足,细分 job(subA)能补" | **否掉**。SHARDSPLIT 8/16/32 = 0.690/0.720/0.920 ms(两次重复完全一致)⇒ 加 job 单调更差。na=12 是 **compute(op)受限**:MAC ∝ NASS=36(常数)而字节 ∝ na,故 na≤12 时时间平坦、na≥20 才转访存 |
| R83 | §136 "两个内核都非 op-bound" 的推论 | **修正**。该推论只比较了 lk 的绝对 op 数,没结合"na=12 实测处于平坦区(⇒op 受限)"这一事实。**正确表述**:na=12 口径下我们确实 op 受限(~170M 条 zmm ≈0.57 ms ≈ 实测),所以"减指令"是有效杠杆;§135 的方向没错,错的是 §136 的自我否定 |

| R84 | 把"减指令"当作 na=12 的杠杆(§140 的 op-bound 推论) | **部分否掉**。给 bf16 路径加列分块后它确实从 0.72-0.76 变到 0.64-0.66(证实 GEMV 是 R65 病根),但 FMA 侧减掉 1/3 的 op 只换来 3-4% ⇒ **na=12 也不是单纯 op-bound**。不要再按"数 op 数"来预测收益 |
| R85 | 启用 `XIAOTU_MOE_DPBF16=1` 作为默认路径 | **不能启用**:精度门禁 me=2 用例 max_rel 8.42e-3 > 项目门限 2e-3(其余用例都 OK)。列分块改善不了该精度问题。保持 fp32 为默认 |

| R86 | draft 上 GPU(`XIAOTU_MOE_RESIDENT_DRAFT=1` + `GPU_RESIDENT_LAYERS=43-45`)配 KV 8 GiB | **配置需调小才可用**。draft 3 层共 9.57 GiB + KV 8 GiB + 固定 14.2 GiB = 31.8/39.5 GiB ⇒ 预热阶段 `Triton Error [CUDA]: out of memory`。**不是方向错,是 KV 与常驻层抢显存**;已改为 KV 4 GiB 重试(进行中) |

| R87 | draft 3 层常驻 + KV 4 GiB | **仍失败**:`Engine core initialization failed`(预热阶段,无显式 OOM)。两次尝试合起来的结论:**单卡 39.5 GiB 装不下「固定 14.2 + draft 3 层 9.57 + KV + Triton 工作区」**。下一步应改为**只常驻 1 层**(需先确认 MTP 真实层号)或 KV 2 GiB,或等 TP=2 修好 |

| R88 | draft 3 层常驻 + KV 4 GiB 但 maxlen 仍 262144 | **失败原因是配置校验而非内存**:`7.19 GiB KV needed > 4.0 GiB`。**改 KV 必须同步改 maxlen**(29.4 KB/token 硬绑定)。不要把它当成"显存不足"记 |

| R89 | TP=2 + 11 个目标层常驻(预算 18 GB) | **推理时 OOM**:`Triton Error [CUDA]: out of memory`(engine 在第一个请求上死)。固定 7.1 + KV 8 + 17.5 = 32.6 GiB/卡 ⇒ 留给 Triton 预填充内核/autotune 的 <7 GiB 不够。**Triton 工作区是常驻层数的实际上限**(TP=2/maxlen262144 下约 8-9 层) |

## R90. TP=2 长上下文(32K)后 `persistent_topk` 非法访存致 EngineCore 死亡
- **配置**:TP=2 + 8 常驻层 + V1 runner + execute-timeout 3600,`max_num_batched_tokens=8192`。
- **现象**:32K tokens 预填充(客户端 TTFT 36.1 s)之后,Worker_TP0 在
  `sparse_attn_indexer → persistent_topk` 抛
  `occupancy query failed: an illegal memory access was encountered`,
  紧接着 EngineCore 因 dequeue 超时 `RuntimeError: cancelled`,服务端 500,引擎退出。
- **不是**:Triton workspace OOM(R89 的那种)、也不是显存不足(GPU 0/1 = 38.2/40 GiB,无 OOM 报错)。
- **状态**:**未解决**,已列入下一轮第一优先级;在定位前,**不要把 32K 以上长预填充当成可用能力**,
  也不要把它写进交付配置。

## M8c. `kill_serve.sh` 只杀 API server、留下孤儿 EngineCore/Worker(又被白等一轮)
- **现象**:`scripts/kill_serve.sh` 报 `killed=[]` 却"完成";下一次启动报
  `ValueError: Free memory on device cuda:1 (13.16/39.49 GiB) on startup is less than
  desired GPU memory utilization (0.9, 35.54 GiB)`,白等一轮 13 分钟加载后才失败。
- **根因**:vLLM 的引擎子进程会**改写 `argv[0]`**。实测 `/proc/<pid>/cmdline` 首段就是
  `VLLM::EngineCore` / `VLLM::Worker_TP0` / `VLLM::Worker_TP1`。旧脚本只匹配
  `ENV_PY` 前缀(`.../envs/vllm-xiaotu-moe/bin/`)⇒ 只有 API server 命中并被杀,
  EngineCore+Worker 存活并各占 ~26 GiB/卡。
- **修复**(`scripts/kill_serve.sh`):①argv[0] 增加 `VLLM::` 前缀匹配;
  ②残留检查改为 `grep -acE 'vllm serv[e]|VLLM::(EngineCore|Worker)'`;
  ③新增**硬校验**:任一卡 `used >= 1024 MiB` 就 `exit 3`,不再"假完成"。
- **验证**:对 3 个真实孤儿(1193440/1193547/1193548)`killed=[(…,'VLLM::EngineCore'),
  (…,'VLLM::Worker_TP0'),(…,'VLLM::Worker_TP1')]`,三卡全部回到 0 MiB,exit=0。
- **教训**:凡是"等显存释放"类脚本,**必须自带失败退出码**,否则它会把 M9 伪装成
  "环境问题",让人在错误方向上多花一整轮。

## R91. TP=2 常驻专家层(GPU-resident MoE layers):收益仅 2-4%,且是 32K 崩溃的诱因 —— 从交付配置移除
- **配置**:`XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-11` + `XIAOTU_MOE_RESIDENT_BUDGET_GB=14`
  ⇒ 实际常驻 8 层(每 rank 1.59 GiB/层)。
- **收益(同数据集 nat8192/nat32768,同会话,TP=2)**:
  8192 1333-1345 → **1371-1396 t/s**(+2.3%);32768 1160-1164 → **1185 t/s**(+1.6%)。
  DMA 模型预测应为 ~9-10% ⇒ 实测只有 1/4。
- **代价 / 风险(决定性)**:
  | 配置 | 32K 尝试 | 崩溃 |
  |---|---|---|
  | 常驻=0 | 5 | 0 |
  | 常驻=8 | 4 | **2** |
  一次是 `persistent_topk ... illegal memory access`(Python 可见),一次是
  `VllmWorker-0 died unexpectedly (exit code: None)`(原生信号崩溃,无 traceback)。
- **结论**:8 层常驻 = 用 12.7 GiB 显存换 2-4% 预填充,同时引入长上下文段错误 ⇒
  **净负**。交付配置改为 `XIAOTU_MOE_GPU_RESIDENT_LAYERS` 不设(常驻=0)。
- 注:R89(Triton OOM 只在常驻层存在时出现)与本条同源,可一并归档为该机制的第三个反例。

## R92. 在前向热路径里插 GPU 事件 + `torch.cuda.synchronize()` 埋点 —— **把引擎挂死,数据全废**
- **动机**:`_gp_add` 的 seg/rest 都是 host 侧 `perf_counter`,而内核是**异步**的,
  所以此前所有 prefill 数字里**没有一项是 GPU 侧时间**;想量出"每层 compute / 未重叠 DMA"。
- **做法(已回退)**:在 `gpu_moe_layer` 里每层建 3 个 `torch.cuda.Event`,`e_in`(进层)、
  `e_k0`(等到预取 ready 后)、`e_k1`(内核发射完);攒够 40 层就 `torch.cuda.synchronize()`
  读 `elapsed_time`,打印 `[gp-gpu] wait/kern/layer`。
- **后果(实测)**:
  1. 读出的 `kern`/`layer` 是**负值**(`wait=+78.4/+22.3/+5.8ms`,而 `kern=-64.4ms`)——
     三个事件的时基互相不一致,说明该路径上的"当前流/线程"语义与我的假设不符;
  2. CPU 线程池**卡死**:日志连续 4 分钟刷
     `[pool] WATCHDOG(sharded) gen=650 total=256 rem=1 exec=256` + `node 0..7 jobs=32 pulled=47`,
     同时 `shm_broadcast: No available shared memory broadcast block found in 60 seconds`;
  3. 常驻=0 下**第 2 次 8192 就崩**:`Worker proc VllmWorker-1 died unexpectedly (exit code: None)`
     ⇒ 本轮所有 `[gp-gpu]` 数字**作废**;`seg` 也从 4.66-6.39ms 涨到 11.76-22.31ms(埋点自身开销)。
- **结论**:**不要在 MoE 前向热路径里做任何会阻塞/同步的埋点**。GPU 侧分解改用
  **nsys(进程外,不侵入)** —— 仓库里已有 `scripts/tune_nsys.sh`。
- **副产物(必须记住)**:这条**反过来污染了我对 32K 崩溃的归因**。r119 之前"常驻=0
  5/5 通过"是**无埋点**状态测的,可信;r119 的常驻=0 崩溃**不能**算作反例,
  因为埋点是新引入的变量。⇒ 第 119 轮需在**无埋点**状态下重测常驻=0 的 32K 稳定性,
  才能维持/推翻 R91。

## R93. 【更正 R92】"热路径同步埋点导致池卡死" —— 归因错误,病在生产代码
- r121 在**无埋点、无常驻、纯交付配置**下复现了同一个
  `[pool] WATCHDOG(sharded) gen=... total=128 rem=1 exec=127` ⇒ 池卡死与埋点无关。
- 真实链条:`numa_pool.hpp:713-721` 分片调用等不到 `remaining_==0` ⇒ 300s 后 **`abort()`**
  ⇒ worker 原生死亡(exit code None)⇒ `cancelled` ⇒ 500。
- R92 中仍然有效的部分:该 GPU 事件埋点方案本身不可用(读出负值)、
  且**不应在前向热路径做同步埋点**。R92 中"是它导致挂死"的结论**撤回**。

## R94. "陈旧 `node_base_` 读导致票据被丢弃" —— 假设否证,改动已回退
- **假设**:`numa_pool.hpp:937` 在领票前把 `node_base_[myn]` 缓存进 `base`,而调用方在发布期
  (`:693` 置奇代 → `:695` 写 `node_base_`)写它 ⇒ worker 用**上一代的 base** 算
  `loc = t - base`(t 已是本代票据)⇒ `loc >= nj` ⇒ `:943 break` ⇒ 票据被消费却不递减。
- **改动**:把 `base` 的读移到 `g == gen`(偶代)确认**之后**,并附时序论证
  (worker 看到偶代且本代仍有未递减任务时,调用方必然还在 `:709` 等待、不可能在写 `node_base_`)。
- **结果**:**无效**。重新编译 5 个 ISA 变体后重启,连续两次尝试仍是同一签名:
  | 尝试 | dump |
  |---|---|
  | a1 | `total=128 rem=1 exec=127` |
  | a2 | `total=384 rem=1 exec=384` |
  与修复前**逐字相同** ⇒ 陈旧 base 不是(至少不是唯一)原因。
- **已回退**(`git checkout`),并已后台重建基线 `.so` 使源码与产物一致。
- **教训**:在并发协议上"读起来最可疑"的那一处不一定是真凶;必须先加**计数埋点证明**
  再改(这条我在 §174(e) 已经写下,却没遵守,浪费了一次 13 分钟加载 + 一次重编译)。

## R95. 【更正·我错了】"CPU 通路会越过 EP shm stride 8 倍写入" —— **不存在此越界**
- **我的错误断言**(第 121 轮向用户报告):"CPU 通路的 EP shm 缓冲按固定
  `XIAOTU_MOE_EP_SHM_TOKENS=1024` 分配 stride,而 `gpu_prefill_min_tokens()` 默认为 0
  ⇒ CPU 通路会拿到 qlen 高达 8192 的批次,越过 1024-token 的 stride **8 倍**"。
- **实际情况:有两道独立门禁,不存在越界写。**
  1. **引擎侧**(`binding.cpp:437-440`):
     ```cpp
     const size_t bytes = (size_t)qlen * engine->config().hidden_size * sizeof(float);
     if (bytes <= ep->capacity) {   // ← 只在放得下时才走 shm barrier
     ```
     放不下就**不进 shm 路径**(`ep->capacity = stride`,由 `configure_ep` 传入)。
  2. **Python 侧**(`hybrid_model.py:983`):
     ```python
     _need_allreduce = self._ep and qlen > self._ep_shm_tokens
     ```
     超出容量时回退 `tensor_model_parallel_all_reduce`(NCCL)。
  ⇒ **纯 CPU 预填充(qlen=8192)是正确的**:引擎跳过 shm、Python 做 NCCL all-reduce。
  用户要求的"可以慢,但是不能出错"**当前已经满足**。
- **错误性质**:我只读了 Python 侧的分配 + 注释(`hybrid_model.py:736-738`),
  **没有检查引擎侧**就用了"8 倍越界写"这种确定性措辞,并让用户据此下了指令。
  这是本会话第 5 个错误结论,也是**最危险的一个**(把性能问题说成了内存破坏)。
- **仍然成立的、真实存在的小问题(仅性能,非正确性)**:
  `hybrid_model.py:736-738` 的注释断言"取 `max(预分配, 阈值)` 即可覆盖 CPU 路径的最大 qlen",
  但代码实际只取 `XIAOTU_MOE_EP_SHM_TOKENS`(默认 1024)。若把 GPU 阈值设到 > 1025,
  CPU 通路就会**静默地**走 NCCL 回退 —— 正确但慢,且无任何日志提示。
  ⇒ 建议(未实施,待用户确认):把 `_toks` 改为按 `max(EP_SHM_TOKENS, gp_min-1)` 计算
  (gp_min>0 时),并在真正走到回退时打一次 warning。**纯 CPU 模式(gp_min=0)不应按
  8192 预分配**(stride 134 MB × 43 层 × 2 rank ≈ 11.5 GB /dev/shm),回退即正确。

## R96. `XIAOTU_MOE_EP=0`(关专家并行、每 rank 冗余跑全 256 专家)—— 排除
- **动机**:隔离跨 rank 归约的成本(§193 怀疑它占 TP=2 解码差距的大头)。
- **结果**:确实隔离出了归约成本(`compute` 1.3 → **0.85 ms/层**,即归约 ≈0.45 ms/层),
  **但该配置本身不可用**:
  | | EP=1 | EP=0 |
  |---|---|---|
  | period | 1.96 ms | **3.68 ms** |
  | rest | 0.72–0.74 ms | **2.85 ms** |
  | C=1 解码 | 10.46 t/s | **0.01 t/s(TPOT 299508ms)** |
  并出现 `qlen=22 period=270ms rest=260ms` 的严重停顿。
- **结论**:冗余执行双倍专家的代价(主机↔设备往返 + 双倍访存)远大于省下的归约 ⇒
  **EP=0 从候选配置中排除**,不作为交付形态。
- **保留价值**:它给出了"无归约地板"(0.85 ms/层),量化了归约的代价,方向仍然指向
  "优化 shm 归约"或"解码改用 TP=1"。

## R97. 【归因更正】"引擎不稳定"不是一个问题,是**两个独立问题**
- 第 120-138 轮我一直把"请求几次就 abort"当作**单一**故障(线程池领票/递减不配对),
  并已修好三处、复发率大幅下降。
- 第 139 轮同一版本连跑 4 个会话:前 3 个全过,第 4 个失败 —— **但日志里没有看门狗**,
  而是 `CUDA error: an illegal memory access was encountered` → worker `died unexpectedly
  (exit code: None)` → `EngineDeadError` → 500。
- ⇒ **存在第二种、独立的故障**:**GPU 侧访存越界**。它与第 116 轮**最早**看到的签名一致
  (`persistent_topk ... illegal memory access`,R90),而我从未处理过它。
- **教训**:早期把两个签名(有看门狗 / 无看门狗)混成一句"引擎不稳定",
  导致我在其中一类上投入了十几轮。**今后凡是"不稳定",第一步就要按日志特征分类,再动手。**

## R98. 用 Triton 回退(`use_persistent_topk=False`)绕开故障 B —— **不行**
- **动机**:§206 怀疑故障 B 来自 `persistent_topk` 的固定 1 MiB radix workspace 越界,
  于是把该分支关掉、回退到 `ops.top_k_per_row_decode`。
- **稳定性**:关掉后跑到 **95 个请求无故障 B**(已超过补丁前 §202 首次复现的 96 请求长度附近)。
- **但性能不可接受**:
  | | 回退后 | 回退前 |
  |---|---|---|
  | 解码 C=1 | **0.89 t/s**(TPOT 799.67 ms) | **10.46 t/s**(TPOT 91.05 ms) |
  | 32K 预填充 | 1084 t/s | 1157 t/s |
  ⇒ **解码慢 11.7 倍**。
- **结论**:该内核是解码性能的关键。**回退方案否证**;故障 B 必须通过
  "让这个内核正确地工作"(放大 workspace / 修上游)来解决。
- 主线仓的补丁**保留**(作为诊断/兜底),但**不作为交付配置**;
  交付必须用 `persistent_topk`。

## M8d. `pkill -f "compute-sanitizer"` 第 4 次自伤(同一家族)
- **现象**:第 166 轮里我用 `pkill -f "compute-sanitizer"` 清理 sanitizer 残留,
  结果**把我自己那条命令行也匹配并杀掉了**(我的 bash 命令行里就含这个字符串)⇒
  整个后台作业被 SIGTERM 终止,白起一次服务。
- **教训(第 5 次重申,这次必须变成条件反射)**:
  **凡是 `pkill -f <pattern>`,只要 pattern 出现在自己这条命令行里,就会自杀。**
  正确写法:`pkill -f "compute-sanitize[r]"`(括号技巧**在这里有效**,因为 pattern
  本身不含字面量 `compute-sanitizer`),或改用 `pgrep -f "...[r]"` 再逐个 `kill`。
- 注:`scripts/kill_serve.sh` 早已用"读 /proc/<pid>/cmdline 的 argv[0]"避开了这个问题,
  **清理由它负责的进程时应当只用它**;`pkill -f` 只用于它管不到的东西(如 sanitizer),
  并且必须带括号保护。

## R99. draft/speculator 与目标模型**共用 EP shm barrier**(已修,已验证)
- **现象**:投机解码净负(position-0 接受率仅 ~26%,而训练良好的 draft 应 60~80%);
  用户在本机实测 lk_moe 加 draft 能到 50 t/s ⇒ **draft 在这台机器上本可用**。
- **根因**:`_ep_shm_attach` 只用 `layer_idx` 命名 shm 文件,而 vLLM 的 DSpark 起草模型是
  目标模型的**又一份完整实例、层名(prefix)相同**(插件自己的注释就这么写的)⇒
  **目标第 L 层与 draft 第 L 层共用同一个 barrier 头与同一块部分和区域** ⇒
  两套调用序列交织推进世代、互相覆写部分和 ⇒ **draft 的 MoE 算错**。
  主模型的**输出质量**看不出来(验算把错的 draft 拒掉),只表现为**变慢**。
- **修复**:shm 文件名加入模型实例判别(`_shm_instance_tag`:目标 `T` / draft `D1`…),
  独立计数以免与 `_is_duplicate_model_instance` 的 `seen` 互相干扰。
- **验证(默认 `EP_SHM=1`)**:position-0 接受率 **26% → 74.4%**,每步接受 1.3 → **1.95** token,
  C=1 单流 6.30 → **7.11 t/s**,与 `EP_SHM=0` 上界(72.5%)一致。
- **注**:投机仍未转正(7.11 < 不开投机 10.76),原因是**每层固定开销**导致
  "6 个 token 花 6 倍的钱" —— 见 §240(b),那是下一场仗。

## R100. `--speculative-config '{"method":"mtp"}'` —— **不是"引擎不支持",是"checkpoint 里没有 MTP 权重"**(第 208 轮结案)
- **试过**:`{"method":"mtp","num_speculative_tokens":5}`(带/不带 `model` 键都试),
  三次启动全部失败,报 `KeyError: 'model.layers.43.mtp_block.main_norm.weight'`,
  栈顶在 **`vllm/models/deepseek_v4/nvidia/mtp.py:480`**(即主线**原生**的
  DeepSeek-V4 MTP 实现,**不是**通用 `deepseek_mtp.py`)。
- **曾经的错判**:由此推断"主线缺 nvidia 专用实现,需要移植 `Lvllmds4-x` 的 1400 行",
  并写成 §254 的"架构级根因"。**该结论作废**。
- **真因(权重清单直接证伪)**:把 `model.safetensors.index.json` 的 72317 个 key 全量统计,
  `enorm/hnorm/e_proj/h_proj/eh_proj/shared_head/mtp_block` 的出现次数**全是 0**;
  `mtp.{0,1,2}.*` 的 4705 个张量是 **DSpark 草稿**(`main_proj/main_norm` + 3 个
  decoder block + `mtp.2` 上的 `norm/hc_head_*/markov_head/confidence_head`),
  与 `nvidia/dspark.py:_remap_dspark_name()` 的表逐条对上。
  ⇒ **这件 checkpoint 是"带 DSpark 草稿"的版本,不含 MTP 权重。**
- **处置**:**永久放弃 `method:"mtp"`**,不再尝试改名映射/移植(见 NOTES §255)。
  原生命中路径 `DeepSeekV4MTP` 就在这里、是好的,只是没有权重喂它。
- **教训(铁律 7)**:比对参考实现前,必须先确认"参考"是**哪一棵被 import 的树**、
  跑的是**哪条权重布局**;否则会把"命名/布局不匹配"误判成"架构缺失"。

## R101. `XIAOTU_TORCH_PROFILE_DECODE` + `cudagraph_mode=FULL_DECODE_ONLY`(捕获被作废)
- **做法**:在一个已经成功的 cudagraph 配置(mir2)上只加
  `XIAOTU_TORCH_PROFILE_DECODE=/tmp/decprof.json`,想拿 decode 的 kernel 分解。
- **结果**:启动在 `compile_or_warm_up_model` 失败:
  `CUDA error: operation failed due to a previous error during capture
  (cudaErrorStreamCaptureInvalidated)`;日志紧邻 `SyncActivityProfilerHandler profiler_start/stop`。
- **根因**:`_maybe_profile_decode()` 惰性 `torch.profiler.profile().__enter__()`,而"第一次
  MoE forward"就发生在**捕获期** ⇒ profiler 启动的同步操作落在捕获区内。
- **处置**:回退(不加 profiler)。**在 FULL_DECODE_ONLY 下分层诊断只用
  `XIAOTU_CD_TIMING`(host 回调内,零同步、每次 replay 都跑)。**
  同理 `XIAOTU_DEBUG_L1` 等含 `.item()`/`.cpu()` 的开关在捕获下都会破坏启动。

## R102. TP=1 + `GPU_UTIL=0.90` + 草稿常驻(10.12 GiB)= 预热期 CUDA OOM
- **做法**:为了"草稿永远在 GPU",把草稿层 43-45 加进 `LVLLM_GPU_RESIDENT_MOE_LAYERS`
  (TP=1 ⇒ 草稿 10.12 GiB/rank),`GPU_UTIL=0.90`、`MAXLEN=8192`、`EAGER=1`
  (tag `lkport25spec`)。
- **结果**:加载与 KV 划分都成功(`Model loading took **20.66 GiB**`= 目标 10.44 + 草稿 10.12,
  证明草稿确实进了显存;`Available KV cache memory: 5.85 GiB / 15,941 tokens`),
  随后**预热阶段 OOM**:
  `torch.OutOfMemoryError: Tried to allocate 2.00 GiB. GPU 0 has 39.49 GiB total, 1.89 GiB free,
   this process has 37.33 GiB in use` ⇒ `RuntimeError: Engine core initialization failed`.
- **根因**:`GPU_UTIL` 只约束 **KV cache 划分那一刻**的账;之后 lk 引擎自己还要在 GPU 上分配
  缓冲(`prepare_decode_buffers`/`_initialize_cuda_graph_buffers`/gpu_prefill staging,实测 ≈5-6 GiB)
  以及稀疏 MLA 预热的临时显存(≈2-3 GiB)。TP=1 下 `0.90` 没有给这部分留余量。
- **处置**:①**改为按 TP 估账**(脚本内 `MODEL_EST_GIB=auto` → TP=1:11 / TP=2:7,
  另加 `LK_BUF_GIB=6`、`WARMUP_GIB=3`、`KV_MIN_GIB=6`),现在 TP=1+util0.90+草稿常驻会被
  护栏**提前拦下并禁用草稿**(`need 36.12 > budget 35.50`),不再浪费一次 13 分钟加载;
  ②**回到作者配方的 TP=2 / `GPU_UTIL=0.80`**(草稿只 5.34 GiB/rank),见 `lkport26spec`。
- **教训**:显存护栏必须把"KV 划分之后才分配的第三方缓冲"算进去;只按
  `util×显存 − 模型 − 草稿` 估账会低估 8-9 GiB。

## R103. 作者配方原样(图模式 `EAGER=0`)+ 投机解码 ⇒ 捕获失败(`cudaErrorStreamCaptureInvalidated`)
- **做法**:完全照作者配方跑(`TP=2 / GPU_UTIL=0.80 / MAXLEN=1048576 / SEQS=2 / PREFETCH=1 /
  **不加 `--enforce-eager`** / `SPEC=auto` ⇒ dspark + 草稿常驻 43-45`),tag `lkport26spec`。
- **结果**:权重与 KV 都成功(`Model loading took 11.39 GiB`/rank = 目标 6.22 + 草稿 5.34 ✅),
  卡在 **CUDA graph 捕获**:
  ```
  forward_fn(CUDAGraphMode.NONE) → model(...) → layer(...) → self.ffn(x, input_ids)
    → fused_out = self.routed_experts._cpu_decode(...)
    → cuda_graph.capture_end()
  torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture
  (cudaErrorStreamCaptureInvalidated)
  ```
  ⇒ `RuntimeError: Engine core initialization failed`。
- **定位**:失败点就在 **lk 的 `_cpu_decode`**(CPU 目标层,43 层)被放进捕获区时。
  lk 的 `_cpu_decode` 本身没有同步(`routed_experts.py:1708`,只把 `current_stream().cuda_stream`
  传给引擎),所以违规发生在**我方引擎的 `cpu_decode` 内部**。
- **已排查**:引擎确实*有*捕获安全路径(`binding.cpp:50-90` 注释 + `cudaLaunchHostFunc` at `:585`,
  且 `cudaStreamIsCapturing` 分支 at `:388`)。但 `CpuDecodeState::ensure_buffers(..., retire=true)`
  在**捕获期间**若缓冲不够,仍会执行 `cudaHostAlloc`(`binding.cpp:100`)——**捕获区内分配显存/锁页内存是非法操作**,
  足以让 `capture_end` 报 `cudaErrorStreamCaptureInvalidated`。
- **最可能的触发条件(强怀疑,待证)**:lk 的图缓冲是按**序列数**而非**token 数**预分配的 ——
  `RoutedExperts._initialize_cuda_graph_buffers()`(`routed_experts.py:1694-1706`)把**全层共享**的
  `RoutedExperts.output_gpu` 开成 `(max_num_seqs, hidden)`;而投机解码的一步里
  目标模型要验证的 token 数 ≈ `num_seqs × (num_spec_tokens + 1)`(本次 2×6=12 > 2),
  于是 `ensure_buffers` 在捕获期增长 ⇒ 非法分配 ⇒ 捕获作废。
- **处置**:本轮先用 **`EAGER=1`**(我们移植时一贯的可用配置)拿到作者配方的其余全部参数
  (`lkport27spec`),不在这一步卡住;图模式留给下一步(见"下一步"栏)。
- **下一步(明确)**:
  1. 让引擎的 pinned 缓冲**一次开够**(`prepare_decode_buffers(max_qlen, top_k)` 里按
     `max_qlen × (1 + num_spec_tokens)` 或再乘一个安全系数分配),并保证**捕获期间 `ensure_buffers` 永不分配**
     (不够就用预先开好的最大 scratch,或直接报错而不是 `cudaHostAlloc`);
  2. 同时核对 lk 的 `output_gpu`(`(max_num_seqs, hidden)`,**全层共享**)在投机解码下是否越界:
     若目标验证批 = `seqs×(spec+1)`,则它必须按 **token** 数开,而不是序列数;
  3. 改完用 `EAGER=0` 重测(图模式对解码吞吐通常是 1.3-2×,值得修)。

## R104. 【流程教训】不要在模型加载期间跑重型引擎对拍 —— 会把 worker 挤成 OOM 被杀
- **现象**:`lkport27spec`(TP=2/EAGER=1)已经 `Model loading took 11.39 GiB` 成功、进入预热后,
  15:53:24 突然 `Worker proc VllmWorker-0 died unexpectedly, shutting down executor`
  → `RuntimeError: cancelled`(`shm_broadcast.py:701 acquire_read`)→
  `Engine core initialization failed`。**worker 日志里没有任何 Python 异常/CUDA 报错**(静默死亡)。
- **根因**:同一时间我在另一个进程里跑 `scripts/test_block23_equiv.py`(真实 MXFP4 层、256 专家、
  多组对拍 + torch 参考,单进程可吃几十 GB),而两个 rank 各自已把 ≈69 GiB 专家权重放在主机内存里。
  `/proc/vmstat` 的 `oom_kill` 计数为 **24**(内核 OOM killer 出手过)⇒ worker 被 SIGKILL。
- **处置/规矩(加到操作纪律)**:模型加载/预热期间**只做轻量工作**(读日志、写文档、git),
  不要在同一个 box 上跑 `test_block23_equiv.py` / 多引擎对拍 / 大内存基准。
  真要跑对拍,等 __serve 起来并测完__再跑,或者明确分时。
