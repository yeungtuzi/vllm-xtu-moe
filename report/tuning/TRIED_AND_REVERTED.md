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
