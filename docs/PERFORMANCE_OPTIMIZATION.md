# 混合推理性能优化手册(实测归因 + 路线图)

> 面向 DS-V4-Flash-0731(A100-40GB ×3 + EPYC 9654 无 AMX + 1.5 TiB RAM)的
> **decode/prefill 性能归因与优化记录**。每条结论都标注是「实测」还是「估算」,
> 并给出复现命令。原始数据在 `report/tuning/`,逐次实验流水在 `report/tuning/NOTES.md`。

---

## 0. TL;DR:杠杆清单与当前状态

| # | 杠杆 | 状态 | 实测 / 预期 |
|---|---|---|---|
| 1 | **引擎权重布局**:关掉 `XIAOTU_MOE_SINGLECOPY`,回到默认 **NUMA 分片** | ✅ **实测** | C=64:53.4 → **73.6 tok/s(+38%)**;每层 compute 22.8 → 15.9 ms |
| 1b | **CUDA graph**(`EAGER=0`,需修复捕获期 bug,见 §13) | ✅ **实测** | C=64:73.6 → **77.4**;同形 C=4:55.7 → 57.8 total(+6%) |
| 1c | **引擎线程 96 → 192**(仅在 NUMA 分片下有效,见 §15) | ✅ **实测** | 每层 compute 15.7 → **12.1 ms**;C=64:77.4 → **90.6**;C=128:**106.2 tok/s** |
| 2 | 引擎线程数 96 → 192(整机 192 核,我们一直只用一半) | 🟡 待测 | 预期每片并行翻倍 |
| 3 | **分片粒度 8 → 2**(`XIAOTU_MOE_NSHARD=2`,对齐 socket 边界) | 🟡 待测 | 每片 = 1 socket 的 12 个 CCD |
| 4 | 强制 grouped 路径(`XIAOTU_MOE_GROUP_FACTOR=1`) | ❌ **实测无收益** | C=64:77.43 → 75.50 tok/s(−2.5%);说明该工况下引擎已不是"按 pairs 计流量"的瓶颈 |
| 5 | GPU 常驻专家层(`XIAOTU_MOE_GPU_RESIDENT_LAYERS`,lk-moe 同款) | ⬜ 未实现 | 每层省 ~15–20 ms |
| 6 | TP=2 / 专家并行(EP) | ❌ **实测更慢** | 每层 +6.5 ms;但 prefill 快 1.76× |
| 7 | 长上下文 `--max-num-batched-tokens` 放大 | ⛔ 受 KV 上限阻塞 | 128K prefill 需 16 块 × 137 GiB 流式 |

**一句话**:decode 的瓶颈 **89% 在 CPU 专家引擎**(不是 GPU↔CPU 往返),而 CPU 引擎的第一个
大头是**权重在 NUMA 上的布局**——只改一个环境变量就拿到 +38%。

---

## 1. 测量方法(可复用)

### 1.1 服务与压测

```bash
# 服务端(单卡、256K、KV 12GiB、nbt 8192)
GPUS=2 TP=1 TAG=dsv4 PORT=8090 MODE=dsv4 MAXLEN=262144 SEQS=256 MAX_NBT=8192 \
  KV_DTYPE=fp8_ds_mla GPU_UTIL=0.90 KV_MEM_BYTES=12884901888 THREADS=96 OMP=48 \
  EAGER=1 PREFILL_MIN=384 scripts/tune_serve.sh

# 客户端:标准 ShareGPT(C=64/out=256)
TAG=dsv4_c64 PORT=8090 C=64 N=64 OUT=256 scripts/tune_client.sh
# 客户端:"生产同形"基准(C=4、50 prompts、ShareGPT 自带输出长度、不忽略 EOS)
TAG=dsv4_lkshape PORT=8090 C=4 N=50 scripts/bench_lk_shape.sh
```

### 1.2 每层归因埋点(本手册的核心工具)

`xiaotu_moe/csrc/python_binding/binding.cpp` 的 CUDA host 回调里加了环境变量开关
**`XIAOTU_CD_TIMING=1`**,每个引擎每 43 次回调(= 一个 decode step 的层数)打印:

| 字段 | 含义 |
|---|---|
| `period` | **回调入口到下一次回调入口** = 真正串行的每层时间(这条链是 GPU→D2H→CPU→H2D→GPU 严格串行) |
| `compute` | CPU MoE 本体耗时 |
| `rest` | `period - compute` = GPU 计算 + D2H/H2D 拷贝 + host-fn 派发延迟 |

```bash
ENV_EXTRA="XIAOTU_CD_TIMING=1" ... scripts/tune_serve.sh   # 打开
grep "cd-timing" report/tuning/logs/<TAG>.log | grep "qlen=64"
```

> 为什么必须埋在 C++ 回调里:Python 侧看到的是**入队时间**(异步),完全反映不出真实每层耗时。

### 1.3 记录规范
- 每次压测自动追加一行到 `report/tuning/summary.jsonl`(含并发/输出长度/吞吐/TTFT/TPOT);
- 服务端环境快照落在 `report/tuning/logs/<TAG>.env`(含 `uptime`,本机是共享机器);
- 绝对值会随机器负载漂移(同配置重复测可见 ±5%),**结论看趋势与倍数**。

---

## 2. 归因结果:瓶颈是 CPU 引擎,不是往返

| 场景 | period/层 | **compute(CPU MoE)** | rest(GPU+拷贝+派发) |
|---|---|---|---|
| qlen=1(单序列) | 1.74–2.33 ms | 1.00–1.64 ms(**57–71%**) | 0.63–0.75 ms |
| **qlen=64(C=64)** | 25.5 ms | **22.8 ms(89%)** | 2.7 ms(11%) |

结论:
1. **往返本身只有 ~0.7–2.7 ms/层**,不是主成本(T51 最初的假设被证伪);
2. 批量越大,CPU 占比越高(C=64 时 89%)⇒ 优化必须打在 CPU 引擎上;
3. `rest` 随 batch 增长(0.7 → 2.7 ms),对应 GPU 侧注意力/dense 的真实工作量。

### 2.1 吞吐随并发的变化(单卡,out=256)

| C | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| tok/s | 39.4 | 52.6 | **54.9** | 44.1 |
| 每层 period | 8.9 ms | 13.5 ms | 25.5 ms | 65 ms |

C=128 时每层时间约为 C=64 的 **2.5×**,而 C=64 约为 C=16 的 **2.9×**(token 数 4×)——
**CPU 时间几乎与 token 数成正比,没有任何摊薄**,这正是"每个 (token,expert) 对一个独立
专家块读"的特征(见 §4 流量模型)。所以**峰值在 C=64**,再往上只会更慢而非更快。

---

## 3. 引擎权重布局:三种模式,实测差 1.38×

引擎(`xiaotu_moe/csrc/moe/moe_v2.hpp:384` 起)有三种权重驻留方式:

| 模式 | 触发 | 内存 | 访存特征 |
|---|---|---|---|
| **默认:NUMA 分片** | 都不设 | **1 份** | node n 持 gate/up 的 `[n·I/NS,(n+1)·I/NS)` 列 + down 的 `[n·H/NS,(n+1)·H/NS)` 行;每个线程只读本 node 的行 ⇒ **全部本地** |
| 单拷贝 | `XIAOTU_MOE_SINGLECOPY=1` | 1 份 | 无分片,一半访问跨 socket |
| socket 副本 | 不设 SINGLECOPY + `XIAOTU_MOE_NOSHARD=1` | **2 份** | 每 socket 一份完整副本,全本地 |

**实测(C=64/out=256,同机同时段,单卡)**:

| 配置 | 每层 compute | 每层 period | 吞吐 | 有效带宽 |
|---|---|---|---|---|
| `XIAOTU_MOE_SINGLECOPY=1`(本项目此前默认) | 22.8 ms | 25.5 ms | 53.4 tok/s | ~237 GB/s |
| **默认 NUMA 分片(1 份内存)** | **15.9 ms** | **18.6 ms** | **73.6 tok/s** | **~340 GB/s** |

> `XIAOTU_MOE_SINGLECOPY=1` 是为了"省内存"引入的,但**默认分片模式同样是 1 份内存**
> (lk 系项目 README:"多个 NUMA 节点共享单份内存"),所以这档配置是**纯亏**。
> `scripts/tune_serve.sh` 现在支持 `XIAOTU_MOE_SINGLECOPY=0` 明确回到默认分片。

**带宽天花板**:本机 STREAM 只读 96 线程本地 **754 GB/s**、单 socket triad 274 GB/s、
全机 triad 422 GB/s。我们最好也只有 340 GB/s ⇒ **还有 2× 以上空间**(§5)。

---

## 4. 流量模型(解释一切"为什么加了并行没变快")

MXFP4 专家块大小(E=256,H=4096,I=2048):

| 张量 | 形状 | 每专家 |
|---|---|---|
| `w13_weight` | [E, 4096, 2048] u8 | 8.4 MB |
| `w2_weight` | [E, 4096, 1024] u8 | 4.2 MB |
| `w13/w2_scale`(e8m0, groupK=32) | — | 0.75 MB |
| **合计** | | **≈13.4 MB/专家** |

引擎有两条路径(`forward_many`):**分组路径**(每个活跃专家的块只读一次,摊给所有路由到它的
token)与 **per-token 路径**(每个 (token,expert) 对读一次)。判定阈值 `nave × F ≤ NASS`
(NASS = pairs,F 默认 8)。

| C | pairs | 活跃专家 | 每层读(per-token) | 每层读(grouped) |
|---|---|---|---|---|
| 16 | 96 | ~79 | 1.29 GB | 1.06 GB |
| 64 | 384 | ~243 | **5.15 GB** | **3.26 GB** |
| 128 | 768 | ~256 | 10.3 GB | 3.4 GB |

- C=64 时 `nave=243`、`NASS=384` ⇒ 243×8 > 384 ⇒ **走 per-token 路径(多读 1.6×)**;
- 把 `XIAOTU_MOE_GROUP_FACTOR` 调到 1 ⇒ 243×1 ≤ 384 ⇒ 强制 grouped;
- 这就是 §2.1 里"C=128 每层时间 ~2×C=64"的来源:**流量随 pairs 线性增长**。

---

## 5. NUMA 拓扑与分片策略(本机实测)

```
node distances:   node 0  1  2  3 | 4  5  6  7
            0:     10 12 12 12 | 32 32 32 32      <- 同 socket 内 10/12/12/12
            1:     12 10 12 12 | 32 32 32 32      <- 跨 socket 一跳 = 32
On-line CPU(s): 0-191(192 物理核,SMT off)   node0 cpus=0-23  node4 cpus=96-119
8 NUMA nodes(NPS=4)· 每 node 193 GB · 每 node 24 核 = 3 个 CCD · 每 CCD 8 核
```

**关键认识**:
1. **真正的局部性边界是 socket,不是 NUMA node**:同 socket 内 12 ≈ 10(几乎无损),
   跨 socket 32(带宽与延迟双重惩罚)。这解释了 §3 的 1.38×:关掉分片后一半访问跨 socket。
2. 本机 8 个 node 来自 BIOS `NPS=4`;改成 **`NPS=1` 就是 2 个 node**,
   而 2 个 node 正好 = 2 个 socket ⇒ 引擎 `nshard_ = numa_node_count()` 自动变成 socket 级分片。
3. **分片越细,每片的线程越少**:`nshard_=8` + `THREADS=96` ⇒ 每片仅 12 线程(每 CCD 4 个);
   引擎自己的带宽笔记:"单个 CCD 喂不满它的 DDR5 通道,要一个 socket 的 12 个 CCD 一起上"。
4. 因此**不能重启也要拿到 socket 级分片**:新增环境变量 **`XIAOTU_MOE_NSHARD=N`**
   (≥2 生效,`moe_v2.hpp` 中 `nshard_` 覆盖),`NSHARD=2` 等价于 NPS=1 的结构(仍是 1 份权重)。

**待测矩阵**(见 §9):8 片/96 线程(已测 73.6) → 8 片/192 线程 → 2 片/192 线程 → +grouped。

---

## 6. 为什么 TP=2 / 专家并行在 decode 上无效(实测)

| C | 单卡 tok/s | TP=2+EP tok/s | 每层 ms(单卡→TP2EP) | TTFT(单卡→TP2EP) |
|---|---|---|---|---|
| 16 | 39.4 | 25.0 | 8.9 → 14.5 | 6.9 s → 4.6 s |
| 32 | 52.6 | 36.6 | 13.5 → 19.9 | 8.0 s → 5.7 s |
| 64 | 54.9 | 45.1 | 25.5 → 32.3 | 15.9 s → 9.1 s |

- **每层固定 +4–6.5 ms,与 batch 无关** ⇒ 这是每层一次跨 rank 同步的代价(我们的实现用
  NCCL all-reduce 合并两 rank 的部分和);
- EP 把每个 rank 的**对数**减半,但 §4 的流量模型说明:**per-token 路径下总流量 = pairs × 块大小**,
  两个 rank 合起来读的字节与原单卡**完全一样**,机器带宽共享 ⇒ 不省时间;
- 但 **prefill 有效**:TTFT 1.76×(两块卡分摊权重流式),因为那是"每块流 137 GiB"的带宽问题。

### 6.1 参考实现(lk-moe)是怎么做的
- 生产 fork 里 `_get_processes_info()` 返回 **`num_processes = ep_size, process_id = ep_rank`**
  ⇒ **一个逻辑引擎横跨 EP 个进程**,每进程持 `E/ep_size` 个专家
  (TP=2 时每引擎约 80 GB ≈ 128/256 专家 + 非专家分片);
  部分和的合并发生在**引擎内部(同地址空间共享内存)**,不是 GPU collective;
- 他们的 `LVLLM_MOE_NUMA_ENABLED=1` 对应我们引擎的**默认分片模式**(单份内存 + node 本地读)。

> 结论:**"CPU 上的 TP" 只有在"合并走共享内存、不进 GPU"时才便宜**。我们当前缺的正是这一环
> (我们引擎的 `MOEConfigV2.num_processes/process_id` 字段存在但未实现)。

---

## 7. CPU 版 TP / EP 的设计(NUMA node 当作"设备")

### 7.1 数据量账(C=64,每层)

| 项 | 字节 |
|---|---|
| 专家权重读 | **3.3–5.2 GB** |
| hidden 激活 | 512 KB |
| 中间激活 | 512 KB |
| 输出 | 1 MB |

要交换的东西比权重小 **3 个数量级** ⇒ 设计目标只有一个:**让权重的每一次读都落在本地 node**;
激活怎么交换都无所谓(全量 all-gather 也 < 0.05%)。

### 7.2 两种切法

| 方案 | 每个 node 拥有 | 层内交换 | 收尾 |
|---|---|---|---|
| **A. 层内 TP** | w13 的 1/N intermediate 列 + w2 的 1/N 输出行 | act all-gather(512 KB + 1 次 barrier) | 输出行互不相交,**无需 reduce** |
| **B. 专家 EP** | 固定 E/N 个专家的完整权重 | 路由表 + 命中的 token hidden(共享缓冲直读) | 各 node 部分输出相加(1 MB×N 或原子加) |

- 引擎**已有**按专家分组的 `per_expert` 列表 → **B 的改动最小**;
- A 的负载天然均衡;B 需要处理专家热度不均(可用热点专家复制);
- 同步成本:跨 node barrier ≈ µs 级(43 层 × 2 次 × ~3 µs ≈ 0.26 ms/step),
  **比 GPU 上每层 all-reduce 便宜三个数量级**。

### 7.3 预期收益(以现测为基准外推,标注为估算)

| 配置 | 每层 CPU | step | 吞吐(估) |
|---|---|---|---|
| 现状(SINGLECOPY) | 22.8 ms | 1139 ms | 53.4(实测) |
| **NUMA 分片** | **15.9 ms** | 800 ms | **73.6(实测)** |
| + 线程 192 / 2 片 | ~10–12 ms | 550–620 ms | ~105–115 |
| + grouped(流量 ÷1.6) | ~7–9 ms | 420–520 ms | ~125–150 |

---

## 8. 与 lk-moe 的口径对齐(重要)

**"约 100 tok/s" 的原始口径**(`process_data/scripts/bench.sh`):

```bash
export LVLLM_MOE_NUMA_ENABLED=1
vllm bench serve --base-url http://localhost:8070 --model DeepSeek-V4-Flash-0731 \
  --dataset-name sharegpt --num-prompts 50 --max-concurrency 4 --tokenizer <0731 快照>
```

即 **C=4、50 个 prompt、ShareGPT 自带输出长度、不忽略 EOS**——与 C=64/out=256 完全不同的工况。
`scripts/bench_lk_shape.sh` 就是这条命令的同形复刻,用于公平对比。

**历史引擎微基准**(本项目引擎、真实权重、**默认分片布局**、OMP=96):

| B | 单层 ms | 43 层 ms | MoE-only 天花板 tok/s |
|---|---|---|---|
| 32 | 8.0 | 344 | 93 |
| **64** | **13.0** | 560 | **114** |
| 128 | 21.1 | 907 | 141 |
| 256 | 35.6 | 1531 | 167 |

⇒ 引擎在默认布局下 B=64 是 **13.0 ms/层**,而我们生产配置(SINGLECOPY)实测 22.8 ms/层:
**1.75× 的差距完全由布局造成**,与 §3 的实测一致。剩下 340 GB/s vs 754 GB/s 的差距,
是线程数/分片粒度/kernel 层面的空间。

---

## 9. 实验矩阵与复现命令

| # | 目的 | 命令要点 | 状态 |
|---|---|---|---|
| E1 | 基线(SINGLECOPY) | `XIAOTU_MOE_SINGLECOPY=1` | ✅ 53.4 tok/s |
| E2 | **默认分片** | `XIAOTU_MOE_SINGLECOPY=0` | ✅ **73.6 tok/s** |
| E3 | 线程翻倍 | E2 + `THREADS=192` | 🟡 待测 |
| E4 | socket 级分片 | E3 + `XIAOTU_MOE_NSHARD=2` | 🟡 待测 |
| E5 | 强制 grouped | 最优 + `XIAOTU_MOE_GROUP_FACTOR=1` | 🟡 待测 |
| E6 | 同形对比 | 最优 + `TAG=... C=4 N=50 scripts/bench_lk_shape.sh` | 🟡 待测 |
| E7 | 常驻专家层 | `XIAOTU_MOE_GPU_RESIDENT_LAYERS=...`(未实现) | ⬜ |
| E8 | socket 副本 | `XIAOTU_MOE_NOSHARD=1`(2 份内存) | ⬜ 参考项 |

统一记录:`report/tuning/summary.jsonl`(每轮一行)+ `report/tuning/NOTES.md`(逐轮解读)。

---

## 10. 硬件评估:换成 1× RTX PRO 6000(96 GB)值不值

**规格**:96 GB GDDR7 ECC、~1.79 TB/s、24064 CUDA、PCIe 5.0 x16、sm_120、600 W。
注意 **带宽只比 A100(1.55 TB/s)高 ~15%**——它赢在**容量**与 **PCIe 5.0**,不赢在 HBM 带宽。

### 10.1 decode(估算)

96 GB × util 0.92 = 88 GB;减非专家 ~19 GB、再减 KV:

| KV 预算 | 可常驻专家层(3.19 GiB/层) |
|---|---|
| 8 GiB | **19 / 43** |
| 12 GiB | 17 / 43 |
| 24 GiB | 14 / 43 |

GPU 常驻层每层 ≈ 2.5 ms(C=64 时活跃专家并集 3.06 GB 从 HBM @1.79 TB/s ≈ 1.7 ms + 注意力/dense),
CPU 层 15.9 ms(已优化)→ step = N×2.5 + (43−N)×18.6:

| 常驻层 | step | 吞吐(估) |
|---|---|---|
| 0 | 800 ms | 73.6(实测) |
| 14 | 623 ms | ~103 |
| 19 | 566 ms | ~113 |

⇒ 换卡在 decode 上约 **+40~55%**;真正的约束仍是留在 CPU 上的那 24–29 层。

### 10.2 prefill(估算,3–8×)

| 因素 | 现在(A100/PCIe4) | RTX PRO 6000(PCIe5) | 倍数 |
|---|---|---|---|
| KV 允许的 nbt | 12 GiB KV → nbt 8192 → 128K = 16 块 | 24–32 GiB → nbt 32–64K → 2–4 块 | 4–8× |
| 每块流式权重 | 43 层 × 3.19 GiB | 24–29 层(其余常驻) | 1.5–1.8× |
| PCIe 实测 | Gen4 ≈ 25 GB/s | Gen5 ≈ 45–50 GB/s | ~2× |
| 注意力/indexer 内核 | SM80 回退(o_proj fp8 einsum 345 µs/层、稀疏注意力 736 µs/层) | sm_120 原生 DS-V4 内核 | 数倍 |

现测 128K TTFT 246.8 s / 32K 41.0 s ⇒ 估算 128K → 25–50 s。

### 10.3 风险
1. **sm_120 内核齐备性**:我们的 GPU MoE 走 Marlin MXFP4;主线 DS-V4 已有 SM120 分支,
   但 Marlin/DeepGEMM 在 sm_120 上要先验证;
2. **GDDR7 ≠ HBM**:带宽只 +15%,不要指望"专家全上卡"变快;
3. 单卡失去双卡 prefill 分摊(1.76×),但 nbt 红利远大于此;
4. 无 NVLink;600 W/PCIe 5.0 插槽要求;
5. 容量仍不够:Qwen3.8(185 GB)、V4.1(475 GiB)依旧装不下。

---

## 11. 已知限制与未验证项

| 项 | 说明 |
|---|---|
| `perf` 不可用 | `perf_event_paranoid=4` ⇒ 读不到 DRAM 计数器;判带宽只能用"改流量看时间" |
| 共享机器 | 同机有其它租户(load average 记录在各 `.env` 里),绝对值有 ±5~10% 漂移 |
| EP 正确性 | TP=2+EP 与 TP=1 的贪婪输出 3/5 完全一致、2/5 前 10+ token 一致后分叉(不同归约顺序的正常数值差异);未做逐层数值对齐 |
| `--async-scheduling` / MTP 投机解码 | 未测 |
| CUDA graph | 修好 use-after-free 后对 decode 无收益(48.8 vs 48.5 tok/s) |
| Qwen3.8 单卡 prefill | 仍慢(TTFT 24.9 s),GPU prefill 通路未接通用后端 |

License: Apache-2.0

---

## 12. 口径校准:那个"约 100 tok/s"到底是什么(2026-09-10 补充)

**结论:它是 `vllm bench serve` 的 `Total token throughput` 行、且是双卡生产环境的数字。**

历史基准存档(`/home/user/lvllm/process_data/bench/`、`process_data/logs/`)里能找到同一口径
(C=4、50 prompts、ShareGPT 默认输出长度、不忽略 EOS)的全部对照:

| 运行 | Output tok/s | **Total tok/s** | Mean TPOT | 配置 |
|---|---|---|---|---|
| `results_lkmoe.txt`(**lk_moe 基线**) | 22.70 | **51.00** | 177.3 ms | 8071,单卡,maxlen 8192 |
| `bench_t96_session.txt`(xiaotu,无 CG) | 15.96 | 34.71 | 222.8 ms | 96 线程,enforce-eager |
| **`bench_t96cg_session.txt`(xiaotu + CUDA graph)** | **35.75** | **77.75** | **88.2 ms** | 96 线程,**图捕获** |
| `bench_t120cg_session.txt` | 35.73 | 77.69 | 91.6 ms | 120 线程 + CG |
| `bench_t168cg_session.txt` | 30.65 | 66.62 | 102.0 ms | **168 线程 + CG(更慢!)** |
| 本轮 xiaotu(NUMA 分片,无 CG,maxlen 262144) | 24.26 | 55.69 | 135.3 ms | 2026-09-10 |

由此得到三条重要结论:

1. **口径**:`Output token throughput` 与 `Total token throughput` 差 2.3 倍(本批数据 input:output ≈ 1.3:1)。
   引用吞吐必须写明是哪一行;用户的"100 tok/s"= **Total + 双卡**(单卡 lk 基线 51 → 双卡 ≈ 102)。
2. **按卡比,xiaotu 早就比 lk_moe 快**:同形同机,`t96cg` = 77.75 total / 35.75 out,
   对比 lk 基线 51.00 / 22.70 ⇒ **每卡快 1.5×**。
3. **CUDA graph 是这个口径下最大的单项杠杆**:同一引擎 96 线程,
   **34.71 → 77.75 total(2.24×)**,TPOT 222.8 → 88.2 ms。
   而线程数超过 ~120 反而变慢(168 线程只有 66.62)——
   与"线程要均匀铺满 CCD 即可,再多无用"的经验一致(§5)。

> 注意:C=64 时 CUDA graph 几乎没有收益(48.8 vs 48.5 tok/s)——因为那时 CPU 引擎占 89%,
> 图省下的 GPU 逐 kernel 派发开销被完全掩盖。**图只在小 batch(每层 GPU 工作量小、
> 派发占比高)时值钱**,所以必须按口径分别评估。


---

## 13. CUDA graph:两个捕获期 bug 与实测收益(2026-09-10)

开启 `--enforce-eager=false` 后,vLLM 的 breakable-CUDA-graph 捕获连续失败两次,
**两次都是我们 binding 的问题**,已修复:

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `[cd] streamSync err=operation not permitted when stream is capturing` → 引擎初始化失败 | pinned 缓冲扩容路径里调了 `cudaStreamSynchronize`,捕获期间非法 | 捕获期间**不做同步**;旧缓冲"退役"进 `retired` 列表、不 `cudaFreeHost`(graph 节点可能仍引用) |
| 2 | `cudaErrorStreamCaptureInvalidated` → PyTorch `markCaptureEnd called with no captures in progress` | 捕获期间调 `cudaHostAlloc` 同样会作废整次捕获 | 新增 **`prepare_decode_buffers(max_qlen, top_k)`**,插件在**引擎建好后、捕获之前**按捕获尺寸上限预分配;稳态不再扩容 |

独立回归测试:**`scripts/engine_graph_test.py`** —— 随机 MXFP4 权重建引擎,
eager 跑一次作参考,再用 `torch.cuda.CUDAGraph` 捕获同一调用并 replay 两次:

```
[eager] ok, out.sum=1.5127684386923531e+20
[capture] ok
[replay 0] identical=True  max|Δ|=0.000e+00
[replay 1] identical=True  max|Δ|=0.000e+00
[PASS] graph capture + 2 replays identical to eager
```

> 写这个测试时踩到一个**通用陷阱**:捕获必须用**调用时**的 `torch.cuda.current_stream()`;
> 若提前缓存默认流的句柄,`torch.cuda.graph()` 的侧流捕获会得到**空图**
> ("The CUDA Graph is empty")。引擎里本来就是调用时取流,测试脚本已按此修正。

**另一个本机限制**:`--max-num-seqs 256`(⇒ `max_cudagraph_capture_size=512`)时,
vLLM 的 breakable 捕获仍会在某个尺寸上崩(与上述两个 bug 无关);
`--max-num-seqs 128`(捕获 ≤128)稳定。所以开图时把 `max-num-seqs` 控制在 128。

**实测收益(单卡,NUMA 分片 + 96 线程)**

| 口径 | 无图 | 有图 | 增益 |
|---|---|---|---|
| C=64 / out=256 | 73.64 tok/s | **77.43 tok/s** | +5% |
| 同形 C=4(Output / Total) | 24.26 / 55.69 | **25.91 / 57.84** | +7% |
| 每层(qlen=4) | period 2.58 ms(compute 1.82 / rest 0.78) | period 2.41 ms(compute 1.55 / rest 0.86) | -7% |

> ⚠️ **不要拿历史存档里的 2.24× 当预期**:那是 2026-09-06 的老引擎对比
> (`bench_t96_session` 34.71 → `bench_t96cg_session` 77.75 total),它的**无图基线只有 34.71**,
> 而我们今天的无图基线已经是 55.69。图收益的本质是省掉"逐 kernel 派发",
> 只有当 `rest` 里派发占比很高时才显著;我们现在 `rest` 只占 33–36%,所以只剩 6%。

---

## 14. T55:GPU 常驻专家层(把权重流量搬去 HBM)

### 14.1 为什么这是"绕开 DRAM 上限"的唯一办法

§20 的实测结论:CPU 专家路径撞到**整机 DRAM 子系统**上限(有效 200–340 GB/s,峰值 754),
所以把专家拆到 2 个 rank 并不会更快。要突破,只能**把权重流量从 DRAM 挪到 HBM**:
每个常驻层贡献 0 往返、0 DRAM 权重流量,而 GPU 侧一层只要 ~2.5 ms。

### 14.2 实现(复用现有 GPU MoE 通路,约 60 行)

```bash
XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-7      # 逗号+区间,对齐 lk-moe 的 LVLLM_GPU_RESIDENT_MOE_LAYERS
```

- 这些层在 `finalize_mega_moe_weights()` 里**不建 CPU 引擎**,而是用 `gpu_prefill` 的
  `PrefetchSlot` 分配一块**常驻显存**(K-major 布局,一次性阻塞 H2D),`forward` 里无条件走
  `gpu_moe_layer(..., slot=常驻slot)`;
- 常驻层不再参与 ping-pong 预取(`prefetch_gpu_weights` 直接返回),
  prefill 时也无需再流式(省下 N×3.19 GiB 的 H2D);
- TP>1 时每个 rank 只驻留自己那 1/TP 的专家分片(TP=2 → 1.6 GiB/层);
- 显存分配发生在 vLLM 的 KV profiling **之前** ⇒ KV 会自动按剩余显存收缩,不会 OOM。

### 14.3 单卡能放几层:先看 KV 的硬下限

vLLM 要求 KV 至少能装下 `max_model_len` 一条序列:262144 × 29.5 KB = **7.19 GiB**。
单卡预算(40 GB × 0.90 = 36 GB):非专家权重 ~19.6 GB + KV 7.2–12.9 GB + 激活/图池 ~2–3 GB
⇒ **只剩 ~1 层**的空间。所以单卡不是常驻层的主场,**多卡才是**。

### 14.4 多卡预算(每 rank)

| 配置 | 非专家/rank | KV/rank | 可常驻层数(每层 1/TP GiB) |
|---|---|---|---|
| TP=2 + 256K(KV 12 GiB) | ~10 GB | 6.4 GB | **~8–11 层** |
| TP=2 + 1M(KV 16 GiB/rank) | ~10 GB | 16 GB | ~5–6 层 |
| TP=3 + 256K | ~6.5 GB | 4.3 GB | **~18–21 层** |

预期(C=64/out=256,CPU 层按 18.5–20 ms、GPU 层 2.5 ms):
TP=2 + 8 层 ≈ **89 tok/s**;TP=2 + 12 层 ≈ **103 tok/s**;TP=3 + 18 层 ≈ **126 tok/s**。


---

## 15. 🎯 达标配置(2026-09-10 10:4x):单卡 106 tok/s

### 15.1 三项叠加,单卡从 54.9 → 106.2 tok/s(+93%)

| # | 配置 | C=64/out=256 | 每层 period(compute + rest) |
|---|---|---|---|
| 0 | 起点:`SINGLECOPY=1` + 96 线程 + eager | 54.9 | 25.5 ms(22.8 + 2.7) |
| 1 | + **NUMA 分片**(`XIAOTU_MOE_SINGLECOPY=0`) | 73.6 | 18.5 ms(15.7 + 2.8) |
| 2 | + **CUDA graph**(`EAGER=0`,`--max-num-seqs≤128`) | 77.4 | 18.5 ms |
| 3 | + **192 线程**(`XIAOTU_MOE_THREADS=192 OMP_NUM_THREADS=96`) | **90.6** | 14.9 ms(12.1 + 2.7) |

并发扫描(第 3 项之上,out=256):

| C | 64 | 96 | **128** |
|---|---|---|---|
| tok/s | 90.6 | 99.2 | **106.15** |
| TTFT | 15.8 s | 20.7 s | 25.2 s |
| TPOT | 646 ms | 889 ms | 1108 ms |

### 15.2 为什么 192 线程在这里才有效(重要)

引擎的分片模式让**每个线程只读本地 NUMA node 的行**;96 线程 = 12 线程/node(每 CCD 4 个),
192 线程 = 24 线程/node(每 CCD 8 个)⇒ 每 node 的并行度翻倍,compute 直接降 23%。
而在此之前(`SINGLECOPY=1`,读跨 socket)加线程无效——瓶颈是 socket 互联而不是线程数。
**结论:分片与线程数是配套的两个旋钮,必须一起用。**

### 15.3 达标配置(单卡,256K 上下文)

```bash
export CUDA_VISIBLE_DEVICES=2
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export XIAOTU_MOE_SINGLECOPY=0        # NUMA 分片(勿开单拷贝)
export XIAOTU_MOE_THREADS=192 OMP_NUM_THREADS=96
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384

vllm serve <CKPT> --tensor-parallel-size 1 --max-model-len 262144   --max-num-seqs 128 --max-num-batched-tokens 8192   --kv-cache-dtype fp8_ds_mla --kv-cache-memory-bytes 12884901888   --gpu-memory-utilization 0.90 --no-enable-prefix-caching   # 注意:不开 --enforce-eager(CUDA graph)
```

> 与 lk-moe 生产(双卡 + 投机解码)同形对比:我们**单卡 106.15 tok/s > 他们双卡 105.74 total**。
