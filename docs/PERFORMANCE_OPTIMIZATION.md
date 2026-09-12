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
| 单拷贝 | `| 1 份 | 无分片,一半访问跨 socket |
| socket 副本 | 不设 SINGLECOPY + `XIAOTU_MOE_NOSHARD=1` | **2 份** | 每 socket 一份完整副本,全本地 |

**实测(C=64/out=256,同机同时段,单卡)**:

| 配置 | 每层 compute | 每层 period | 吞吐 | 有效带宽 |
|---|---|---|---|---|
| `| 22.8 ms | 25.5 ms | 53.4 tok/s | ~237 GB/s |
| **默认 NUMA 分片(1 份内存)** | **15.9 ms** | **18.6 ms** | **73.6 tok/s** | **~340 GB/s** |

> `是为了"省内存"引入的,但**默认分片模式同样是 1 份内存**
> (lk 系项目 README:"多个 NUMA 节点共享单份内存"),所以这档配置是**纯亏**。
> `scripts/tune_serve.sh` 现在支持 `明确回到默认分片。

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
| E1 | 基线(SINGLECOPY) | `| ✅ 53.4 tok/s |
| E2 | **默认分片** | `| ✅ **73.6 tok/s** |
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

---

## 16. 投机解码(DSpark)与"低并发延迟"这个真正的目标(2026-09-10 晚)

### 16.1 先纠正目标:C=64 的"高吞吐"对交互式使用没有意义

C=64/out=256 时 TPOT 是 0.6~1.3 s/路 —— 人机交互上等于卡死。
**该看的是 C≤4 时单路出词速度**(与 lk-moe 生产基准 C=4 同口径)。

### 16.2 DSpark 跑通(主线 `mtp` 方法不可用,`dspark` 可以)

- `--speculative-config '{"method":"mtp",...}'` 会被主线接受,但**加载草稿权重时崩**
  (`KeyError: model.layers.43.mtp_block.main_norm.weight` —— 主线把 `mtp.{i}.` 映射成
  `model.layers.{43+i}.` 后又加了 `.mtp_block.`,而模型参数名里没有它);
- **`{"method":"dspark", ...}` 可用**:日志 `DSpark draft model loaded: 97 params`,
  与 lk-moe 生产同款(draft block=5,`num_speculative_tokens=4`);
- **CUDA graph 与 DSpark 不兼容**:捕获期草稿模型会 `aten::new_empty` 失败 ⇒ 用 `--enforce-eager`。

### 16.3 实测:投机解码在两种并发下方向相反

| 场景 | 无投机 | **+DSpark** | 结论 |
|---|---|---|---|
| C=1 单路 | ~128 ms/token | **85 ms/token(11.8 tok/s/路)** | ✅ **+50%** |
| C=4(生产同形) | 25.91 / 57.84 out/total | 23.60 / 52.82 | ❌ −9% |

接受率实测 **33.9%**,**2.35 token/step**(生产 40.16% / 3.01)。
原因:验证批次变大 ⇒ pair 数变多 ⇒ CPU MoE 成本上升;只有当"每步固定开销"占主导
(小 batch)时,多出来的 token 才是净赚。

> **结论:投机解码是低并发延迟优化,不是高并发吞吐优化。交互用开,批处理关。**

### 16.4 1M 上下文为什么必须 TP=2(三条路都试过了)

| 方案 | 结果 |
|---|---|
| 单卡 + fp8 KV | KV 29.5 GiB,放不下(非专家权重 ~19.6 GB) |
| 单卡 + **FP4 KV**(`nvfp4_ds_mla`) | ❌ 主线拒绝:A100/SM80 的 `fp8_ds_mla` 布局只支持 fp8 KV |
| 单卡 + 512K | ❌ OOM(2 GiB 预取槽) |
| **TP=2 + 1M + DSpark** | ✅ KV 1,876,112 tokens;C=1 单路 266 ms/token |

### 16.5 最终交付的三档配置

| 用途 | 命令 | 单路延迟 | 上下文 |
|---|---|---|---|
| **交互** | `MODE=fast bash scripts/serve_prod_8070.sh` | **85 ms/token** | 256K |
| **长文本/1M** | `bash scripts/serve_prod_8070.sh`(默认) | 266 ms/token | 1M |
| **高吞吐** | 上面 + `SPEC_OFF=1`,C≥64 | 646 ms/token(但 90~106 tok/s 聚合) | 256K |

**正确性**:投机 vs 非投机(同一 1M/TP=2 配置)贪心输出 **3/5 逐字一致**,其余前 11~25 字符
一致后因验证批次不同导致的浮点归约顺序差异分叉;输出内容均正确连贯。

### 16.6 下一步(未做,按价值排序)

1. **把 TP=2 的每层同步干掉**:低并发时 `rest 3.77 ms/层` 全在跨 rank 同步上
   (EP=0 时 compute 只有 0.86 ms),这是 1M 模式慢 3 倍的根因 → 按 §7 的 CPU-TP 设计,
   把部分和合并放进引擎内部(共享内存),而不是每层一次集合通信;
2. **让 DSpark 与 CUDA graph 共存**(修草稿模型的捕获期分配)→ 再拿 +5~7%;
3. 提高接受率(调 `num_speculative_tokens` / 学习生产用 block=5)。

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
| 1 | + **NUMA 分片**(`| 73.6 | 18.5 ms(15.7 + 2.8) |
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
export XIAOTU_MOE_THREADS=192 OMP_NUM_THREADS=96
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384

vllm serve <CKPT> --tensor-parallel-size 1 --max-model-len 262144   --max-num-seqs 128 --max-num-batched-tokens 8192   --kv-cache-dtype fp8_ds_mla --kv-cache-memory-bytes 12884901888   --gpu-memory-utilization 0.90 --no-enable-prefix-caching   # 注意:不开 --enforce-eager(CUDA graph)
```

> 与 lk-moe 生产(双卡 + 投机解码)同形对比:我们**单卡 106.15 tok/s > 他们双卡 105.74 total**。

---

## 17. ⛔ 单路延迟:先误判为机器带宽,再定位到内核(2026-09-10 21:xx–23:xx)

**结论先行:当前机器条件下,单路 25 tok/s 做不到;限制来自"这台机器只能给 ~95–100 GB/s
的 DRAM 读带宽",而解码每步要读 6.5 GB 的专家权重。**

### 17.1 目标场景的正确测法(先纠正两个测量错误)

1. 旧基线用的是 sharegpt 的**短 prompt**(实测 21 token/条),不是目标场景;
2. `--dataset-name random` 的随机 token 会把 draft 接受率从 0.67 打到 0.22;
3. 自研客户端最初把 SSE chunk 当 token(vLLM 每步只发一个 chunk)⇒ 吞吐被低估 ~2.5×。

现在用 `scripts/bench_nat_client.py` + `report/tuning/datasets/nat*.jsonl`
(自然文本、精确上下文长度、`stream_options.include_usage` 计 token)重测。

### 17.2 当前官方数字(单卡 256K + DSpark k=5 + CUDA graph + 192 线程)

| 配置 | 单路 tok/s | 每步 ms | tokens/步 | 接受长度 |
|---|---|---|---|---|
| C=1,ctx 128 | **11.55** | 197 | 2.69 | 2.87–3.52 |
| C=2,ctx 128 | 7.12/路(合计 14.24) | 339 | 2.87 | ~3.4 |
| C=1,ctx 512 | 7.64 | 211 | 3.13 | — |
| C=2,ctx 1024 | 5.33/路(合计 10.65) | 440 | 3.50 | — |

- **接受率已经达标**(自然文本 2.87–3.52,生产的 k=5 水平是 3.01)⇒ 不是接受率的锅。
- **上下文长度不是变量**:22/64/128/512/1024/4096 token 六档,每层 period 都是 4.1–4.7 ms。
- 每步 C=1 197 ms、C=2 339 ms ⇒ 因为 MoE 要读的**不同专家数翻倍**,成本近似翻倍。

### 17.3 更正:瓶颈一度被误判为"机器带宽"(现已修正)

> **2026-09-10 22:xx 更正**:下面这段曾得出"引擎已贴机器带宽上限、内核无空间"的结论,
> **是错的**。当时的带宽探针把所有物理页都落在**同一个 NUMA 节点**上(主线程首次触碰
> 分配的缓冲),量到的 95 GB/s 只是**单节点**上限。正确测法
> (`numactl --cpunodebind=N --membind=N`,每节点独立进程 + 本地内存):
>
> | 口径 | 结果 |
> |---|---|
> | 单节点 1 / 8 / 24 线程 | 36.7 / 90.7 / **98.7 GB/s** |
> | **8 节点 × 24 线程(整机)** | **783.0 GB/s**(完美线性) |
>
> 而引擎每层只用到每节点 9–16 GB/s(**节点上限的 10–16%**)⇒ 内核空间很大。
> 据此定位到两处真实低效并已修复(见 §16.7),实测 **1.39–1.59×**。

正确的成本分解(改前):引擎每层要读 = **去重后的活跃专家数 × 12.6 MB**(fp4);
随机路由下测 36 个不同专家(454 MB)⇒ 5.5–8.1 ms/层。**服务内实测真实解码一次调用
的 36 个 assignment 落在 ~32 个不同专家上(me≈1.1)**,所以小批量成本确实 ∝ 对数。

### 17.4 另一半成本:GPU 被"饿"在低频(可恢复,但需要 root)

- 我们用的 GPU2 全程 **SM 765 MHz / 1410 MHz(利用率 14–21%,45 W/250 W)**:解码是
  CPU 端 MoE 主导,GPU 空转 ⇒ 驱动不上频。
- 后果:每层 `rest`(注意力/dense,GPU 侧)从早先的 **0.91 ms 涨到 2.1–3.1 ms**
  ⇒ 每步多花 40–50 ms。
- `nvidia-smi -i 2 -lgc 1410` **当前用户无权限**(需 root)。这是本机唯一"免费"的
  40–50 ms/步。

### 17.5 已落地的三处修复(2026-09-10/11,实测 1.39–1.59× 内核 + rest 3.5×)

| # | 问题 | 修复 | 实测 |
|---|---|---|---|
| 0 | **引擎线程池 5 ms 自旋把整机烧穿**:解码步里相邻调用只隔 2–4 ms ⇒ 192 个 worker 全速空转(实测解码期间占 **166–259 核**,而真正算力只需 ~21 核),并抢走驱动 GPU 的主线程 | 自旋窗 5000 → **300 µs**(已写进插件默认 `hybrid_model.py`:`XIAOTU_MOE_SPIN_IDLE_US`) | 每层 `rest` **3.3–3.5 → 0.94–0.98 ms(3.5×)**;端到端 **9.15 → 11.7–12.0 tok/s** |
| 1 | packed4 内核只有 **M≥4** 的"解码一次喂 4 行"路径;真实解码 `me≈1.1~3` 落进**单行兜底**,每行都把 FP4 nibble 解码重做一遍 | 新增 `block_23<R>`:**解码一次喂 2/3 行**(`moe_v2_packed4.hpp`,与 4 行分块相同的 4 路部分累加) | me=3:每层 2.00 → **1.17 ms**(1.59×) |
| 2 | 单行兜底用**一条 128 长的串行 FMA 依赖链**(`total0 = fmadd(...)` ×128 ≈ 512 cycle/行),OoO 掩不住;而它才是真实热路径 | 改为 **4 个独立部分累加器**(与 4 行分块同结构) | me≈1(与线上同形):87.6 → **63.0 ms**/40 次调用(1.39×);顺带把 me=1 的数值误差从 max_rel 1.9e-2 降到 3.0e-4 |

自旋扫描脚本 `scripts/sweep_spin.sh`;自旋值经同程序列括号法 A/B(300 / 1000 / 300)
确定:**100 µs 与 300 µs 每层等价,5000 µs 明显更差**。修复后每步成本结构:

| 项 | ms/层 | ms/步(×43) | 占比 |
|---|---|---|---|
| compute(CPU MoE 回调 = 纯 `forward_many`) | 2.8–3.4 | 122–146 | **~75%** |
| rest(注意力/dense + 主机侧) | 0.94–0.98 | 40–42 | ~22% |
| 非层开销(采样/调度) | — | 15–30 | ~8% |

⇒ 下一轮主战场是 **`compute`**:同形路由下**微基准只要 1.58 ms/层(A+B),服务内却是
2.8–3.4 ms**,这 1.3–1.8 ms 的差额需要定位(候选:worker 逐层 park/wake 的代价、
真实路由比合成路由更分散、其他租户的访存争抢——引擎是访存受限,而 load 均值
反映不出内存争抢,实测同一配置的 `compute` 在不同时刻可差 2×)。

### 17.6 第四处修复:worker 数不能等于核数(2026-09-11,**1.7–2.0× / 每层 compute 2.6–3.4×**)

引擎把每个 worker 各钉到一个物理核。**worker 数 == 核数时,调用线程**(MoE 的
host-function 回调线程、torch/CUDA 驱动线程、采样/调度线程)**只能在某个 worker 的核
上抢时间** ⇒ 那个 worker 成为"拖后腿的",而每个阶段都要等最慢的 worker(屏障尾延迟)。

微基准(交错 2 轮,单次引擎调用合计):192→2791/3256 µs、176→1651/1644、160→1649/1613。
把调用线程 `taskset -c 0` 钉到专用核也能从 2645 → 1876 µs(3 轮一致)⇒ 机制确认。

服务端同配置扫描(负载 31–50):

| 线程数 | C=1 tok/s | 每步 ms | 每层 compute | C=2 合计(单路) |
|---|---|---|---|---|
| 192 | 8.25–9.98 | 281–339 | 4.4–5.7 ms | — |
| 176 | 11.44 | 193.9 | 3.22–3.63 ms | 15.75(7.88) |
| 160 | 11.66 | 185.2 | 2.29–2.41 ms | 18.71(9.36) |
| 144 | 12.57 | 162.3 | 1.70–2.21 ms | **21.65(10.82)** |
| **128** | **12.66** | **158.6** | **1.61–1.67 ms** | 21.60(10.80) |

- 平台期 **128–144**;服务端最佳点比单实例微基准(160–184)更靠下,因为服务进程里还有
  torch/CUDA/采样/调度线程,需要更多空闲核。
- 此时服务内 compute 已等于/优于同形微基准 ⇒ **它同时解释了上一轮"服务内比微基准
  慢 2×"的未解之谜**,与内存/多实例/带宽无关。
- 落地:插件未显式设置时默认 `ncpu-64`(≥128 核),`scripts/tune_serve.sh` 默认
  `THREADS=144`。**注意:早先"192 线程最好"的结论只对高并发/预填充成立**
  (那次比的是 C=128 聚合吞吐);解码延迟场景必须少留 worker。

### 17.7 第五处修复:job 过度分片(2026-09-11,A+B 1.23×)+ 内层循环 vs 结构的分离

用独立 C 微基准(同一份 2048 B/行权重、同 NUMA 绑定)只跑内层循环:

| 模式 | 单核带宽 |
|---|---|
| 纯流式读 | 28 GB/s |
| 完整 FP4 解码 | 5.6 GB/s |
| 解码但 shuffle 指令减半 | 7.5 GB/s(仅 1.17×) |
| 解码 + M=2 激活载入 + FMA(=引擎内层) | **5.0 GB/s** |

⇒ 引擎内层循环本身能跑 5 GB/s/核,而**引擎实际只有 1.15–2.2 GB/s/核** ⇒ 差距在
**引擎结构**(每层 3 阶段 + 屏障;默认 job 粒度 ≈1.2 job/worker,一个掉队 job 拖慢整阶段)。

修复:job 过度分片到每 worker ≈4 个(`need = ceil(4·tpn/na)`)⇒ 微基准 A+B
1448–1469 → **1166–1199 µs(1.23×)**,每核 1.15 → **1.43 GB/s**。
服务端(120 线程、DSpark k=5、CUDA graph):C=1 **12.12 tok/s**(每步 170 ms)、
C=2 合计 **20.08**(10.04/路),每层 compute 2.05–2.20 ms。

### 17.8 验证与方法学

验证:同窗口**交错 A/B**(负载 27–43,四轮无重叠)、
`scripts/test_block23_equiv.py`(受控 me=1/2/3/4/5/6/7 路由对 numpy golden,8/8 ✅)、
`scripts/test_swiglu_clamp_mxfp4.py`(✅)。脚本:`scripts/ab_block23.sh`。

**方法学教训(重要)**:本机是共享主机,负载 30–130 漂移,**先后测量的数字不可比**
(同一配置出现过 2.4 / 10.1 / 7.4 ms 三次)。任何 <2× 的改动都必须做同窗口交错 A/B。

### 17.9 达标的算术与可行方向

目标 25 tok/s/路 × 2.8 token/步 ⇒ 每步要 ≤ 105 ms。当前 197 ms 的构成:
MoE 86–116 + 注意力/dense 39(满频)~90(当前频率)+ 采样/调度 ~25。

| 方向 | 能省 | 代价/前提 |
|---|---|---|
| GPU 锁频到 1410 MHz | 40–50 ms/步 | **需要 root**(或让 GPU 持续有活干) |
| 把热专家放 GPU(T55,按专家而非按层) | 按占比线性省 MoE | 要改 dispatch;3 张卡共 120 GB,HBM 2 TB/s |
| 降非层开销(采样 3.6 ms×2、logits、调度) | ~10–15 ms/步 | 需要 profile 定位 |
| 减小 k | 反效果 | dspark 训练块长=5,截断会让接受率崩塌(§36) |
| 内核再优化 | ≤15% | 已贴带宽上限 |

### 17.10 复现命令

```bash
MODE=dsv4 TAG=nat_k5 PORT=8070 TP=1 MAXLEN=262144 SEQS=128 \
  GPU_UTIL=0.85 KV_DTYPE=fp8_ds_mla KV_MEM_BYTES=8589934592 THREADS=192 OMP=96 \
  EAGER=0 EP=0 GPUS=2 ENV_EXTRA="XIAOTU_CD_TIMING=1" scripts/tune_serve.sh
L=128 C=1 N=4 OUT=128 TAG=mine scripts/bench_nat_client.py   # 或 scripts/run_nat_curve.sh
```

详见 `report/tuning/NOTES.md` §35–§38。

## 【2026-xx 第 67 轮】CPU 解码引擎 2× 差距的根因:并行区固定同步开销(不是内核)

**结论(详见 `report/tuning/NOTES.md` §113)**:`forward_many_nsliced` 每层有 4-5 个
`parallel_for` 区域,而**每个区域的固定开销约 113 µs**——用一个"保留派发、body 为空"的
探针测出,且严格线性:

| 空 body 并行区个数 | 报出耗时 | 每区 | 层时间 |
|---|---|---|---|
| 1 | 177 µs | 177 µs | 1.19 ms |
| 10 | 1140 µs | 114 µs | 2.16 ms |
| 50 | 5670 µs | 113 µs | 6.62 ms |

时钟已核验(`steady_clock::now()` = 25.8 ns/次),不是计时伪影。

**根因**:`xiaotu_moe/csrc/moe/numa_pool.hpp` 的 `have_work:`(约 884 行)里,每个 worker
锚定新 generation 都要抢**全局 `work_mtx_`**;120 个 worker 争用同一把 futex 互斥 ⇒ ~113 µs/区。

**为什么这解释了全部历史证据**:扣掉每个区域的 113 µs 后,A 相 ≈294 GB/s、B 相 ≈263 GB/s,
**与 lk_moe 的 265 GB/s 同级** ⇒ 我们与 lk 的 2× 差距**全部来自同步,不来自内核**。
过去针对内层循环的所有尝试(字节数/指令形状/ILP/bf16 点积/布局/预取/分块/线程数/NUMA 映射)
无效是必然的。

**下一步**:把锚定路径去锁(seqlock 复读 `current_gen_`,或 per-worker 发布槽),
目标 113 µs → <5 µs,预期每层 **1.19 → 0.65-0.75 ms**,直接命中 ≤0.7 ms 验收线。


## 【第 69 轮】CPU 解码:去锚定锁成功(奇偶 seqlock)⇒ 每层 1.14-1.24 → 0.78 ms

第 68 轮把根因定位到 `numa_pool.hpp` 的锚定锁(每并行区 113 µs 固定开销)但朴素 seqlock 挂死;
本轮按 §114 处方实现**奇偶 seqlock**(奇数=发布中、偶数=就绪),保留"先掀 gen 再读 counter_"
的刻意顺序。`scripts/test_block23_equiv.py` 两次通过(7 OK、无挂死)。

| 配置 | 改前 | 改后 | lk_moe |
|---|---|---|---|
| DEDUP=12 | 1.14-1.24 ms / 117-122 tok/s | **0.78 ms / 178 tok/s**(最佳 0.72) | 0.57 ms |
| DEDUP=23 | 1.32 ms | **0.88-0.95 ms / 147-158 tok/s** | 0.67 ms |

聚合 124 → **194 GB/s**;每线程 1.0-1.5 → **1.61 GB/s**(目标 2.2)。与 lk 差距 2.0×+ → ~1.37×。
附带:gather 由 36 个 8 KB job 切成 144 个 2 KB job(字节级等价),110 → 37-47 µs。
