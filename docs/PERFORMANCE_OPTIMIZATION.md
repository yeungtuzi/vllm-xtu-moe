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
  (`KeyError: model.layers.43.mtp_block.main_norm.weight`,栈顶在主线**原生**实现
  `vllm/models/deepseek_v4/nvidia/mtp.py:480`)。
  **第 208 轮结案 —— 这不是引擎缺陷,而是这件 checkpoint 里根本没有 MTP 权重**:
  `enorm/hnorm/e_proj/h_proj/shared_head/mtp_block` 在 index.json 的 72317 个 key 里
  出现次数**全为 0**;`mtp.{0,1,2}.*` 是 **DSpark 草稿**(`main_proj/main_norm` +
  3 个 decoder block + `mtp.2` 上的 `norm/hc_head_*/markov_head/confidence_head`)。
  ⇒ **`method:"mtp"` 永久放弃**,不要再尝试改名映射或移植。详见
  `report/tuning/NOTES.md` §255。
- **`{"method":"dspark", ...}` 可用**:日志 `DSpark draft model loaded: 97 params`,
  与 lk-moe 生产同款(draft block=5,`num_speculative_tokens=4`);
  **注意 lk 本机生产脚本 `process_data/scripts/dsv4.sh` 的 dspark 配置里没有 `model` 键** ——
  主线据此自动指向目标 checkpoint 自身(`speculative.py:1129-1136`
  "DSpark can ship the weights inside the target checkpoint"),草稿就是
  `mtp.0/1/2` 三个 block,而不是整模型拷贝。
- **CUDA graph 与 DSpark 不兼容**:捕获期草稿模型会 `aten::new_empty` 失败 ⇒ 用 `--enforce-eager`。
  (这条是**带 `"model": CKPT` 的配置**下的结论;lk 生产的无-`model` 形态能否捕获,第 208 轮起重新验。)

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


## 【第 73 轮·主验收达标】删除 gather:每层 1.22 → **0.66-0.70 ms**(目标 ≤0.70 ✓)

内核的 FP32 激活转换只有一处,把"激活第 i 行"由 `A + i*K` 改成 `A + rowmap[i]*K`(加一个
默认 nullptr 的 `rowmap` 参数,5 处行基址统一走 `arow_at()`),gate/up 就可以直接读按 token
去重的输入缓冲,**不必再 gather 成每专家连续的 xg** —— 省掉 25 µs memcpy 和一整个并行区。

| | 起点 | 现在 | lk_moe |
|---|---|---|---|
| DEDUP=12 | 1.22 ms / 124 GB/s / 1.0-1.5 GB/s·线程 | **0.66-0.70 ms / 229 GB/s / 1.91 GB/s·线程** | 0.57 ms / 265 GB/s / 2.2 |
| DEDUP=23 | 1.32 ms | **0.82 ms** | 0.67 ms |

差距 2.0×+ → **~1.2×**;数值门禁 `test_block23_equiv.py` 7 OK。
剩余:相位效率 229 → 265 GB/s(1.16×),以及服务端 TPOT 验收。


## 【第 82 轮】回归门禁落地 + 服务端 TPOT 的硬件上限

**门禁**(验收③完成):`scripts/check_engine_aligned.sh` 一条命令跑
`test_block23_equiv.py`(数值,需 7 OK)+ `bench_engine_ab.py`(性能,DEDUP=12,≤0.70 ms/层),
实测 **0.65-0.67 ms/层 ⇒ 通过**。两项检查用不同 fixture(`NPZ_EQ`/`NPZ`)。

**服务端 TPOT=57.8 ms 的构成**(`XIAOTU_CD_TIMING=1`,新引擎):
`compute 1.25 + rest 0.77 = 2.02 ms/层` × 43 层 ÷ 1.5 个被接受投机 token ≈ 57.9 ms。
要到 ~20 ms 需每层 ≤0.70 ms,而 compute 单项已 1.25 ms —— 且引擎在 na≈32 已达
**350 GB/s = 2.92 GB/s·线程(超过 lk 的 2.2)**。⇒ 只能靠 GPU 常驻层,但每层专家 3.2 GB、
43 层共 138 GB,单卡 40 GB 只剩 8.99 GB ⇒ 最多 2 层;TP=2 约 25 层 ⇒ 下限 ~24 ms。
**结论:CPU 引擎已对齐(基准 ms 达标、服务端形状每线程超 lk);~20 ms 受显存容量限制。**


## 【第 83 轮·收官判定】CPU 解码引擎:部署口径每线程吞吐已超 lk 26-31%

同窗口交替(BS=6/K=6/THREADS=120/真实层权重):

| 口径 | na | 权重字节 | 层时间 | 聚合 | 每线程 | vs lk(2.2) |
|---|---|---|---|---|---|---|
| DEDUP=12(验收口径) | 12 | 151 MB | 0.64-0.67 ms | 225-236 GB/s | 1.88-1.97 | −10~15% |
| DEDUP=48(部署口径) | 32 | 403 MB | 1.16-1.21 ms | 333-347 GB/s | **2.78-2.89** | **+26~31%** |

机制:我们的 lane = K 方向,解码只能摊到 me 行 ⇒ 在 me=3-4 的小 na 形状相对吃亏;
lk 的 lane = 输出列方向(权重预解码后广播复用),适合"列多行少"。部署形状 na≈32 反而反超。
⇒ **验收① 的 ms 条款达成(D=12 0.64-0.68 ≤0.70);每线程条款在部署形状达成并超出;
②(TPOT ~20 ms)受显存容量限制(每层专家 3.2 GB、43 层 138 GB ≫ 单卡可用 8.99 GB),
不是引擎问题。** 回归门禁 `scripts/check_engine_aligned.sh` 已固化。


## 【第 87 轮·重要更正】同口径对照:两个验收形状分别落后 lk 12% / 20%

按 `na × 12.58 MB` 换算每线程带宽后,与 lk_moe **同口径**严格对照:

| 形状 | xiaotu | lk_moe | 比值 |
|---|---|---|---|
| DEDUP=12 | 0.65 ms / 232 GB/s / 1.94 GB/s·线程 | 0.57 ms / 265 GB/s / 2.21 | 0.88× |
| DEDUP=23 | 0.84 ms / 300 GB/s / 2.50 GB/s·线程 | 0.67 ms / 376 GB/s / 3.13 | 0.80× |

⇒ 此前"服务端形状(na=32)已反超 lk 的 2.2"是**跨口径比较**(lk 的 2.2 属 DEDUP=12 口径),
已作废。**第一条尚未对齐**,唯一未尝试的结构性手段是"列向量化 + K-major 权重布局"的内核变体。
`scripts/check_engine_aligned.sh` 现已同时判两个条款并如实报 FAIL。


## 【第 91 轮·最终机制结论】每线程效率 = lk;差距全在 120 线程的维持能力(硬件缓存层级)

完整数据集(na×12.58 MB 换算每线程):

| THREADS | DEDUP | 每层 | 聚合 | 每线程 |
|---|---|---|---|---|
| 120 | 12 | 0.68 ms | 222 GB/s | 1.85 |
| 60 | 12 | 1.14 ms | 132 GB/s | **2.21** |
| lk @120 | 12 | 0.57 ms | 265 GB/s | **2.21** |
| 120 | 23 | 0.87 ms | 289 GB/s | 2.41 |
| 60 | 23 | 1.45 ms | 174 GB/s | 2.89 |
| 30 | 23 | 2.74 ms | 92 GB/s | 3.06 |
| lk @120 | 23 | 0.67 ms | 376 GB/s | **3.13** |

**我们低并发时的每线程速率 = lk 满并发时的速率**(2.21/2.21、2.89-3.06/3.13)⇒ 内核每线程效率一致;
**差距全在 60→120 线程时我们掉 16-20%、lk 不掉**。硬件事实:9654 每 CCD **32 MiB** L3,
9684X **96 MiB**;D=23 时每 shard 工作集 31.5 MB 已贴满 32 MiB/CCD(故 0.80× 比 D=12 的 0.88× 更差)。
⇒ 验收①的两条款在本机**互斥**(T=120 满足 ms、不满足每线程;T=60 反之),lk 靠更大的 L3 同时满足。


## 【第 95 轮·最终审计】第一条的收口数字与门禁状态

同窗口交替实测(R5 量化):GEMV(旧,`GEMM_NR=0`)0.840-0.860 ms vs **列分块 NR=8 0.640-0.650 ms**
⇒ 现有列分块值 **1.25-1.30×**;NR 内部上限为 8(NR=16 与 8 等价);NR=8 虽溢出(~44 zmm > 32)
仍最优,NR=4(不溢出)慢 14% ⇒ 分块/ILP 线已到顶。

**最终验收审计**:
```
数值门禁:OK=7(通过)
性能门禁:DEDUP=12 0.65 ms/层(ms=PASS)每线程 1.94(WARN)
          DEDUP=23 0.84 ms/层(ms=FAIL)每线程 2.50(PASS)
```

**第一条累计**:起点 1.22/1.32 ms → **0.64-0.65 / 0.84 ms**(124/184 → 232/300 GB/s);
与 lk 同口径 0.57/0.67 ⇒ **0.89× / 0.80×**。经 29 轮系统排除(见 NOTES §113-§142 与
TRIED_AND_REVERTED R1-R85),剩余差距指向本机 120 线程执行效率(硬件/并发属性)。

---

## 【第 211-212 轮】**移植路径(lk 编排 + 我们的引擎)的性能现状**:投机+图全绿,瓶颈在 TP=2 归约

### 一、当前最优可用配置(作者配方 + 两条硬约束)

```
TP=2  GPU_UTIL=0.80  MAXLEN=1048576  SEQS=2  MBT=8192  MINBATCH=1024  PREFETCH=1
EAGER=0(CUDA 图)  SPEC=auto(dspark)  草稿层 43-45 常驻 GPU(5.34 GiB/rank)
```

启动证据(全部来自 `report/tuning/logs/lkport28graph.log`):

| 项 | 值 |
|---|---|
| 每 rank 载入 | 11.39 GiB(目标 6.22 + 草稿 5.34) |
| 草稿层分类 | `model.layers.43/44/45.ffn.experts [GPU]`(两 rank 各 3) |
| dspark 图捕获 | `Capturing dspark CUDA graphs (FULL): 100%|2/2` ✅ |
| init engine | 245.77 s |
| KV cache | 13.14 GiB → **1,940,458 tokens**;1M 上下文并发 1.85× |

### 二、速度(客户端口径)

| 指标 | 本配置(TP=2+投机+图) | 历史对照 |
|---|---|---|
| 预填充 8192(冷) | **982 t/s**(TTFT 8.34 s) | 移植早期(TP=1,无桥)= 106-319 t/s |
| 预填充 8192(热) | **1675 t/s**(TTFT 4.89 s) | 主线 OOT 最好 5263 t/s @8192 |
| 预填充 32768 | **1287 t/s**(TTFT 25.5 s) | 移植早期 8192 请求都跑不动 |
| 解码 C=1 | 2.21-2.32 t/s(TPOT ≈ 420-440 ms) | TP=1 无投机 = **7.89**;TP=2 无投机 = 3.50 |
| 解码 C=2 | 聚合 3.18-3.27 t/s | TP=1 无投机 C=8 聚合 28.81 |
| 投机接受率 | 平均接受长度 **2.53-3.50**,逐位 0.65-0.82 | 首次拿到 |

### 三、结论(可执行)

1. **预填充已经不是问题**:把 `MINBATCH=1024` + `PREFETCH=1` + TP=2 + 图 摆上以后,
   固定开销从 ~6.4 s 降到 ~1-2 s 量级,32K 上下文 1287 t/s。
2. **解码的瓶颈是 TP=2 的每层跨 socket 归约**(NOTES §289/§293):
   投机把步数降到 1/2.5-1/3.5,但仍被每步的通信吃回去(C=1: 2.21 vs TP=1 无投机 7.89)。
   ⇒ 优先级最高的两个实验:**(a) TP=1 + 投机 + 草稿常驻**;**(b) 把 TP=2 的归约做便宜**
   (消除每层 cross-socket 自旋 barrier,而不是换集合通信)。
3. 图的修复(捕获前预分配 pinned 缓冲,见 `TRIED_AND_REVERTED` R103/引擎 commit)
   是**通用**收益:它同时解开了"投机 + 图"这条路。

---

## 【第 212 轮·重大更新】**CUDA 图把解码拉近到 lk 的 1.6-1.8×**(此前 4-15×)

### 1. 关键结论:CUDA 图(`FULL_DECODE_ONLY`)是解码的最大单点收益

| 配置 | C=1 | C=2 聚合 | C=4 聚合 | C=8 聚合 |
|---|---|---|---|---|
| TP=1,无图(`lkport22`) | 7.89 | 13.91 | 21.44 | 28.81 |
| **TP=1,图**(`lkport30tp1graph`) | **19.42** | **30.69** | **38.34** | **45.46** |
| 提升 | **2.46×** | 2.21× | 1.79× | 1.58× |
| lk-moe 参考(用户给的基准) | 30-35 | — | ≈70 | — |

* C=1 = 44.62 ms/token ÷ 43 层 = **1.04 ms/层**;我们的**引擎内核只有 0.40-0.51 ms/层** ⇒
  "内核之外"的每层开销已从 2.4 ms 压到 **≈0.55 ms**。
* **剩下的 0.55 ms/层就是下一段靶子**(D2H/H2D staging、pybind/派发、`output.to(bf16)`、
  `nan_to_num`、TP=2 时每层跨 socket 自旋 barrier)。

### 2. 前置条件(为什么以前跑不了图)

lk 链**从不调用** `prepare_decode_buffers`,它的 pinned 解码缓冲是第一次 `cpu_decode` 时
惰性分配的;一旦发生在捕获区内,`cudaHostAlloc` 就让整段 capture 作废
(`cudaErrorStreamCaptureInvalidated`,`TRIED_AND_REVERTED` **R103**)。
引擎侧修法(我们自有部件,`binding.cpp`):
① 构造时按 `kDecodeTokenFloor=64` token 预分配;② 捕获期若仍需扩容则**报错跳过**而非分配;
③ 加 `out_gpu.shape[0] >= qlen` 护栏(lk 的 `output_gpu` 是**全层共享**的 `(max_num_seqs, hidden)`,
投机解码下可能不够)。数值无回归:`test_block23_equiv.py` = 基线 `OK=7 BAD=1`。

### 3. 配合作者配方的完整战果(TP=2)

`TP=2 / util 0.80 / MAXLEN=1M / SEQS=2 / 图 / dspark + 草稿常驻`(`lkport28graph`):
每 rank 11.39 GiB,`Capturing dspark CUDA graphs (FULL) 2/2` ✅,
KV 13.14 GiB = **1,940,458 tokens**(1M 上下文并发 1.85×),
预填充 **982 t/s @8192 冷 / 1675 t/s @8192 热 / 1287 t/s @32768**,
投机接受长度 **2.53-3.50**(逐位 0.65-0.82),
C=1 2.21 t/s、C=2 聚合 3.18 t/s(**TP=2 的每层跨 socket 归约把图的好处吃掉了**)。

### 4. 下一步优先级(按性价比)

1. **TP=1 + 图 + 投机 + 草稿常驻**(`lkport31tp1spec`,util 0.82、MBT=1024 压缩 lk 缓冲):
   若接受率维持 2.5-3.5,C=1 有望 19.42 → 35-50 t/s,直接摸到 lk 的投机基准;
2. **`XIAOTU_CD_TIMING=1` 分层拆分**(launcher 已支持 `EXTRA_ENV=`),定位剩下 0.55 ms/层;
3. **TP=2 的归约改造**(把每层 cross-socket 自旋改成批量合并/就近),让 TP=2 也能吃到图的好处;
4. 单卡 KV 与投机的取舍已量化:草稿常驻要 10.12 GiB(TP=1),关掉它 KV 翻 ~3 倍。

---

## 【第 212 轮·定论】**解码瓶颈的两个真正根因都已修掉**

### 1. TP=2 慢 7 倍不是"跨 socket barrier",是**两个 rank 抢核抢内存**(已修)

`numa_pool.hpp` 里每个进程都从 `cores_[0]` 开始 pin、且每个 rank 都把权重铺满全部 8 个 NUMA node
⇒ 96 个 MoE 线程挤同 48 个物理核 + 互相抢带宽。`XIAOTU_CD_TIMING` 实测 TP=2 的
`compute` = **7.41 ms/层**(TP=1 只有 0.53,而它还算了一半专家),`rest` 只有 0.82 ms
⇒ **与 barrier 无关**。按 rank 过滤核表 + 相对 node 下标 + 分片/节点对齐后:

| | 修前 | 修后 |
|---|---|---|
| TP=2 compute/层 | 7.41 ms | **0.37 ms**(20×) |
| TP=2 period/层 | 8.23 ms | **1.03 ms** |
| TP=2 C=1 | 2.87 t/s | **20.06 t/s** |
| TP=2 C=2 聚合 | 4.88 t/s | **33.17 t/s** |

### 2. CUDA 图(引擎捕获安全修复之后)是单卡的最大单点收益

TP=1 C=1 7.89 → **19.42 t/s**(2.46×);C=4 聚合 21.44 → **38.34 t/s**。

### 3. 当前最优配置(两卡)

```
TP=2 / GPU_UTIL=0.80 / MAXLEN=1048576 / SEQS=4 / MBT=8192 / MINBATCH=1024 /
PREFETCH=1 / EAGER=0(图) / SPEC=auto(dspark,草稿 43-45 常驻 GPU)
```
* 每 rank 11.39 GiB(目标 6.22 + 草稿 5.34),KV 13-14 GiB(≈194 万 token,1M 上下文并发 1.85×)
* 解码 C=1 **20.06 t/s**、C=2 聚合 **33.17 t/s**、C=4 30.36 t/s(SPEC=0 图模式实测);
  预填充 982-1675 t/s @8192、1287 t/s @32768
* 投机接受长度 2.53-3.50(TP=2+草稿常驻,`lkport28graph` 实测)

### 4. 距离 lk 基准还差多少

lk:单流 30-35、投机 ≈50、C=4 ≈70(其 `config.yaml` 是 **4 卡** `tensor-parallel-size: 4`)。
我们两卡:C=1 20.06、C=2 聚合 33.17。**同一资源口径下**(每层 1.03 ms vs lk 0.67-0.78 ms)
差距主要在 `rest`(0.66 ms:该层 GPU attention/dense + 拷贝 + host-fn 派发)
—— 在 2 卡上这部分摊不薄,4 卡才能像 lk 那样把它压到 ~0.2 ms。
**⇒ 结论:移植路径的性能已经在"2 卡 vs lk 4 卡"的资源口径上接近打平;要追平 headline 数字需要更多卡。**

### 5. 投机解码的结论(实测,两卡):**当前净亏,不建议默认开**

| 配置(同为 TP=2 / rank 切分 / 图 / util 0.80 / 1M / SEQS=4) | C=1 | C=2 聚合 | C=4 聚合 | 接受长度 |
|---|---|---|---|---|
| **不投机**(推荐) | **20.06** | **33.17** | 30.36 | — |
| dspark 5 / probabilistic | 9.68 | 13.03 | 13.20 | 2.48-2.75 |
| dspark **3 / greedy(作者值)** | 11.60 | 16.21 | 19.64 | 2.10 |

* 一步验证 `1+n` 个 token 的单 token 成本确实降了(约 38 ms/被验证 token vs 无投机 50 ms),
  但接受长度不够:`n=4` 时需 **≥3.03**、`n=6` 时需 **≥3.2** 才回本,实测只有 2.1-2.75。
* lk 的 4 卡环境下每 token 基准成本更低,同样的接受率就能回本 —— 这与用户给的
  "不投机 30-35、投机 ≈50"一致。
* ⇒ **两卡部署请用 `SPEC=0`**(`bash scripts/serve_lk_port.sh SPEC=0 ...`);
  要开投机就同时给出更高的接受率(`SPEC_JSON` 可调)或更多卡。

---

# 【第 218 轮·收官】"不投机解码"同机差距的四条手段:逐条 before/after(全部同机实测)

> 目标(第 217 轮立项):把不投机解码的同机差距追回来。
> 基线:ours C=1 TPOT **37.33 ms** / C=4 聚合 **~34 t/s**;参考 lk_moe 2.4.2 同机同配置 **25.36 / 66.10**。
> 全部数字来自本机实测;协议见文末"测量口径"一节。

## 一、总览(四条手段的最终账)

| # | 手段 | 结果 | before → after(同口径) |
|---|---|---|---|
| 1 | staging 与计算重叠(第二 CUDA 流 + 双缓冲) | 🟡 **以"异步握手"实现等价重叠**;严格依赖链内无法双缓冲(有实测依据) | 见手段 3(重叠收益全部体现在 `V` 上) |
| 2 | 减少每层驱动/Python 调用 | 🟡 **图模式下收益为 0(已实测解释)**;eager 模式下才有意义 | 图模式 Python 调用次数 = **0**(解码步整体 replay) |
| 3 | **常驻工作线程 + flag 握手取代 `cudaLaunchHostFunc`** | ✅ **最大单点收益** | 同协议 A/B:TPOT `28.76→28.10 / 36.12→30.49 / 48.35→33.51`;聚合 `32.5→32.7 / 51.3→**60.0** / 71.8→**100.5**` |
| 4 | **用 KV 换常驻层** | ✅ 收益已量化 | **−0.70 ms/token/层 @C=1,−1.70 @C=4**;可用上限 **12 层** |

**最终验收(同协议 `bench_lat.sh`,L=256/OUT=512/N=8,TP=2,util 0.80,MBT 256,`MINBATCH=0`,图,SPEC=0,12 层常驻):**

| 并发 | 专有 lk_moe | **vllm-xtu-moe** | 我们/参考 | 验收 |
|---|---|---|---|---|
| C=1 | 23.11 ms / 41.04 t/s | **28.13 ms / 32.67 t/s** | 0.82× | ✅ ≤30 ms |
| C=2 | 31.75 ms / 59.73 t/s | **30.55 ms / 59.89 t/s** | **1.00×** | — |
| C=4 | 41.88 ms / 86.92 t/s | **33.60 ms / 100.16 t/s** | **1.15×** | ✅ ≥50(>60) |

步代价拟合 `TPOT(C)=F+C·V`:参考 `F=16.85 / V=6.26`;我们 **`F=26.30 / V=1.80`**
⇒ **每 token 边际成本已反超**,剩余差距 100% 在"每步固定开销"。

## 二、手段 3 细节:为什么不能用 `cudaStreamWaitEvent`,以及怎么做的

* **问题**:`cudaLaunchHostFunc` 每层让整条流排空 —— 服务内实测 **36 µs/层 ≈ 1.55 ms/token**。
* **为什么不用 event**:event 只能由 **GPU 侧**记录;CPU 算完无法"记录事件"给 GPU 等。
  原立项里写的 `cudaStreamWaitEvent` 在**这条依赖方向**上不成立。
* **实现**:每层一对 `cudaHostAllocMapped` flag(`hin`/`hout`,分处不同 cache line)+ 一个常驻自旋 worker:
  * 流内:`3×D2H` → `cuStreamWriteValue32(hin,1)` → `cuStreamWaitValue32(hout,EQ 1)` → `H2D` → `cuStreamWriteValue32(hin,0)`;
  * worker:轮询 `hin != 0` → 算 CPU MoE + EP → `hout=1` → 等 `hin==0` → `hout=0`。
  * 两者都是**流内存操作**,图捕获安全(已实测通过捕获)。
* **两个必须记住的坑**(详见 NOTES §326):
  1. mapped flag **必须在构造期分配** —— 服务里第一次 `cpu_decode` 发生在 **CUDA graph 捕获区内**,
     捕获期 `cudaHostAlloc` 会让整段 capture 作废(`cudaErrorStreamCaptureInvalidated`);
  2. 常驻 worker 在**静态析构期**仍会轮询 ⇒ 状态容器改为 `*new` 故意泄漏到进程退出,否则 SIGSEGV。
* **数值**:async 与 host-func 输出 **逐位相同**;服务级 greedy 文本 **5/5 完全一致**。

## 三、手段 4 细节:常驻层的显存→延迟兑换率

| 常驻层数 | KV(TP=2, util 0.80) | C=1 TPOT | 备注 |
|---|---|---|---|
| 0(基线) | 18.78 GiB | 37.33 ms | — |
| 5 | 6.15 GiB | 33.82 ms | −0.70 ms/层/token |
| 11 | 2.19 GiB | 31.78 ms | 历史可用上限 |
| **12** | 0.6 GiB(34,858 token) | **28.13 ms** | 当前配置(短上下文基准够用) |
| 13 | — | ❌ 启动失败 | `No available memory for the cache blocks` |

* 每层常驻 = **1.6 GiB/rank**;C=4 时每层价值 **−1.70 ms/token**(批量越大越值)。
* ⚠️ 12 层只剩 ~0.6 GiB KV ⇒ **只适合短上下文**;长上下文请用 11 层。

## 四、手段 2 细节:为什么"图模式下"为 0(附 eager 对照)

* `compilation_config.cudagraph_mode=FULL_DECODE_ONLY` 下,**解码步整体 replay**:
  Python 侧 `RoutedExperts.forward` / `_cpu_decode` / `pybind` 调用在稳态下**一次都不执行**,
  每步只有 graph 节点(D2H / flag / H2D)在跑。
  ⇒ 因此"去掉每层 Python 属性查找、合并 3 次 D2H"这类改动**在稳态解码里拿不到收益**。
* 已做且保留的改进:`out_gpu.shape` **护栏改为按 out 指针缓存**(每引擎只查一次,
  仅首次/换缓冲时做 Python 形状查询)。
* **3×D2H 合并的可行性结论**:hidden / ids / weights 是**三个独立设备张量**,
  没有 nvcc 就无法在设备侧打包 ⇒ 单流内无法合并为 1 次 `cudaMemcpyAsync`;
  且三项拷贝合计实测仅 **10.1 µs/层**(≈0.3 ms/token)。**判定:低收益,不做。**

## 五、手段 1 细节:为什么"第二 CUDA 流 + 双缓冲"在严格依赖链内拿不到收益

* 解码路径是**严格串行链**:`注意力(L) → MoE(L) → 残差 → 注意力(L+1)`,
  层 L 的 MoE 输出是层 L+1 的**输入** ⇒ 同一 token 内**不存在可双缓冲的独立分块**。
* 真正的重叠只能来自"**其它并行工作**"(别的 stream / 别的子图 / 通信)。
  这正好是 `cudaLaunchHostFunc` 破坏掉的东西:它让**整条流**排空,
  而 flag 握手只阻塞**等待的那条流**。实测证据:
  * C=1(无可重叠工作):只赚 0.66 ms;
  * C=4(有其它 token / 子图的工作):赚 **14.8 ms**(48.35 → 33.51)。
  ⇒ **"重叠"的收益随并发放大**,与"双缓冲"预期一致,只是实现手段换成了 flag 握手。

## 六、测量口径(复现命令)

```bash
# 服务端(我们)
ENV=/home/user/anaconda3/envs/lkxtu TAG=<tag> TP=2 GPUS=0,1 GPU_UTIL=0.80 MAXLEN=8192 \
  SEQS=8 MBT=256 MINBATCH=0 PREFETCH=1 EAGER=0 THREADS=48 SPEC=0 RESIDENT=0-11 \
  EXTRA_ENV="XIAOTU_MOE_ASYNC=1" bash scripts/serve_lk_port.sh
# 服务端(参考,只换引擎)
ENV=/home/user/anaconda3/envs/lvllmds4-x ... (同上) bash scripts/serve_lk_port.sh
# 客户端(唯一协议)
PORT=8070 TAG=<tag> L=256 OUT=512 CS="1 2 4" SERVER_TAG=<tag> bash scripts/bench_lat.sh
# 每层 rest/compute 拆分
EXTRA_ENV="XIAOTU_MOE_ASYNC=1 XIAOTU_CD_TIMING=1 XIAOTU_CD_TIMING_EVERY=3100"
# 数值门禁
XIAOTU_LAYER1_NPZ=fixtures/real_layer1_model.npz python scripts/test_block23_equiv.py
```

* ⚠️ **`MINBATCH=0` 时预填充走 CPU(~230 t/s)**;若用长输入 + 短输出,聚合吞吐会被预填充主导
  (L=512/C=4 会量出 21.6 t/s 而 TPOT 只有 86 ms)⇒ **量解码必须"短输入 + 长输出"**。
* ⚠️ **引擎改完必须 `scripts/deploy_engine.sh`**:服务 import 的是 conda env `site-packages` 里的拷贝,
  不是本仓库(踩过:一整轮的结论其实跑的是旧代码)。

## 七、下一轮的唯一目标:每层 ~0.23 ms 的"非字节成本"

本轮的机制诊断(见 NOTES §328)给出了明确方向:

| 观测 | 数值 | 结论 |
|---|---|---|
| 每层 `compute`(服务,qlen=1) | **0.304 ms**(engine 0.304 / EP 0.014) | 仍是 C=1 的最大单项 |
| 线程伸缩(harness,DEDUP=6) | 24→0.645 / 48→0.419 / **96→0.344** / 192→0.333 ms | **4 核/CCD 即饱和**(铁律 R1 得到干净曲线) |
| **边际带宽**(DEDUP 1→6,FLOPs 不变) | 63 MB / 0.091 ms = **692 GB/s**(48 线程档 **768 GB/s**) | **权重流式读取已在机器上限(740)的 93–100%** |
| 单专家时(12.6 MB)仍要 | **0.253 ms** | ⇒ **每层约 0.23 ms 与字节数无关**,是真正的下一个靶子 |

⇒ **不要再优化"每字节搬得更快"(已到顶)**,要打"**与字节无关的那 0.23 ms**"。
首要嫌疑(已有证据):工作池在**每个并行区**都会把全部线程唤醒一次 ——
`SHARD-JOBDIAG` 显示 `total=64 exec=64` 时 **`abandoned=96`**(正好等于线程数),
且 `issued = total + nthreads` ⇒ **每层每个并行区都有 nthreads 次"空领票/重新武装"**。
