# DeepSeek-V4.1-Flash 常驻内存足迹:**同一把尺子下的参考 vs 我们**

日期:2026-09-16 · 机器:2× EPYC 9654(192 物理核,SMT off,2 NUMA node)+ 3× A100-PCIE-40G
(只用 0,1) · 模型:`DeepSeek-V4.1-Flash` ModelScope 快照 · TP=2 · 目标 = **NOTES §498 的 item (5)**

---

## 1. 方法(先说清楚尺子,否则数字没法比)

* 工具:`report/tuning/probes/mem_footprint.sh` —— 每 15s 采样**服务进程树**的总 RSS
  (APIServer + EngineCore + 2×Worker;按 `/proc/<pid>/stat` 的父子关系做 BFS),
  同时记最胖进程的 `smaps_rollup`(Rss/Pss/Private_Dirty/RssShmem)与每 node 空闲。
* **为什么必须按"进程树总 RSS"**:TP=2 的服务是 4-5 个进程;而且
  * 全机"最胖进程"早期会命中机器上别的常驻进程(实测 `dsh web` 1.8 GiB),量错对象;
  * **每 node 空闲**会把**页缓存**算进"已用"(参考脚本还专门 `VLLM_ENGRAM_DROP_PAGE_CACHE=0`
    把 189 GB 的 Engram 表的页缓存留住)—— 页缓存是可回收的,不该算进常驻集。
    实测:参考跑完时 node 已用 ≈1254 GB,而它的服务树只有 604 GiB ⇒ 差的那 ~650 GB 是页缓存。
* 两边**逐字相同**的东西:同一个模型快照 / 同一个 vLLM 基座(**lvllm 2.5**)/ TP=2 / `gpu_util 0.95` /
  `maxlen 65536` / `MBT 8192` / `max-num-seqs 2` / `fp8_ds_mla` /
  `--compilation-config {mode: VLLM_COMPILE, cudagraph_mode: FULL_DECODE_ONLY}` / dspark /
  **都不设 GPU 常驻层** / `THREADS`(=`LK_THREADS`)= 60。
  **唯一变量 = CPU MoE 引擎**(参考 = lk_moe;我们 = 插件在同一个 env 里接管,见 §501 的
  `_install_lvllm_engine_substitution`)。
* 复现:`bash report/tuning/probes/ref_v41_mem.sh`(参考)、
  `bash report/tuning/probes/xtu_v41_mem.sh`(我们)。

---

## 2. 结果

| | 参考(LvLLM + lk_moe) | 我们(xiaotu-moe,同一 env) | 差 |
|---|---|---|---|
| **服务进程树总 RSS(峰值)** | **603.9 GiB** | **1466 GiB** | **+862 GiB(2.43×)** |
| 最胖单进程 | 307.7 GiB(Pss 260.2) | 731.9 GiB(Pss 731.9) | 2.38× |
| 两个 rank | ~302 GiB × 2 | **731.9 GiB × 2** | — |
| 跨 rank 共享 | 有(Pss < Rss 约 48 GiB) | **几乎没有(Pss = Rss)** | — |
| ready 用时 | 381 s | ~623 s | 1.6× |
| 跑完时全机空闲 | ~860 GB | **4 GB(available 20 GB)** | 机器被打满 |

折算到 40 层:**+21.5 GiB/层**。

⚠️ 这个数字**取代**此前"上游 +9.18 GiB/层 ≈ 367 GiB"的估计:那个估计是在
**两把不同的尺子**之间做差(我们的"单进程峰值" vs 参考的"文档 590 GB" / node 已用含页缓存)
得到的。§503 这次是同机、同模型、同基座、同参数、同一把尺子。

---

## 3. 能立刻得出的结论

1. **两个 rank 各存一份完整的主机侧权重集**(Pss = Rss ⇒ 没有跨进程共享)。
   参考没有这个问题(Pss < Rss)。⇒ 第一优先:**让 TP=2 的两个 rank 共享主机权重**
   (我们在 Mode A 里有 `EP-shm`(`/dev/shm/xiaotu_ep_*.bin`)这条路;Mode B 下显然没生效)。
2. **机器被打满不是"理论风险"**:V4.1 稳态下 `free` 只剩 4 GB、`available` 20 GB
   —— 任何一次瞬时峰值都会触发 OOM-kill(此前 §495 就是在排查这个方向)。
3. 参考的 604 GiB 与它 release notes 里写的 "peak observed ≈ 590 GB" 吻合 ⇒
   **尺子是可信的**(同一把尺子量出的两边数字才可比)。

---

## 4. 下一步(按优先级,item (5) 的收敛路径)

1. **拆来源**:把 1466 GiB 分成 ①专家权重(fp4 量化后每层多少 GiB × 40)②Engram 表(已知 ~189 GB)
   ③上游那部分"每层 +9.18 GiB"④pinned 缓冲/EP shm。工具已经就位:采样里带上
   `/proc/<pid>/smaps` 的**大段聚合**(而不是只 rollup),必要时按 mapping 名字分类。
2. **消掉重复**:确认 Mode B 下为何两个 rank 各自持有整份主机权重(参考实现是共享的),
   把 Mode A 的 `EP-shm` 机制接过来;验收 = 服务树总 RSS 接近参考量级(±10%)。
3. **回归**:每改一处都用 `ref_v41_mem.sh` / `xtu_v41_mem.sh` 同一把尺子复测,
   并把 V4 的数值门禁(1.873e-02)+ 确定性门(11/11)一起过一遍(硬约束)。

---

## 5. 更新(2026-09-18):`ENGRAM_LAST` 转正后的实测足迹 —— 本文档第 2 节的数字已被取代

| 配置(TP=2,真实权重) | 峰值(服务树 RSS) | READY |
|---|---|---|
| `XIAOTU_ENGRAM_LAST=0`(旧默认) | **1087.6 GiB** | 436 s |
| `XIAOTU_ENGRAM_LAST=1`(现默认) | **637.9 ~ 644.4 GiB(−41%)** | 351-353 s |
| 同上 + 1M + 投机 + 常驻 2 层(§570 验收) | **629.4 GiB** | 388 s |

**曲线形状**(`TAG.memfoot`,15 s 一点):t≈45-126 s 爬升到峰值(`load_weights`);
**t≈145 s 一次性 −358.9 GiB**;此后爬升到稳态 ~590 GiB。
⇒ 逐层源释放**已经是逐层的**(80 条 `released 3.16 GiB` = 40 层 × 2 rank),只是发生在
`process_weights_after_loading` 阶段 ⇒ **峰值只比稳态高 ~48 GiB(7.5%)**,
①"Patch B" 的残余空间已从"500 GiB 级"缩到"50 GiB 级",**判定不值得再做大手术**。

**NPS4 单节点余量**:峰值时最低 node 余量 ≥35.0 GB(旧默认下逼近单 node 耗尽)。

**逐 chunk 表内容校验**(默认开):`checked=98 chunks=97` 每张表,两 rank 的
`vocab_start` 分别 0 / 192001740(layer1)、0 / 192007016(layer14)⇒ 8/8 PASS。
