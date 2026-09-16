# 长 prefill 的逐层 GPU 流式 —— **实现与设计**(开发文档)

> 一句话:**一层一层把专家权重从「引擎自己的 NUMA 分片」DMA 进显存算 MoE,GPU 只额外占两层权重,
> 主机侧一分钱不多花**;当前层在算的同时,下一层的 H2D 已经在飞(ping/pong 双槽 + 异步 stream)。
> 开启方式:一个阈值 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`(某层 prefill token 数 ≥ 阈值 ⇒ 该层走 GPU)。

**读者**:未来的我 / 任何接手的人。本文写的是**代码里现在真实跑的路径**(§508 更新),
不是设计草案 —— 本文档曾经漏掉最关键的一点(权重从哪来),导致我据此做出了错误结论并差点
关掉这个功能,故把"实现"单独写成一节,并附**证据位置(文件:行)**。

---

## 1. 设计原则(为什么要有它)

| 事实 | 推论 |
|---|---|
| 小 batch decode 在 CPU 上带宽足够(几十 ms/token) | decode 留在 CPU —— **不要**把专家权重常驻显存 |
| 长 prefill(几千 token)在 CPU 上极慢 | prefill 应该上 GPU |
| 整层专家权重常驻显存放不下(单卡 137 GiB 量级) | 只能**逐层流式**:算完即释放 |
| H2D 有地板(单卡全量 ≈137 GiB / ~25 GB/s ≈ 5.9 s) | 必须**重叠**:算第 L 层时搬 L+1 层,否则流式白做 |
| 主机内存是总约束(IRON_RULES **R-VRAM**) | 权重**不许**在主机侧再复制一份 ⇒ 必须复用引擎已有的分片 |

### 与其它实现的对照

| 实现 | 阈值开关 | 说明 |
|---|---|---|
| vLLM fork(`Lvllmds4-x`) | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | 命中后走标准 vLLM GPU MoE(Marlin);只有部分层标记为 GPU 常驻 |
| ktransformers | `KT_GPU_PREFILL_TOKEN_THRESHOLD`(默认 2048) | 逐层 GPU prefill,自有 Triton MXFP4 kernel |
| **本插件** | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | 不依赖 Marlin(vLLM 的 fp4 Marlin 只支持 NVFP4 group-16,不支持 MXFP4 block-32);自写 Triton 分组 MXFP4 MoE;**权重直接从引擎分片流式** |

---

## 2. 实现:一次 GPU prefill 的**完整数据流**

下面每一步都标了证据位置,便于维护者对照代码。

1. **逐层判定**(`vllm_xiaotu_moe/mixed_experts.py:1123-1140`)
   `_gp_min = gpu_prefill_min_tokens()`;若 `_gp_on and qlen >= _gp_min` ⇒ 本层走 GPU。
   *阈值每层每次 forward 都读一次* ⇒ 可以用**文件桥**在运行期切换(见 §3.4)。
2. **显存预检**(`gpu_prefill.staging_bytes()` :779 / `fits_device()` :800)
   需要量按 `max(w13_kmajor + w13_node_shard, w13_kmajor + w2_kmajor + w2_node_shard) + 2*scales`
   算,并要求 `free ≥ need × 1.25`(25% 余量)。**预检在 vLLM 定完 KV cache 之后**才允许
   (`gpu_prefill.install_profile_guard` 包住 `profile_run`/`_initialize_kv_caches`;
   日志 `startup finished (KV cache sized) -> GPU prefill is now allowed subject to the VRAM preflight`)。
   不够 ⇒ **优雅退回 CPU**(不报错),打印可操作建议(见 §6)。
3. **装配 K-major 权重 —— ★我们最关键的一步**
   `kmajor_from_engine_shards(engine, dev, hidden, I, E, group_k)`(`gpu_prefill.py:840`)
   内部调用 `engine.copy_hostbuf_to_device(which, node, dst_ptr, stream)`(`:831`),
   把**引擎自己的 NUMA 分片**直接 DMA 成 GPU 上的 K-major 布局。
   ⇒ **主机侧零额外内存**;`[E,2I,H/2] → [E,H/2,2I]` 的转置由"装配"一步完成,
   设备端不再做 transpose。
   *只有*当引擎**没有分片**时才返回 `None`,并退到"checkpoint 源张量"兜底路径
   (`mixed_experts.py:1278-1282`,`_pf_reason = "checkpoint source (engine has no shards)"`)。
4. **ping/pong 双槽 + 异步 H2D**(`gpu_prefill.prefetch_layer()` :993,`_prefetch_stream()` :984)
   `nslots = max(2, XIAOTU_MOE_PREFETCH_SLOTS)`,**每槽 ≈ 一层权重**(TP=2 每 rank ≈1.7 GB,
   TP=1 ≈3.4 GB)。为 L+1 发 H2D 到另一个槽,计算流只 `wait` 它的 `ready` event;
   `PrefetchSlot.__slots__ = ("bufs","ready","busy","t0","t1")`。
   **下限必须是 2**(1 槽会让 L+1 的 H2D 覆盖 L 正在用的权重 ⇒ 正确性 bug,实测
   `err_vs_L=17.5` vs `err_vs_L+1=0.096`)。
5. **GPU MoE 计算**:Triton 分组 MXFP4 kernel(`gpu_prefill.gpu_moe_layer`,`slot=` 接口)。
   bf16 全 DS-V4 维度 RMS 相对误差 **5.6e-3**(≈ bf16 量化精度 2⁻⁸),golden 见
   `scripts/gpu_prefill_golden.py`。
6. **释放/复用**:槽是**环形复用**的 —— 下一层直接写回本槽,不需要显式 free;
   显存峰值 ≈ 2 层 K-major + 激活。
7. **专家并行(TP>1)**:每 rank 只搬 `E/TP` 个专家,路由 id 做 rank 偏移,越界置 -1、
   权重置 0,计算后归约(EP 下走共享内存部分和 / 否则 `all_reduce`)。

---

## 3. 我们的特色设计(与"把模块拷到 GPU"的做法对比)

### 3.1 ★ 权重源 = **引擎自己的 NUMA 分片**,不是一份 PyTorch 副本
引擎在构造期已经把权重按 socket 分片、拷进自己的紧凑缓冲(并 `mbind` 到本 node)。
GPU prefill **借用**这份数据(`copy_hostbuf_to_device`),因此:
* **主机侧不增加任何常驻内存**(满足 R-VRAM 的总约束);
* 不需要"为了 GPU 而把权重留在主机"的额外一份(那种做法在 V4.1 上是 **~6.7 GiB/层/rank**,
  40 层 ≈ **200+ GiB/rank** —— 实测:同一配置下把 `XIAOTU_GPUPREFILL_WCOPY` 打开,
  服务进程树从 **691.6 GiB → 1097 GiB**);
* 分片是 socket 本地的 ⇒ H2D 的源内存也是本 node 的,不制造跨 socket 流量。

### 3.2 ping/pong 双槽 + 真异步 H2D
"算 L 的同时搬 L+1"。实测收益(单层微基准,§5.1):T=8192 时 717 → **1495 tok/s**;
H2D 地板 ~127 ms/层在 T ≥ 8K 时被计算完全掩盖。

### 3.3 三级优雅降级(**永不因为显存不足起不来**)
```
ping/pong 重叠  ──VRAM 不够──▶  同步逐层 H2D(无重叠)  ──还不行──▶  该层留在 CPU
        (prefetch_layer 返回 None)         (打印 "GPU prefill DISABLED for this layer -> staying on CPU")
```
OOM 结论会**粘住**(`self._gpu_pf_ok = False`),避免每次 forward 反复试错/反复打印。

### 3.4 阈值驱动 + **运行期可切换**(文件桥)
TP>1 时模型在 worker 子进程,改客户端 `os.environ` 不生效 ⇒ 阈值可以用**文件**指定
(`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE`),配合"每层每次 forward 都读一次"实现热切换。
⚠️ 历史坑:该文件桥曾经只认不带 `VLLM_` 前缀的名字 ⇒ 所有"运行期切换阈值"的 A/B **实际都走了同一条路径**
(已在 `gpu_prefill_min_tokens()` 里同时接受两种前缀)。

### 3.5 CUDA graph 契约 + "KV 先定序"
* 捕获期间任何 `cudaHostAlloc/cudaFreeHost` 都会让 capture 失效
  (`cudaErrorStreamCaptureInvalidated`)⇒ pinned 缓冲**在捕获之前按最大捕获尺寸预分配好**,
  稳态不再扩容;prefill 若需要更大缓冲,只在**非捕获**路径上扩容。
* vLLM **先**定 KV cache 尺寸,所以 GPU prefill 的预检必须在它之后 —— 否则会拿"还没被 KV 吃掉"
  的假空闲显存做判断,启动时看着够、跑起来 OOM。

---

## 4. 旋钮

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | `0`(**默认关**) | 阈值;`0` = 关闭。**必须同时把 `MBT`(`--max-num-batched-tokens`)提到 ≥ 阈值**(否则 batch 永远到不了阈值),实践用 4096-8192 |
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE` | 未设 | 阈值文件(运行期切换;两种前缀都认) |
| `XIAOTU_MOE_PREFETCH_SLOTS` | `2` | 环深度;每槽 ≈ 一层权重。**不许设 1**(正确性 bug) |
| `XIAOTU_GPUPREFILL_WCOPY` | `0` | 是否在构造期再做一份 Python 侧权重副本。**只服务兜底路径**(引擎无分片);默认 0,走到兜底时会打告警 |
| `XIAOTU_GP_SPLIT` | 未设 | `1` = 打印每层装配耗时/空闲显存(`[gp-split] asm=..ms free=..`) |
| `XIAOTU_GPU_PREFILL_BM/BN/BK/BH` | `64/64/64/64` | 分组 GEMM tile |
| `XIAOTU_GPU_PREFILL_STAGES` | `2` | `tl.range` 软件流水级数 |

---

## 5. 实测

### 5.1 单层微基准(同步 vs 重叠;**V4 维度** E=256/H=4096/I=2048/topk=6,43 层折算)
| T | 无重叠 tok/s | 重叠后 tok/s | 重叠后 ms/层 |
|---|---:|---:|---:|
| 2 048 | 252 | 373 | 127.5 |
| 4 096 | 438 | 748 | 127.4 |
| 8 192 | 717 | 1 495 | 127.4 |
| 16 384 | 1 070 | **1 918** | 198.7 |
| 32 768 | 1 414 | **1 994** | 382.2 |
| 49 152 | 1 583 | **2 013** | 567.9 |

### 5.2 TP=2 专家并行等效(E=128/topk=3,每 rank 只算自己那半)
| T | 重叠后 tok/s |
|---|---:|
| 4 096 | 1 495 |
| 8 192 | 2 989 |
| 16 384 | **3 803** |
| 32 768 | **3 979** |

### 5.3 端到端(**历史数据,V4 维度、M-chunking/按工作量切分之前**,仅作趋势参考)
| prompt tokens | CPU prefill | GPU prefill(1 卡) | GPU prefill(2 卡 TP=2) |
|---|---|---|---|
| 2 091 | 112.93 s | **5.638 s** | 2.921 s |
| 4 137 | 439.38 s | **5.763 s** | **3.993 s** |
| 8 229 | — | **8.639 s** | — |
| 16 039 | — | **18.237 s** | **13.956 s**(≈1 149 tok/s) |

### 5.4 **最新(V4.1,TP=2,MBT=8192,本机)**
| 项 | 数值 |
|---|---|
| TTFT @2048(热) | **824 ms** |
| TTFT @8192(热) | **1003 ms** |
| 对照:CPU 预填充 TTFT @512 | 4661 ms |
| 主机内存(服务进程树,GPU prefill 开 + WCOPY=0) | **691.6 GiB**(两个 worker 各 ~345 GiB) |
| 同一配置把 WCOPY 打开 | **1097 GiB**(⇒ 那份 Python 副本 ≈ **200 GiB/rank**) |
| 显存占用(GPU prefill 开,MAXLEN=32768,gpu_util .90) | ~24.7 GB/卡 |
| 首帧冷启动 | 24.2 s(一次性 Triton JIT;**不是**性能问题,预热后 824 ms) |

---

## 6. 坑与诊断

| 现象 | 原因 / 处置 |
|---|---|
| 设了阈值却完全不生效 | ① `MBT < 阈值`(batch 永远到不了阈值);② 阈值文件桥名字前缀(已修);③ 预填充形状被 CUDA 图捕获成 CPU 分支 ⇒ 用 `--cudagraph-capture-sizes` 只捕获解码尺寸 |
| `GPU prefill DISABLED for this layer -> staying on CPU` | 预检没过(打印里有 `per-layer staging`/`要求 free`/`实际 free` 与可选处置:TP=2 + `RANK_SPLIT=2`、换大显存卡、降 `gpu_util`) |
| `no VRAM for ping-pong slots, falling back to synchronous H2D` | 显存放不下双槽 ⇒ 自动降级(仍然能用,只是没有重叠)。**不要**改成 1 槽省显存 |
| 首次请求特别慢(几十秒) | Triton JIT 编译,一次性;量性能前**先同形状预热** |
| 主机内存暴涨到 1 TB 级 | `XIAOTU_GPUPREFILL_WCOPY=1`(或引擎没有分片走了兜底路径)⇒ 看是否打了兜底告警 |
| 兜底路径崩溃(原生) | checkpoint 源张量已被 `clean_weights_after_loading` 释放;正解是让引擎有分片(`RANK_SPLIT=2` + 正确的 nshard) |

---

## 7. 怎么验证(命令 + 期望)

```bash
# 1) 数值:纯 torch 参考(golden)
python scripts/gpu_prefill_golden.py                 # 期望 RMS 相对误差 ~5.6e-3(V4 维度)

# 2) 单层:同步 vs 重叠
python scripts/bench_gpu_moe_prefetch.py             # 期望 T=8192 时重叠 ≈ 2x

# 3) 端到端 + 内存(本仓统一尺子:服务进程树总 RSS)
TAG=v41_gp GP_MIN=1024 MBT=8192 MAXLEN=32768 \
  bash report/tuning/probes/xtu_own_v41_mem.sh       # 期望 TTFT ~0.8-1.0 s、树 ~690 GiB
```

判定"GPU prefill 有没有真的生效"的三条硬证据:
1. 服务日志里没有 `GPU prefill DISABLED` / `no VRAM for ping-pong slots`;
2. `XIAOTU_GP_SPLIT=1` 会打印每层的 `[gp-split] asm=..ms free=..GiB`;
3. 同形状第二次请求的 TTFT 明显低于 CPU 预填充(本机 824 ms vs 4.6 s@512)。

---

## 8. 相关文件

| 文件 | 作用 |
|---|---|
| `vllm_xiaotu_moe/gpu_prefill.py` | kernel + **分片装配**(`kmajor_from_engine_shards`)+ **预取双槽**(`prefetch_layer`)+ 驱动 + torch 参考 |
| `vllm_xiaotu_moe/mixed_experts.py` | 逐层判定/预检/降级(GPU prefill 分支 :1123-1300) |
| `vllm_xiaotu_moe/hybrid_model.py` | Mode A 的 GPU-prefill 分支与层间预取编排 |
| `xiaotu_moe/gpu_prefill_bridge.py` | 给引擎类补 `gpu_prefill` 接口;K-major 兜底副本(`WCOPY`) |
| `scripts/gpu_prefill_golden.py` / `scripts/bench_gpu_moe_prefetch.py` / `scripts/dsv4_prefill_curve.py` | golden / 单层微基准 / 端到端曲线 |
| `report/tuning/probes/xtu_own_v41_mem.sh` | 端到端 + **服务进程树内存**测量 |
| `report/tuning/IRON_RULES.md` | R-VRAM(显存优先级)、R-VERIFY(别从开关反推能力) |

### 更正记录
* **§508(2026-09-16)**:本文档此前**漏写了"权重来自引擎分片"**这一核心事实,导致(我)误以为
  GPU prefill 必然要额外复制一份权重(+253 GiB/rank)并据此判它"不可用"。**真相**:
  活跃路径零额外主机内存,显存只要两层。相应修正:`XIAOTU_GPUPREFILL_WCOPY` 默认**恒 0**,
  规划器 `GPU_PREFILL_HOST_GIB = 0` ⇒ 在 1M 上下文下 **GPU 预填充 + GPU 投机 + 2 层常驻** 同时可行。
