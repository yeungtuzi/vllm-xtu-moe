# long-prefill GPU 化(环境变量驱动)

> 目标:一个可调参数(环境变量 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`),当某层 prefill
> token 数达到阈值时把该层 MoE 改为逐层流式 GPU 计算,实现 long prefill 的 GPU 加速。
> 实测吞吐见 §结果;完整实验报告见 `docs/EXPERIMENT_REPORT.md`。

## 原理与对照

- **fork(Lvllmds4-x)**:`should_use_gpu_prefill = is_gpu_prefill_layer and
  qlen >= LVLLM_GPU_PREFILL_MIN_BATCH_SIZE`(prod 默认 1024),cudagraph capture 时恒关;
  命中走标准 vLLM GPU MoE(Marlin),**非自定义内核**。`is_gpu_prefill_layer` 只标部分层
  (VRAM 放不下 43 层全 GPU 常驻 → 这就是"显存不足无法全层 gpu prefill")。
- **ktransformers(母系,开源)**:`KT_GPU_PREFILL_TOKEN_THRESHOLD` 默认 2048,layerwise GPU
  prefill,GPU slot 惰性分配,GPU MoE 用自有 Triton MXFP4 kernel。
- **我方实现**:不依赖 Marlin(vLLM 的 `marlin_utils_fp4` 只支持 NVFP4 group-16,
  **不支持 MXFP4 block-32**),自写 Triton 分组 MXFP4 MoE:
  逐层把该层**原始**权重(K-major 字节)从 pinned 主机缓存 DMA 到 GPU → 内核内反量化
  (e2m1 半字节 × e8m0 块标量,block=32)→ gate_up + down 两个分组内核 → 释放。
  只把原始 3.19 GiB/层放进 VRAM(反量化成 bf16 的 12.9 GiB/层会 OOM)。

## 关键设计点

1. **主机侧 K-major 缓存**:`_pinned_kmajor()` 把 `[E,2I,H/2]` 一次性转成 `[E,H/2,2I]`
   并 pin 住(按 storage 指针+offset+shape 缓存,且持有源 storage 防地址复用命中脏数据)。
   H2D 落进 GPU 即内核所需布局 → 去掉每层一次的设备端 transpose。
2. **双 slot 预取流水**(`prefetch_layer` + `PrefetchSlot`):层 L 的 forward 里先发起
   层 L+1 的 H2D(独立 CUDA stream + event),计算流只 `wait_event(ready)`;
   slot 消费完 record `busy`,下次写入同一 slot 前 `wait_event(busy)`。
   显存不足以放 ping-pong 缓冲时自动降级为同步路径并打印提示。
3. **专家并行(TP>1)**:每 rank 只流 `E/TP` 个专家,路由 id 减 rank 偏移、越界置 -1、
   权重置 0,算完 `tensor_model_parallel_all_reduce`。
4. **一次 prefill 的固定成本**:H2D 地板 = 137 GiB / ~25 GB/s ≈ **5.9 s**(单卡)。
   这是短 prefill 不走 GPU 的原因,也是交叉点(~2.8K token)的来源。
5. **一次性启动成本**:首个 GPU prefill 会构建 pinned K-major 缓存(147 GB / TP 份),
   实测 ~2.6 s/层 ≈ **114 s**(单卡,`pin_memory` 仅 1.6 GB/s);之后走缓存。
   可用 `cudaHostRegister` 就地 pin(实测 0.37 s / 2 GiB,快 3.7×)或并行预建来压缩。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | `0`(关闭) | 阈值;某层 prefill token 数 ≥ 此值 → 该层走 GPU MoE |
| `XIAOTU_GPU_PREFETCH_AHEAD` | `1` | 1=开启双 slot 预取流水;0=同步逐层 H2D |
| `XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE` | 未设 | 阈值文件的路径;设了就以文件内容为准(每层读一次)。**TP>1 时模型跑在 worker 子进程里,客户端改 `os.environ` 不生效,只能靠这个文件做运行期切换** |
| `XIAOTU_GPU_PREFILL_BM/BN/BK/BH` | `64/64/64/64` | 分组 GEMM tile(T=16384 扫描 48 组,默认即最优) |
| `XIAOTU_GPU_PREFILL_STAGES` | `2` | `tl.range` 软件流水级数 |

测试服脚本 `scripts/serve_8071.sh` 已携带该开关(默认 0;需要时设 2048 等)。

## 文件

- `vllm_xiaotu_moe/gpu_prefill.py`——内核 + 预取 + 驱动 + torch 参考(测试)
- `vllm_xiaotu_moe/hybrid_model.py`——`CpuXiaotuMoE.forward` 的 GPU-prefill 分支 + 层间预取编排
- `scripts/gpu_prefill_golden.py`——同步路径 golden(含 `TEST_FULL=1` 全维度)
- `scripts/gpu_prefill_prefetch_test.py`——slot/ring 路径 golden
- `scripts/bench_gpu_moe_prefetch.py`——单层微基准(seq vs 重叠;E=256 与 E=128/topk=3 等效 EP)
- `scripts/sweep_gpu_moe_tiles.py`——tile 扫描
- `scripts/dsv4_prefill_curve.py`——端到端 TTFT/并发/decode 曲线
- `scripts/dsv4_longprefill_test.py`——最早的 CPU vs GPU 单点对比

## 结果

### 纯 MoE 单层微基准(43 层折算,真实层形状 E=256/H=4096/I=2048/topk=6)

| T | 无重叠 tok/s | **重叠后 tok/s** | 重叠后 ms/层 |
|---|---:|---:|---:|
| 2048 | 252 | 373 | 127.5 |
| 4096 | 438 | 748 | 127.4 |
| 8192 | 717 | 1495 | 127.4 |
| 16384 | 1070 | **1918** | 198.7 |
| 32768 | 1414 | **1994** | 382.2 |
| 49152 | 1583 | **2013** | 567.9 |

- H2D 地板 127 ms/层,在 T ≥ 8K 被计算完全掩盖;
- 单卡纯 MoE 跨过 1500 tok/s 的规模从 41K(重叠前)提前到 **8K**(重叠后)。

### TP=2 专家并行等效(E=128/topk=3,每 rank 只算自己那半)

| T | 重叠后 tok/s |
|---|---:|
| 4096 | 1495 |
| 8192 | 2989 |
| 16384 | **3803** |
| 32768 | **3979** |

### 端到端(真实模型,MAXLEN=MAX_NBT=16384,SINGLECOPY=1,阈值 2048)

| prompt tokens | CPU prefill | GPU prefill(1 卡) | GPU prefill(2 卡 TP=2 EP) |
|---|---|---|---|
| 2091 | 112.93 s(18.5 tok/s) | **5.638 s(370.9 tok/s)** | 2.921 s(715.9 tok/s)¹ |
| 4137 | 439.38 s(9.4 tok/s) | **5.763 s(717.9 tok/s)** | **3.993 s(1036.2 tok/s)** |
| 8229 | — | **8.639 s(952.5 tok/s)** | — |
| 16039 | — | **18.237 s(879.5 tok/s)** | **13.956 s(1149.3 tok/s)** |

¹ 2K 那一行的 2 卡数字来自并发 c=1 的测量。并发聚合:1 卡 c=1→8 为
371→677 tok/s;2 卡为 716→1016 tok/s。

- **瓶颈已经转移**:TP=2 的 16K 里 MoE 只剩 4.31 s,注意力/indexer 占 9.65 s(69%)。
- 完整分析、硬件核验、upstream 计划见 `docs/EXPERIMENT_REPORT.md`。

### 正确性

`gpu_prefill_golden.py`(同步路径)与 `gpu_prefill_prefetch_test.py`(slot/ring 路径)
对纯 torch 参考实现全部通过;全 DS-V4 维度(H=4096/I=2048/E=256)RMS 相对误差
**5.6e-3**,即 bf16 输出量化精度(2^-8 ≈ 3.9e-3)。

---
## 修订记录

- **2026-09-09(第 3 版)** — §结果 的端到端表换成修复两个致命 bug 后的实测(单卡
  5.64–18.24 s;双卡 2.92–13.96 s),补并发/瓶颈转移结论;环境变量补
  `XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE`(客户端 os.environ 改不动 worker 进程)。
  依据:`report/curve_tp1*.jsonl`、`report/curve_tp2.jsonl`。
- **2026-09-09(第 2 版)** — 重写:①删除已废弃的 `XIAOTU_GPU_PREFILL_EXPERT_SLICE`
  (改为内核内反量化,不再按专家切片展开 bf16);②补 §关键设计点(主机侧 K-major、
  双 slot 预取流水、EP、H2D 地板、一次性启动成本);③补环境变量表(BM/BN/BK/BH/STAGES/
  PREFETCH_AHEAD);④结果表换成重叠前/后对比 + TP=2 等效 EP 实测 + 正确性 RMS 误差;
  ⑤文件清单更新。依据:`results.txt` 2026-09-09「长 prefill 第 3 轮」、
  `report/moe_micro*.json`、`report/golden.txt`。
- **2026-09-09(第 1 版)** — 首版 + 第 1/2 轮记录。其中「CPU 85 s → GPU 10.97 s = 7.8x」
  与「约 14x vs CPU」是**不公平对比**(CPU 基线取自修复前且机器高负载时),已作废,
  以本版 §结果 与 `EXPERIMENT_REPORT.md` §5 的同进程实测曲线为准。
