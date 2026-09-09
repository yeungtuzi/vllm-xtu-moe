# 长 prefill 的逐层 GPU 流式

> 一个可调参数(`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`):当某层的 prefill token
> 数达到阈值时,把该层 MoE 改为**逐层把权重流式搬上 GPU**计算,其余层仍在 CPU。

小 batch decode 时 CPU 带宽足够,但长 prefill(几千 token)在 CPU 上很慢;
把整层专家权重常驻显存又放不下。逐层流式是两者的折中。

---

## 1. 与其它实现的对照

| 实现 | 阈值开关 | 说明 |
|---|---|---|
| vLLM fork(`Lvllmds4-x`) | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | 命中后走标准 vLLM GPU MoE(Marlin);只有部分层标记为 GPU 常驻 |
| ktransformers | `KT_GPU_PREFILL_TOKEN_THRESHOLD`(默认 2048) | 逐层 GPU prefill,自有 Triton MXFP4 kernel |
| **本插件** | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | 不依赖 Marlin(vLLM 的 fp4 Marlin 只支持 NVFP4 group-16,不支持 MXFP4 block-32);自写 Triton 分组 MXFP4 MoE |

## 2. 关键设计

1. **主机侧 K-major 缓存**:把 `[E, 2I, H/2]` 一次性转成 `[E, H/2, 2I]` 并 pin 住
   (按 storage 指针 + offset + shape 缓存),H2D 落进 GPU 即内核所需布局,
   省掉每层一次设备端 transpose。
2. **双 slot 预取流水**:层 L 的 forward 里发起层 L+1 的 H2D(独立 stream + event),
   计算流只等待 `ready` 事件;显存不足以放 ping-pong 缓冲时自动降级为同步路径。
3. **专家并行(TP>1)**:每个 rank 只流式搬 `E/TP` 个专家,路由 id 做 rank 偏移,
   越界置 -1、权重置 0,计算后 `all_reduce`。
4. **H2D 地板**:单卡 137 GiB 权重按 ~25 GB/s 约需 **5.9 s**——这是短 prefill
   不走 GPU 的原因,也是阈值交叉点的来源。
5. **一次性启动成本**:首次 GPU prefill 会构建 pinned K-major 缓存;
   之后走缓存(`cudaHostRegister` 就地 pin 可显著加速构建)。

## 3. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | `384` | 阈值;某层 prefill token 数 ≥ 此值 → 该层走 GPU MoE;`0` 关闭 |
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE` | 未设 | 阈值文件路径;TP>1 时模型在 worker 子进程,客户端改 `os.environ` 不生效,只能用文件做运行期切换 |
| `XIAOTU_GPU_PREFETCH_AHEAD` | `1` | 1 = 双 slot 预取流水;0 = 同步逐层 H2D |
| `XIAOTU_GPU_PREFILL_BM/BN/BK/BH` | `64/64/64/64` | 分组 GEMM tile(默认即扫描最优) |
| `XIAOTU_GPU_PREFILL_STAGES` | `2` | `tl.range` 软件流水级数 |

## 4. 实测

### 4.1 纯 MoE 单层微基准(43 层折算,真实层形状 E=256/H=4096/I=2048/topk=6)

| T | 无重叠 tok/s | 重叠后 tok/s | 重叠后 ms/层 |
|---|---:|---:|---:|
| 2 048 | 252 | 373 | 127.5 |
| 4 096 | 438 | 748 | 127.4 |
| 8 192 | 717 | 1 495 | 127.4 |
| 16 384 | 1 070 | **1 918** | 198.7 |
| 32 768 | 1 414 | **1 994** | 382.2 |
| 49 152 | 1 583 | **2 013** | 567.9 |

- H2D 地板约 127 ms/层,在 T ≥ 8K 时被计算完全掩盖;
- 单卡纯 MoE 跨过 1 500 tok/s 的规模从 41K(重叠前)提前到 **8K**(重叠后)。

### 4.2 TP=2 专家并行等效(E=128/topk=3,每 rank 只算自己那半)

| T | 重叠后 tok/s |
|---|---:|
| 4 096 | 1 495 |
| 8 192 | 2 989 |
| 16 384 | **3 803** |
| 32 768 | **3 979** |

### 4.3 端到端(真实模型,阈值 2048)

| prompt tokens | CPU prefill | GPU prefill(1 卡) | GPU prefill(2 卡 TP=2) |
|---|---|---|---|
| 2 091 | 112.93 s | **5.638 s** | 2.921 s |
| 4 137 | 439.38 s | **5.763 s** | **3.993 s** |
| 8 229 | — | **8.639 s** | — |
| 16 039 | — | **18.237 s** | **13.956 s**(≈1 149 tok/s) |

- 并发聚合(2K 上下文):1 卡 c=1→8 为 371→677 tok/s;2 卡为 716→1016 tok/s;
- TP=2 的 16K 场景里 MoE 只占 4.31 s,注意力/indexer 占 9.65 s(69%)——
  瓶颈已经转移到注意力侧。

## 5. 正确性

`scripts/gpu_prefill_golden.py` 对纯 torch 参考实现验证通过;全 DS-V4 维度
(H=4096/I=2048/E=256)的 RMS 相对误差 **5.6e-3**,即 bf16 输出量化精度
(2⁻⁸ ≈ 3.9e-3)量级。

## 6. 相关文件

| 文件 | 作用 |
|---|---|
| `vllm_xiaotu_moe/gpu_prefill.py` | 内核 + 预取 + 驱动 + torch 参考 |
| `vllm_xiaotu_moe/hybrid_model.py` | GPU-prefill 分支与层间预取编排 |
| `scripts/gpu_prefill_golden.py` | 数值 golden(同步路径) |
| `scripts/bench_gpu_moe_prefetch.py` | 单层微基准(同步 vs 重叠) |
| `scripts/dsv4_prefill_curve.py` | 端到端 TTFT / 并发 / decode 曲线 |
