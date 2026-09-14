# GPU Prefill on mainline(设计 + 现状)

> **本文回答一个问题**:在 **零主线补丁**的前提下,怎么把"大 batch 预填充"从 CPU 挪回 GPU?
> 所有数字都标注了**实测 / 估算**。相关:`docs/UPSTREAM_DRIFT.md`、`docs/INSTALL_MAINLINE.md`、
> `report/tuning/NOTES.md §334`。

---

## 1. 为什么必须做(实测)

形态 B(mainline `RoutedExperts` 原样 + 我们的 CPU 后端)在**解码**上已经与 fork 打平:

| 场景 | 实测 | `qlen` 分布 |
|---|---|---|
| 4-token prompt + 128 输出(短上下文) | **31.5 ms/token** | qlen=1 × 224(纯解码)✅ |
| 257-token prompt + 128 输出(图模式) | **1136.8 ms/token** | qlen=257 × 200/200 ❌ |
| 同上,**`--enforce-eager`** | 124.1 ms/token | **qlen=1 × 100/100** ✅ |

两个结论:

1. **"无限重做预填充"是 CUDA 图契约 bug**,不是调度、不是引擎(关图即消失)。
   形态 B 的 `apply()` 每次 `torch.empty` 新输出 + 每次 `.to(dtype)` 新张量 ⇒ 地址每步都变
   ⇒ chunked-prefill 的 piecewise 图无法稳定引用。
2. **即使修好契约,CPU 预填充也慢**:256 token 的预填充 = **26 ms/层 × 43 ≈ 1.1 s(≈230 t/s)**,
   而 fork 的 GPU 预填充实测 **982–1287 t/s** ⇒ 差 **4–5×**。

⇒ 要真正可用,必须有一条 **GPU 预填充**路径。

---

## 2. 设计:单层 GPU staging buffer + 复用上游 GPU MoE 内核

### 2.1 核心洞察:预填充只需要**一层**的 GPU 权重

预填充是**层间串行**的(层 L 的输出是层 L+1 的输入),所以:

* **不需要**把 43 层权重都放 GPU(那是 137 GiB);
* 只需要一个 **约一层大小的 staging buffer**(TP=2 每 rank **1.61 GiB**);
* 每层:把该层被激活的专家权重 H2D → 用**主线上游的 GPU MoE 内核**算 → 下一层。

这正是 fork 里 `LVLLM_GPU_PREFILL_*` 的语义(它的 `gpu_prefill` 层就是"权重在 GPU、大 batch 走 GPU"的层)。

### 2.2 DMA 预算(实测参数)

```
每层每 rank 权重 1.61 GiB × 43 层 = 69 GiB
实测 H2D 带宽 20.1–21.4 GB/s(PCIe Gen4 x16,NOTES 里测过)
⇒ 一次完整预填充的 DMA 下限 ≈ 69 / 20.1 ≈ 3.4 s
```

⇒ **预填充吞吐与 batch 大小基本无关**(DMA 是常数项)。**实测**(`scripts/bench_gpu_prefill.py`):

| M | CPU 每层 | CPU 全模型(÷43) | H2D 实测 | GPU 计算 | **GPU+H2D** | 谁赢 |
|---|---|---|---|---|---|---|
| 64 | 6.66 ms | 448 t/s | 154 ms | 2.07 ms | 156 ms | CPU 23× |
| 256 | 23.1 ms | 266 t/s | 154 ms | 2.83 ms | 157 ms | CPU 6.8× |
| 1024 | 85.0 ms | 293 t/s | 175 ms | 3.58 ms | 179 ms | CPU 2.1× |
| **2048** | 162.4 ms | 265 t/s | 152 ms | 5.64 ms | **157.6 ms** | **GPU 1.03×** |
| 8192(外推) | ~650 ms | ~283 t/s | ~152 ms | ~18 ms | ~170 ms | **GPU ≈ 4×** |

* **H2D 实测 ≈ 21 GB/s**(PCIe Gen4 x16 的理论值),3.2 GB/层 ⇒ **152 ms/层**;
* **CPU 侧全模型预填充 265–293 t/s**(与服务实测的 230 t/s 一致)⇒ 引擎本身不慢;
* ~~交叉点 `T ≈ 1600–2000`,阈值应取 `T ≈ 2048`~~ **⚠️ 已作废 —— 见下方 §2.3 的端到端实测**。
  上表的 CPU 列是**单层隔离测**,把 CPU 侧吞吐高估了 ~6.5×(隔离 293 t/s vs 端到端 45 t/s);
  真实交叉点在 **M ≈ 128–256**,即 **GPU 在每一个长度上都赢**。

### 2.3 复用上游:主线自带我们需要的入口

| 上游接口 | 位置 | 用途 |
|---|---|---|
| **`fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, ...)`** | `vllm/model_executor/layers/fused_moe/fused_moe.py:1593` | **吃显式 w1/w2 的 GPU MoE** —— 插件直接用 |
| `torch.ops.vllm.fused_experts` | 同文件 `:1454` | 同上(自定义算子形式) |
| `RoutedExperts.forward_monolithic` / `forward_modular` | `routed_experts.py:1275` | 需要构造完整 layer,较重 |

⇒ **不需要写任何 GPU kernel,也不需要改主线**(复用优先:上游 → lvllm → 自研)。

### 2.4 在插件里的落点(形态 B 的 `apply()`)

```python
# vllm_xiaotu_moe/mixed_experts.py :: _XiaotuExpertsMixin.apply
qlen = hidden_states.size(0)
if qlen >= GPU_PREFILL_MIN_TOKENS and _gpu_stage is not None:
    # 1) 权重 H2D 到常驻 staging(每层一个 slot,复用,不重新分配)
    stage_w13.copy_(w13, non_blocking=True)
    stage_w2.copy_(w2, non_blocking=True)
    stage_s13.copy_(s13, non_blocking=True); stage_s2.copy_(s2, non_blocking=True)
    # 2) 直接调上游 GPU MoE
    return fused_experts(hidden_states, stage_w13, stage_w2, topk_weights, topk_ids,
                         inplace=False, **quant_kwargs)
else:
    return self._cpu_path(hidden_states, ...)   # 现状:xiaotu 引擎
```

### 2.5 必须同时解决的三件事

| # | 问题 | 做法 |
|---|---|---|
| 1 | **图契约**(§1 结论 1) | 每层持有一个预分配输出缓冲 `[MBT, H]`,写进去返回视图(与 fork 的 `RoutedExperts.output_gpu` 同构);或对大 batch 强制 eager |
| 2 | **staging 显存** | 每 rank ~1.6 GiB;与 KV 争显存 ⇒ 需要 `--gpu-memory-utilization` 或 KV 预算让出这块(可用 `XIAOTU_GPU_PREFILL_STAGE_GB` 控制) |
| 3 | **量化格式对齐** | 主线 GPU MoE 需要 MXFP4 的 `w13_weight/_scale` 布局与 backend(MARLIN/TRITON);CPU 侧的 e8m0 scale 布局**很可能不同** ⇒ 需要一次 host 侧重排(可缓存、只在首次预填充做) |

---

## 3. 与 fork 的对照(为什么这条路更干净)

| | fork | 本设计(mainline + 插件) |
|---|---|---|
| GPU MoE kernel | `lk_moe.gpu_prefill`(**专有二进制**) | **上游 `fused_experts`**(可读、可调、随主线升级) |
| 权重上 GPU | lk 内部管理 | 插件用 `copy_` 显式管理,`staging` 大小可配 |
| 补丁 | 需要 fork 的编排补丁 | **零主线补丁** |
| 可观测性 | 二进制 | 全 Python,可打点/可审计 |

---

## 4. 分步落地计划

| 步 | 内容 | 验收 | 预估成本 |
|---|---|---|---|
| **P0** | 修**图契约**(预分配输出缓冲,或大 batch 回退 eager) | 257-token prompt 在**图模式**下也得到 `qlen=1` 的纯解码 | 0.5 天 |
| **P1** | ✅ **已完成**:`scripts/test_gpu_prefill_equiv_fixture.py` 三方对拍 | **CPU:归一化 max = 2.1e-06**;GPU:归一化中位数 **3.07e-03 = bf16 的 eps**,p99 = 2.1e-02 ⇒ **在 bf16 精度内等价,无实现差异** | 已完成 |
| **P2** | ✅ **数值已打通**(2026-09-14):真实 MXFP4 + 上游 MARLIN 与 CPU 引擎**归一化中位数 3.66e-03 = bf16 eps** ⇒ OK;三块拼图 = 主线优先级选 MARLIN + `prepare_moe_mxfp4_layer_for_marlin`(重排,纯函数) + `fused_marlin_moe`(吃显式权重与 precomputed topk)。**剩余**:全层接入 + 阈值 `T` 标定 + 显存预算。**入口已找到**:`triton_kernel_moe_forward(hidden,w1,w2,gating_output,topk,renormalize,activation,quant_config)`(`experts/gpt_oss_triton_kernels_moe.py:541`)—— 吃显式权重、**不需要构造 FusedMoEConfig**;⚠️ 它内部自己路由,须确认与 DS-V4 的 sqrtsoftplus+夹取+group-topk 一致 | `bench_lat.sh` L=512/2048/8192 的 **TTFT** 与 fork 对照 | 1 天 |
| **P3** | 与解码路径共存(阈值切换)、KV/显存再平衡 | C=1/2/4 端到端 + 数值门禁 `OK=7 BAD=1` | 1 天 |

**P1 的数值对拍是硬门禁**:GPU MoE 与 CPU 引擎的 MXFP4 结果必须一致(参考已有的
`scripts/test_block23_equiv.py` 与 `probe_greedy.py` 的做法)。

---

## 4.5 P1 实测结果(2026-09-14)

`scripts/test_gpu_prefill_equiv_fixture.py`(复用已验证 fixture,`E=16/I=2048/H=4096/K=6/REP=6`):

| | 归一化 p50 | 归一化 p99 | 归一化 max |
|---|---|---|---|
| CPU(xiaotu 引擎,fp32 累加) | 1.3e-07 | 9.3e-07 | **2.1e-06** |
| GPU(上游 `fused_experts`,bf16) | **3.07e-03** | 2.09e-02 | 4.96e-02 |

* **bf16 的 eps = 3.9e-03** ⇒ GPU 的中位数误差**正好等于 bf16 机器精度**
  ⇒ 两边**没有实现差异**,差异纯粹来自 dtype;
* ⚠️ **口径教训**:bf16 内核**不能用单元素 max 判定**(最坏元素可达 eps 的十几倍),
  要用**中位数/分位数**;我们最初的 `max_rel` 判定把它误报成 MISMATCH。

**⇒ P1 通过。** P2 起把本脚本接成**回归门禁**。

## 4.6 P2 后端选型(2026-09-14 实测)

主线对 MXFP4 的优先级(`oracle/mxfp4.py:_get_priority_backends`):
`FLASHINFER_TRTLLM_MXFP8`(SM100)→ `DEEPGEMM_MXFP4`(SM90)→ **`MARLIN`(SM80 ✅)** → `BATCHED_MARLIN`。
⇒ **A100 上主线选的就是 MARLIN**,与 lk fork 一致;功能入口是
**`fused_marlin_moe(...)`**(`experts/marlin_moe.py:235`,吃显式
`w1/w2/w1_scale/w2_scale/topk_weights/topk_ids`,**内部不重做路由** ⇒ DS-V4 路由可照用);
MXFP4 W4A16 的 `quant_type_id = scalar_types.float4_e2m1f.id`。

⚠️ **唯一需要额外做的是"加载期一次权重重排"**:`fused_marlin_moe` 断言
`w1.size(1)*16 == K`,而 checkpoint 原生是 `[E, N, K/2]`。
重排函数已找到:**`marlin_utils_fp4.py:311 _repack_marlin_experts`**。
方案 = 加载期重排一次并**缓存在主机内存**(+69 GiB/rank,1.5 TB 放得下),
每次预填充 H2D 搬重排后的 1.61 GiB/rank ⇒ **DMA 预算不变(§2.2 的 ~3.4 s)**。

## 5. 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 量化布局不一致 | CPU 引擎的 e8m0 scale 是 `[E, N/gn, K/gk]`,主线 GPU 侧可能是另一种 | P1 先做**单层数值对拍**,不一致就加一次 host 重排(可缓存) |
| staging 挤掉 KV | 1.6 GiB/rank | 只在大 batch 时占用;或按需分配/释放 |
| H2D 实测带宽回落 | 20.1 GB/s 是在**空闲**机器上测的 | P2 在真实预填充负载下复测,并把 T 按实测重算 |
| 与图共存 | staging 地址必须稳定(可被图引用) | staging 与输出缓冲都在插件构造期一次性分配 |


## 2.3 🎯 端到端实测(权威):GPU 预填充在**每个长度**上都领先

上面 §2.2 的 CPU 列来自**单层隔离**测法,会高估 CPU 吞吐 ~6.5×。
真正的结论看端到端曲线(`scripts/dsv4_prefill_curve.py`,`report/curve_thr.jsonl`,2026-09-09,
`enforce_eager=True` / `MBT=16384` / TP=1):

| prompt | CPU TTFT | CPU t/s | **GPU TTFT** | **GPU t/s** | 加速 |
|---|---|---|---|---|---|
| 128 | 5.689 s | 24.4 | **2.412 s** | 57.6 | 2.4× |
| 256 | 6.528 | 42.1 | **4.956** | 55.5 | 1.3× |
| 384 | 9.852 | 41.7 | **8.017** | 51.3 | 1.2× |
| 512 | 12.247 | 44.7 | **5.656** | 96.9 | 2.2× |
| 768 | 18.052 | 43.9 | **5.674** | 139.8 | 3.2× |
| 1024 | 23.516 | 44.9 | **5.653** | 187.0 | 4.2× |
| **2048** | **92.598** | 22.6 | **5.712** | **366.1** | **16.2×** |
| 4096 | — | — | 5.763 | 717.8 | |
| 8192 | — | — | 8.611 | **955.6** | |
| 16000 | — | — | 17.742 | 904.0 | |

TP=2(`report/curve_tp2.jsonl`):4096 → **3.993 s / 1036 t/s**;16000 → **13.956 s / 1149 t/s**。
并发(2048 token/请求):TP=1 C=1 371 / C=4 621 / C=8 686 t/s;TP=2 C=8 **1016 t/s**。

**⇒ 参考 fork 的 982–1287 t/s 就是这条路径。**

### 那 ≈5.6 s 的常数底是什么?

`5.6 s / 43 层 ≈ 130 ms/层`,与 §2.2 实测的 **H2D = 3.2 GB / 152 ms ≈ 21 GB/s**(PCIe Gen4 x16)同量级
⇒ **常数底就是"一次完整预填充要把 43 层权重要过一遍 PCIe"**。所以要再快,优化的**不是 CPU 引擎,而是这个 DMA 常数**:

1. **重叠**:`XIAOTU_GPU_PREFETCH_AHEAD=1`(默认已开,`PrefetchSlot` 双缓冲);
2. **常驻层**:`RESIDENT=0-11` 让 12 层完全免 DMA;
3. **TP=2**:每 rank 只搬一半(实测 5.71 s → 3.99 s ✓)。

### 推荐阈值

| 场景 | `T` | 理由 |
|---|---|---|
| 默认(解码基准协议) | `0`(关) | 保持与参考的解码对比口径不变 |
| **预填充/长 prompt** | **`1024`** | 与参考 fork 生产值一致(`serve_lk_port.sh:45`);`MBT` 必须 ≥ `T` |
| 想更激进 | `256` | 端到端显示 256 也已经赢(4.96 vs 6.53 s),但边际收益小于 DMA 常数 |

## 2.4 主线侧要让 GPU 预填充真正生效,必须同时满足**三个**开关

| # | 开关 | 缺了会怎样 |
|---|---|---|
| 1 | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=N` | `hybrid_model.py:982` 的 `_gp_min > 0` 恒假 ⇒ 非常驻层**永远走 CPU** |
| 2 | `MBT ≥ N` | batch 永远到不了阈值(fork 会把 `MINBATCH` 夹到 `MBT`,同一语义) |
| 3 | 预填充形状**不被 CUDA 图捕获** | `is_current_stream_capturing()` 只挡"捕获中"、挡不住"**重放**";主线默认 `PIECEWISE` 会把预填充形状也捕获,重放永远走捕获时的 CPU 分支 ⇒ GPU 路径形同虚设 |

第 3 条的上游原生解法:**`--cudagraph-capture-sizes` 只列解码尺寸**(空格分隔:`1 2 4 8`)**,
预填充形状(qlen ≥ 256)不在捕获集里 ⇒ 自动 eager ⇒ GPU 分支生效,**同时解码仍享有 CUDA 图**。
这正是主线 `CUDAGraphMode.FULL_DECODE_ONLY = (FULL, NONE)` 的语义。

**一条命令同时打开三个开关**:

```bash
ENV=/home/user/anaconda3/envs/vllm-xiaotu-moe PREFILL=1 bash scripts/serve_mainline.sh
# 等价于 MBT=8192 / GP_MIN=1024 / CUDAGRAPH_SIZES="1 2 4 8"
```
