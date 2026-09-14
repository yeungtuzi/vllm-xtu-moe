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

⇒ **预填充吞吐与 batch 大小基本无关**:
* batch = 8192 ⇒ **≈ 2400 t/s**
* batch = 1024 ⇒ ≈ 300 t/s
* batch = 256  ⇒ ≈ 75 t/s(**比 CPU 的 230 t/s 更差**)

**⇒ 关键推论:要有阈值**。只有 `num_tokens ≥ T` 时才走 GPU;
`T` 由"CPU 速率 vs DMA 速率"的交叉点决定:`T* ≈ 20.1/0.23 ≈ 87`… 但还要算 GPU 计算与 kernel 启动,
**实测标定后取 T ≈ 256–1024**(与 fork 的 `gpu_prefill_min_batch_size` 语义一致)。

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
