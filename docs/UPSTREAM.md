# 需要上游配合的改动

本插件在**原生 vLLM** 上即可运行:混合模式所需的几处主线行为由
`vllm_xiaotu_moe/mainline_shims.py` 以 monkey-patch 方式提供(见
[`ARCHITECTURE.md`](ARCHITECTURE.md) §5)。本文列出这些改动本身,以及可以
上游化的部分——它们对主线同样有价值,不依赖本插件。

---

## 1. 混合模式的基础设施(可上游化)

### 1.1 `VLLM_EXPERTS_LOAD_DEVICE=cpu`(专家权重放 host)

**问题**:MoE 模型的专家权重若超过单卡显存,主线在**构造期**就会 OOM
(例如 DeepSeek-V4-Flash 的专家约 137 GiB)。主线现有的 offload 机制
(`model_executor/offloader/prefetch.py`)只能搬运**已构造**的权重,无法避免
构造期的分配。

**改动**(3 个文件,约 +60 行):

| 文件 | 改动 |
|---|---|
| `vllm/envs.py` | 新增 `VLLM_EXPERTS_LOAD_DEVICE`(默认 `gpu`,`cpu` 为可选) |
| `fused_moe/routed_experts.py` | 开关打开时在 `torch.device("cpu")` 下调用 `create_weights()` |
| `fused_moe/oracle/mxfp4.py` | 开关打开时优先选 CPU 后端,并跳过 AMX 重打包 |

默认行为不变(`gpu` 时走原路径)。

### 1.2 oracle 优先选 CPU 后端(各格式对称)

`oracle/{fp8,int_wna16,unquantized}.py` 需要在混合模式下把 CPU 后端前置
(与 1.1 中 mxfp4 的改动对称)。本插件目前用 shim 实现。

### 1.3 跳过 CPU 后端的 AMX 重打包

`cpu_moe.prepare_{fp8,mxfp4,int4}_moe_layer_for_cpu()` 会把权重重打包成
AMX 内核需要的布局,这会破坏"原始权重直传第三方 CPU 引擎"的用法,并且依赖
`torch.ops._C.convert_weight_packed`(部分部署未编译)。混合模式下应跳过。

### 1.4 量化方法应通知 experts 后端

`Fp8MoEMethod.process_weights_after_loading()` 与 WNA16 对应方法**没有**调用
`experts.process_weights_after_loading(layer)`,而
`UnquantizedFusedMoEMethod` 会调用。依赖该钩子获取 layer 引用的后端
(包括 OOT 后端)因此拿不到上下文。建议统一调用。

## 2. 主线 CPU MoE 后端的两个正确性问题

这两个问题与第三方后端无关,是主线自带 CPU 内核的行为,已在真实模型上复现:

### 2.1 路由硬编码为 softmax

`fused_moe/experts/cpu_moe.py` 的 `apply()` 中:

```python
topk_weights, topk_ids = select_experts(..., scoring_func="softmax", ...)
```

对使用 sigmoid + `noaux_tc`(GLM 系)或 `sqrtsoftplus`(DeepSeek-V4)的模型,
这会**选错专家**。建议改为使用主线已有的 router 对象
(`create_fused_moe_router`),与本插件做法一致。

### 2.2 忽略 `swiglu_limit` / `alpha` / `beta`

`moe_config.swiglu_limit` 只被 `activation.py` / `b12x.py` / `utils.py` 使用,
CPU 内核完全没有应用它。GLM-5.x、DeepSeek-V4、MiniMax、HY-V4 等模型训练时
都带有该 clamp,忽略它会改变专家输出。建议在 CPU 内核的 gated 激活里实现
`clamp(gate,max=L) * sigmoid(alpha*clamp(gate,max=L)) * (clamp(up,±L)+beta)`。

## 3. 已提交的上游 PR(快照见 `patches/`)

| PR | 范围 | 规模 | 说明 |
|---|---|---|---|
| **PR1** | `VLLM_EXPERTS_LOAD_DEVICE=cpu` + MXFP4 oracle | 3 文件 +59/−2 | §1.1 |
| **PR2** | DeepSeek-V4 `o_proj` 在 SM8.x 的 fp8 einsum 修复 | 4 文件 +422/−12 | Ampere 上 Marlin 重打包破坏 `o_proj` 的直读权重;并补可移植 Triton fp8 einsum |
| **PR3** | DeepSeek-V4 在 Ampere/Ada 的可移植 Triton 回退 | 21 文件 +6350/−119 | 稀疏 MLA / DeepGEMM / MQA / `float8e4nv` dtype 等 SM8.x 回退 |

- PR 描述(目标 / 问题 / 设计 / 算法 / 实测 / 测试):`patches/upstream/pr{1,2,3}_body.md`
- 补丁快照:`patches/mainline_sm80_mixed_mode.patch` 等
- RFC 草稿(CPU-offload MoE 的逐层 GPU prefill):
  `patches/rfc_layerwise_gpu_prefill.md`

> PR2/PR3 的 SM80 移植代码来自 vLLM fork
> [`Lvllmds4-x`](https://github.com/guqiong96/Lvllmds4-x)(Apache-2.0),
> 按其许可保留 SPDX 署名;详见 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

## 4. 给维护者的建议顺序

1. §2.1 / §2.2(小而独立、对主线普遍有益的 bug 修复);
2. §1.1 + §1.2 + §1.3(混合模式基础设施,需要一个小设计讨论);
3. §1.4(一行调用,依赖 1.1 的动机);
4. PR2 → PR3(Ampere 上的 DeepSeek-V4 支持)。
