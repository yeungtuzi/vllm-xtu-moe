# 架构:混合模式与主线集成

本文说明本插件如何把 CPU 专家计算接进 vLLM 主线,以及为什么需要那些集成点。

---

## 1. 目标与约束

- **目标**:专家权重放 CPU、注意力/路由/KV 留 GPU;任意使用主线 `FusedMoEFactory`
  的 MoE 模型都能受益,而不是只支持某一个模型。
- **约束**:
  - 不 fork vLLM(走官方插件入口 `vllm.general_plugins`);
  - 主线自带的量化 CPU 内核要求 Intel AMX,本项目面向**无 AMX 的 x86**
    (AMD EPYC 等),因此 CPU 计算必须由本插件提供;
  - 数值正确性优先:任何"不确定语义"的情形宁可直接报错,也不静默给出错误结果。

## 2. 主线提供的接口

```
FusedMoEFactory(...)
  └─ oracle.select_*_moe_backend(...)      # 按格式/平台/量化 key 选后端
       ├─ GPU 后端(FlashInfer / Triton / DeepGEMM / Marlin ...)
       └─ CPU 后端(CPUExpertsMxfp4 / CPUExpertsFp8 / CPUExpertsInt4 / CPUUnquantizedExperts)
  └─ quant_method.apply_monolithic(...)    # 权重与 router_logits 交给后端
```

- 专家权重由 `RoutedExperts.__init__` 通过 `quant_method.create_weights()` 创建;
- 后端是一个 `mk.FusedMoEExpertsMonolithic` 子类,`apply()` 拿到
  `hidden_states / w13 / w2 / router_logits` 等;
- oracle 对 CPU 后端类采用**惰性导入**,因此替换模块属性即可让 oracle 选中替代实现。

## 3. 本插件的三个部件

| 部件 | 位置 | 作用 |
|---|---|---|
| **通用 CPU experts 后端** | `vllm_xiaotu_moe/mixed_experts.py` | 按量化格式注册 `XiaotuCPUExperts*`,把计算转给内置引擎 |
| **主线集成补丁(shims)** | `vllm_xiaotu_moe/mainline_shims.py` | 在原生 vLLM 上提供混合模式所需的 4 处行为 |
| **长 prefill GPU 路径** | `vllm_xiaotu_moe/gpu_prefill.py` | 阈值触发,逐层把专家权重流式搬上 GPU 计算 |

## 4. 通用 CPU experts 后端

### 4.1 格式注册表

| 权重格式 | 基线类(主线) | 引擎 | 说明 |
|---|---|---|---|
| 无量化 BF16 | `CPUUnquantizedExperts` | `MOE_BF16` | 权重 `[E,2I,H]` / `[E,H,I]` |
| FP8 e4m3 block 128 | `CPUExpertsFp8` | `MOE_FP8` | 块缩放由插件归一化为 fp32 |
| MXFP4 | `CPUExpertsMxfp4` | `MOE_MXFP4` | e8m0 缩放按字节直传 |
| INT4 / WNA16 | `CPUExpertsInt4` | `MOE_WNA16` | 组大小/零点适配进行中 |

`register_mixed_cpu_backend()` 在混合模式下把 `cpu_moe` 模块里的这些类替换为
`XiaotuCPUExperts*`,并把 `_supports_current_device()` 放宽为
"混合模式 + x86"(主线原版要求 `current_platform.is_cpu()` 且具备 AMX)。

### 4.2 路由

monolithic 后端的 `apply()` 只拿到 `router_logits`,因此路由必须自己完成。
主线把路由拆成 `fused_moe/router/*` 下的一组对象;本插件**直接复用这些对象**
(`create_fused_moe_router`),从而覆盖:

- softmax、sigmoid、`sqrtsoftplus`;
- grouped top-k、`noaux_tc`(带 `e_score_correction_bias`);
- 自定义路由函数。

> 主线自带的 CPU 后端在 `apply()` 里硬编码了 `scoring_func="softmax"`,
> 对 GLM(sigmoid + `noaux_tc`)和 DeepSeek-V4(`sqrtsoftplus`)会选错专家;
> 本插件通过复用主线 router 避免了该问题。

### 4.3 激活

引擎实现的是 **packed 布局**的 gated 激活:

```
out = clamp(gate, max=L) * sigmoid(alpha * clamp(gate, max=L)) * (clamp(up, ±L) + beta)
```

`L / alpha / beta` 来自 `moe_config.swiglu_limit / swiglu_alpha / swiglu_beta`,
与主线 `silu_and_mul_with_clamp` 语义一致。因此支持主线的 `SILU` 与
`SWIGLUOAI_UNINTERLEAVE`(两者都是 packed);`SWIGLUOAI`(gate/up 在 w13 内交错,
如 gpt-oss)会被 `_supports_activation` 拒绝,让 oracle 选择别的后端。

### 4.4 显式拒绝而非静默错算

以下情形直接 `raise NotImplementedError`:

- 非 packed 激活(见上);
- 专家并行(`expert_map`);
- `apply_router_weight_on_input`;
- 权重 dtype 与后端不匹配(例如把 FP16 权重交给 BF16 引擎)。

## 5. 主线集成补丁(shims)

混合模式需要主线配合 4 件事,它们在上游尚未合并,因此由插件在导入时以
**幂等 monkey-patch** 提供(`XIAOTU_MAINLINE_SHIMS=0` 可关闭):

| # | 需要的行为 | 插件做法 | 为什么必需 |
|---|---|---|---|
| 1 | 专家权重在 CPU 上创建 | 给每个 `FusedMoEMethodBase` 子类的 `create_weights()` 套一层 `torch.device("cpu")`(对后续才导入的量化类用 `__init_subclass__` 自动补上) | 几百 GB 专家权重若建在显存,构造期即 OOM |
| 2 | oracle 优先选 CPU 后端 | 包装 `oracle/{fp8,mxfp4,int_wna16,unquantized}.py` 的 `_get_priority_backends()`,把 CPU 后端移到队首 | GPU 平台默认不会选 CPU 后端,否则权重在 host、内核在 device |
| 3 | 跳过 CPU 后端的 AMX 重打包 | 混合模式下让 `cpu_moe.prepare_{fp8,mxfp4,int4}_moe_layer_for_cpu()` 原样返回 | 重打包会破坏引擎需要的原始布局,且依赖可能未编译的 `torch.ops._C.convert_weight_packed` |
| 4 | 通知 experts 后端 | 包装 `process_weights_after_loading()`,在主线处理完后调用后端的同名钩子 | fp8/wna16 的量化方法不会调用该钩子,后端就拿不到 layer 引用 |

这些补丁都是幂等的:在已合并相应改动的 vLLM 上重复应用等于 no-op。
上游对应改动见 [`UPSTREAM.md`](UPSTREAM.md)。

## 6. 数据流

```
每个 MoE 层(前向):
  hidden_states(GPU) ──► 主线 router ──► topk_weights/topk_ids(GPU)
                                   │
                                   ▼
        engine.cpu_decode(stream, qlen, topk, hidden, ids, weights, out)
                                   │
     ┌─────────────────────────────┴──────────────────────────────┐
     │ D2H(hidden/ids/weights) → CPU 引擎计算 → H2D(out)          │
     │ 由 CUDA host-function 节点完成,可被 CUDA graph 捕获         │
     └────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                            MoE 输出(GPU) ──► 残差/下一层
```

- 引擎在构造时把权重快照到自己的缓冲(按 NUMA 节点分片),避免后续
  `clean_weights_after_loading` 释放原始参数导致悬空指针;
- 检查点布局与引擎布局不同的格式(INT4/WNA16)在**引擎构造时一次性重排**:
  GPTQ 的 `w13 [E, K/8, 2I] int32` → `[E, 2I, K/2]` u8、缩放 `[E, K/g, N]` → `[E, N, K/g]`,
  并以 `groupN=1 / groupK=group_size` 构建引擎;重排后的张量用显式 `empty+copy_`
  分配,避免 `.contiguous()` 链留下"存储归属别处"的视图;
- 每个层一个引擎实例,引擎之间共享同一个 NUMA 线程池;
- `XIAOTU_MOE_SINGLECOPY=1` 时权重只保留一份并按 NUMA 节点分片,
  每个核心只读本节点数据。

## 7. 长 prefill 的 GPU 路径

小 batch decode 时 CPU 带宽足够;但长 prefill(几千 token)在 CPU 上很慢。
`gpu_prefill.py` 提供按阈值切换的路径:当某层的 prefill token 数达到
`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` 时,该层把专家权重按 K-major 布局
从 pinned host 内存**逐层流式**搬上 GPU,用 grouped Triton GEMM(含 MXFP4 在线
反量化)计算后再释放。设计细节与实测见 [`GPU_PREFILL.md`](GPU_PREFILL.md)。

## 8. 线程与 NUMA

- 引擎自带线程池(按 NUMA 节点分组),不依赖 OpenMP;
  `XIAOTU_MOE_THREADS` 可限制线程数;
- **小批量(decode)按 N 行切片**:一个 token 的 GEMV 会被切成若干行块铺满线程池,
  并把参与的 worker 数收敛到与工作量匹配的规模(池屏障约 1.8 µs/worker,而 B=1
  的有效算术只有几十微秒),其余 worker 停靠而非自旋;`XIAOTU_MOE_NSLICE_SMALL=0`
  可退回旧的逐 token 路径(用于 A/B,两条路径结果逐位一致);
- 权重在构造期按 NUMA 节点分片(`XIAOTU_MOE_NOSHARD=1` 可关闭),
  减少跨节点流量;
- 主线 `VLLM_CPU_OMP_THREADS_BIND` 只影响主线自带的 CPU 内核,不影响本引擎。
