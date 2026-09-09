# xiaotu 计算层 vs vLLM 主线模块化 MoE —— 接口适配与单缓冲 gpu_prefill 落点

日期:2026-09-08(策略研判;依据当日抓取的 vLLM main 源码)
状态:研判结论,为「gpu_prefill 落地」与「只写计算层」两条路的取舍提供事实依据。
相关文档:ref/gpupre_design_xiaotu.md · ref/gpupre_implementation_plan.md ·
         ref/vllm_gpupre_kernel_reference.md · README.md

## 0. 一句话结论
vLLM 主线已把 MoE 模块化,编排 + 按 CPU 架构自动分发(`CpuArchEnum` + AMX/AVX512 特性探测)
**主线全有**;但主线自己的 CPU MXFP4 内核要求 **AMX(Intel 专有)**,而本机 AMD EPYC 9654(Genoa)
**无 AMX** → 主线在**本机**对 MXFP4 没有任何可用 CPU 计算内核。**xiaotu/lk 的生态位恰好就是
「无 AMX x86 上的 MXFP4 AVX512-VNNI/BF16 计算内核」**。因此「给主线只写计算层、免 fork 维护编排」
**成立,且比先前以为的更现实**;但接口 churn 与 repack 契约是两个必须实测的未知数。

## 1. vLLM 主线模块化 MoE 现状(已实证,main 分支)
主线把 `fused_moe/` 重构为模块化架构:
- `experts/cpu_moe.py`(1546 行)、`experts/cpu_int4_moe.py`(267 行)是一批 `FusedMoEExperts` 子类。
- 分发不做用户配置,而是 `is_supported_config()` 在**运行时**逐个类探测
  (`_supports_current_device` + `_supports_quant_scheme` + `_supports_activation`…),命中才启用。
- CPU 侧现有类与量化/ISA 要求:

| 类 | 量化 | 架构要求/ISA |
|---|---|---|
| `CPUUnquantizedExperts`(X86/ARM/Power 子类) | 非量化 | 按 `CpuArchEnum` 分派 |
| `CPUExpertsFp8` | FP8 | x86 + **AMX**(`torch.cpu._is_amx_tile_supported()`) |
| `CPUExpertsMxfp4` | MXFP4 | x86 + **AMX** |
| `CPUExpertsInt4` | INT4 | x86 + **AMX** |
| `CPUExpertsInt8` | INT8 | x86 + **AMX** |
| `CPUExpertsInt4`(cpu_int4_moe.py) | INT4 W4A8 | 仅 **ARM** |

ISA 分发靠运行时特性探测:`get_cpu_architecture()`(X86/ARM/PowerPC/S390X/RISCV)+
`torch.cpu._is_amx_tile_supported()` / `_is_avx512_supported()` / `_is_avx512_bf16_supported()`;
`import_kernels()` 按 AVX512-BF16 / AVX512 / AVX2 自动选 `vllm._C` / `_C_AVX512` / `_C_AVX2`。
**这正是「avx512-vnni、avx2、amx 动态选最优、免配置」——主线已有,不用我们造。**

## 2. 决定性缺口:主线的 MXFP4 CPU 内核只认 AMX,本机没有
`CPUExpertsMxfp4._supports_current_device()`(cpu_moe.py):
```python
return (
    current_platform.is_cpu()
    and current_platform.get_cpu_architecture() == CpuArchEnum.X86
    and torch.cpu._is_amx_tile_supported()
)
```
AMX 是 **Intel 专有**(Sapphire Rapids 起)。本机 **AMD EPYC 9654(Genoa)** flags 含
`avx512f bf16 vnni` 但**无 `amx`** → `torch.cpu._is_amx_tile_supported()` 返回 False。
→ 主线在**本机**对 MXFP4(DeepSeek-V4 的权重类型)没有可用 CPU 内核:
`CPUExpertsMxfp4` 被当前设备谓词挡掉,其余类量化方案不匹配。硬跑即「无支持内核」报错。
**主线「自动分发」在无 AMX 的 x86 上对 MXFP4 是死路。**

## 3. 我们要写的不是"通用分发器",而是"无 AMX x86 的 MXFP4 内核"
- 编排(routing、topk、逐 token 专家分发、权重驻留/offload、KV)主线 `FusedMoEExpertsMonolithic`
  全都给 → **「只写计算层、不碰编排」可行**。
- 计算:本机只能用 AVX512-VNNI/BF16(MXFP4 e8m0 block=32 权重)——即 lk/xiaotu 那套
  `_lk_moe_C_avx512_vnni.so` 的活(区别于 gpu_prefill 的 GPU 侧,这里是 CPU 计算层)。
- 注册路径(两条,须实测):
  - **A. CustomOp 覆盖**:`CustomOp.register("modular_fused_moe")` 可被外部包覆盖(不改主线 C++)。
  - **B. 第 N 个子类**:仿 `CPUExpertsMxfp4` 写 `XiaotuMxfp4CPUExperts`,`_supports_quant_scheme`
    同 MXFP4、`_supports_current_device` 放宽为 x86(不要求 AMX),由分发器自动选中。

## 4. 与 gpu_prefill 落地的关系
- fork 现行 `_ensure_gpu_prefill`(site-packages,routed_experts.py)复用 **vLLM MARLIN 内核**:
  逐层把权重 `.to("cuda")` + `_setup_kernel` + LRU window(`PREFETCH_WINDOW`)。
  **已实测(round6)此路对大 prefill 必 OOM**:MARLIN repack 尖峰 + 每层 ~0.2GB 固有增长 +
  非 PyTorch ~4.7GB,40GB 下 layer41 materialize 即崩。见 results.txt [后端空间封死]。
- 定稿方案「真·单缓冲」= **读共享 raw-fp4 缓冲的 lean 内核**(非 MARLIN):GPU 常驻仅 1 块
  3.2GB raw,无 repack 尖峰、无每层专属副本。此 lean 内核**可独立于 fork 存在**,即上文 §3 的
  计算层内核——**两条路在本机收敛于"写一个无 AMX 的 MXFP4 lean 内核"**。
- 重要:xiaotu 引擎 csrc(moe.cpp/mlp.cpp/linear.cpp/amx_gemm.cpp)**纯 CPU,无 GPU 入口**;
  其 `MOEV2.use_gpu_prefill` 置位只是转发给 lk/闭源。故「单缓冲 GPU 拷贝 + GPU 内核计算」的
  lean 内核必须由我们在 xiaotu 之外以 CUDA 实现,或挂进主线 `FusedMoEExperts.apply()`,
  二者接口不同但内核可复用。

## 5. 成本/风险清单(「只写计算层」的真实代价)
**省下的:**
- 编排、topk、专家分发、权重驻留/offload、KV —— 主线全有,直接复用。
- 动态 ISA 分发 —— 主线 `is_supported_config` 已有。

**省不下的(真成本):**
1. 无 AMX x86 上跑 MXFP4 block=32 的 **A V X 5 1 2 - V N N I / B F 1 6 内核本身**(lk/xiaotu 的活)。
2. 薄薄一层接口适配:
   - `FusedMoEExperts.apply()` 签名(hidden_states,w1,w2,router_logits,activation,…)+
     `_supports_*` 谓词 + `process_weights_after_loading()`。
   - 权重预处理:主线 `prepare_mxfp4_moe_layer_for_cpu`(`convert_weight_packed/convert_scale_packed`)
     —— 我们的 DeepSeek-V4 权重能否直接走它 repack、还是要自己 repack(类比 gpu_prefill 撞的
     marlin repack),**第一个要验证的未知数**。
3. **接口 churn 税**:主线模块化 MoE 仍在剧烈开发(AMX 支持 PR #51546 当前 open 未合;
   `cpu_fused_moe.py` 刚被拆成 `experts/`)。挂 `apply()` 就得跟每次重构改适配层。
   「跟随主线」= 维护薄薄一层 + 定期跟进上游,不是零维护。

**须实测的三个未知数(不测则承诺为空谈):**
- 主线 MXFP4 `fused_experts_cpu` 是否接受 block=32(DeepSeek-V4 是 e8m0 block=32)。
- 分发器怎么认自定义类:CustomOp 覆盖(A)是否足以让新类被选中,还是必须改注册表(B=半 fork)。
- 权重是否走主线 `prepare_mxfp4_moe_layer_for_cpu` 的 repack。

## 6. 建议下一步(小成本钉死未知数)
1. 在测试环境装带模块化 MoE 的 vLLM 主线,对本机跑 MXFP4,看:(a) 无 AMX 是否真报「无支持内核」;
   (b) `fused_experts_cpu` 支不支持 block=32;(c) 分发器认不认自定义类(CustomOp vs 子类)。
   小投入大信息 → 给「值不值得」定案。
2. 出正式设计 `xiaotu 计算层挂主线`:两种注册路径对比 + 接口适配清单 + 需跟进的上游 commit 清单。
3. 继续 fork 内 gpu_prefill 落地(独立于主线的技术资产,无论如何都成立)。

---
## 修订记录
- **2026-09-08(本版)** — 新建。依据当日抓取的 vLLM main 源码(`experts/cpu_moe.py`、
  `cpu_int4_moe.py`、`cpu.py`、`modular_kernel.py`、`fused_moe_modular_method.py`、`interface.py`):
  主线模块化 MoE + 按架构自动分发已存在;但 `CPUExpertsMxfp4` 硬性要求 AMX(Intel 专有),
  本机 AMD EPYC 9654 无 AMX → 主线在本机对 MXFP4 无 CPU 内核。结论:两条路(挂主线 / fork 内
  gpu_prefill)在「写无 AMX 的 MXFP4 lean 内核」上收敛。列出三个须实测的未知数与接口 churn 成本。
