# 通用化路线图:从"DS-V4 专用"到"通用 CPU MoE 后端"

> 用户 2026-09-09 提问:「如果我要 xtu-moe 更通用,即支持所有已知数据格式、
> 支持从 AVX2 到 AMX 的所有指令集优化,还需要做什么?」
> 本文先盘点**已有基础**(比预期完整),再给**缺口清单 + 工作量 + 优先级**。

## 0. 结论先说

| 维度 | 现状 | 通用化还需要 |
|---|---|---|
| **指令集** | `loader.py` 已有 5 级运行时阶梯(scalar→avx2→avx512_base→avx512_vnni→avx512_bf16);`build_variants.sh` 已能一次编出 5 个变体;**但插件仓库只打包了 avx512_bf16 一个 .so** | ①把 5 个变体都编进 wheel(近乎零开发);②AMX 接线(代码已有 3054 行,未接入) |
| **数据格式** | 已有 6 种权重 × 2 种激活 = 10 个实例:BF16 / FP16 / FP8(e4m3, W8A16) / MXFP4 / NVFP4 / WNA16 | INT4 W4A16(组量化+零点)、INT8 W8A8、MXFP8/MXFP6、FP8 e5m2/FNUZ、GGUF k-quants |
| **模型通用性** | **只 OOT 覆盖 `DeepseekV4ForCausalLM`** | 改成**通用 CPU experts 后端**(挂 vLLM 的 `FusedMoEExperts` 接口),任意 MoE 模型可用 |
| **测试** | 单机 golden(本机 AVX-512 BF16) | 格式 × ISA 矩阵;无 AMX/AVX2 机器时的 CI 策略 |
| **打包** | 单 .so | 多 ISA .so 同 wheel(每个约 1.2 MB) |

**最大的"不通用"其实不是格式也不是指令集,而是模型绑定** —— 现在只有 DS-V4 能走这条路。

---

## 1. 指令集:从 AVX2 到 AMX

### 1.1 已经具备的

- `xiaotu_moe/loader.py`:按 `/proc/cpuinfo` 的 flags 选最高可用变体,5 级阶梯;
- `xiaotu-moe/scripts/build_variants.sh`:同一份 `binding.cpp` 编译 5 次,
  每级一个 `_xiaotu_moe_C_<suffix>.so`;AMX 一行被显式跳过(本机无 AMX);
- 内核头文件已按编译期宏分级:
  `moe_v2_packed4.hpp` 里 `#if defined(__AVX512F__) … #elif defined(__AVX2__) … #else`,
  `kernels/bf16_gemm.hpp` 有 `XIAOTU_MOE_HAVE_AVX512_BF16 / _AVX512 / _AVX2` 三级回退。

### 1.2 要做

| 任务 | 内容 | 工作量(估) |
|---|---|---|
| **T1 打包全部 ISA** | 跑 `build_variants.sh`,把 5 个 .so 放进 `xiaotu_moe/build/` 并进 wheel;加一个"变体齐全性"自检 | **0.5–1 天**(几乎零开发) |
| **T2 AMX 接线** | `csrc/moe/amx_gemm.cpp`(3054 行,ggml 风格 tile GEMM)接成 `AMXWeightTraits`;权重需**预 pack 成 AMX tile 格式**;loader 增加 `avx512_amx` 级 + `torch.cpu._is_amx_tile_supported()` 检测;`tile_config_t` 初始化 | **1–2 周**(需 SPR/EMR 机器验证) |
| **T3 AVX-512 FP16 / VNNI 细调** | SPR 的 `-mavx512fp16`、`-mavx512vnni` 的 int8 路径(现在 vnni 变体编了但内核没专门用) | 3–5 天 |

> 注意:AMX 只能加速 **bf16 与 int8** 两类 tile 运算;MXFP4 要走 AMX 的话是
> "反量化到 bf16 tile → AMX bf16 GEMM"。
> 本机实测(真实路由):CPU 引擎 **1.7–2.0 TFLOP/s ≈ AVX-512 BF16 峰值的 15–18%**,
> 时间 99.7% 在两个 GEMM 相位;权重流量 ~10 GB/s(机器可做 740 GB/s)⇒ **不是带宽受限**。
> 反量化占比**尚未被实验确定**(见 `EXPERIMENT_REPORT.md` §7.2b 的更正)。
> AMX 的 bf16 tile 峰值远高于 AVX-512,理论上限最高,但需要 SPR/EMR 机器验证,
> 所以**优先级仍排在"打包全 ISA + 通用后端 + 格式补齐"之后**。

---

## 2. 数据格式

### 2.1 已有(CRTP traits,加一个格式 = 实现 4 个方法)

`moe_v2.hpp` 的 `WeightTraitsBase<Derived>` 要求每个格式实现:
`w13_bytes_impl` / `w2_bytes_impl` / `gate_up_impl` / `down_impl`,
然后在 `binding.cpp` 用 `bind_moe_class<WT, ACT>` 实例化。现有:

| 格式 | traits | 说明 |
|---|---|---|
| BF16 | `BF16WeightTraits` | 无量化 |
| FP16 | (同上,激活不同) | |
| FP8 e4m3 | `FP8WeightTraits` | W8A16,块缩放 |
| MXFP4 | `MXFP4WeightTraits` | e2m1 + e8m0 block-32(W4A16) |
| NVFP4 | `NVFP4WeightTraits` | e2m1 + e4m3 block-16 + 全局 scale |
| WNA16 | `WNA16WeightTraits` | int4 权重量化 |

### 2.2 缺口与优先级

| 格式 | 谁需要 | 工作量(估) | 优先级 |
|---|---|---|---|
| **INT4 W4A16 组量化(+zero-point)** | vLLM 主线 CPU 已有(GPTQ/AWQ 系),我们缺 | 3–5 天 | **P1** |
| **INT8 W8A8**(per-tensor/channel) | 常见服务端量化 | 3–5 天 | **P1** |
| **MXFP8 / MXFP6** | 新一代 MX 格式 | 3–4 天 | P2 |
| **FP8 e5m2 / FNUZ** | AMD gfx942 生态 | 2–3 天 | P2 |
| **GGUF k-quants(Q4_K/Q5_K/Q6_K)** | llama.cpp 生态 | 1–2 周(格式复杂) | P3 |
| **Marlin 打包布局** | 直接吃 vLLM 已打包的权重 | 1 周 | P3 |

每加一个格式的固定成本:**traits 实现 + 权重 loader/映射 + golden 测试 + 文档**。
`docs/GPU_PREFILL.md` 与 `report/golden.txt` 里的 golden 框架可以复用。

> 一个现实约束:**"所有已知格式"是个长尾**。建议先对齐
> "vLLM 主线 CPU MoE 已有的格式集(FP8/MXFP4/INT4/无量化)+ 我们已领先的 NVFP4/WNA16",
> 其余按用户需求增量做。

---

## 3. 真正的通用化:从模型覆盖到通用后端

现在 `vllm_xiaotu_moe/hybrid_model.py` 是 **OOT 覆盖 `DeepseekV4ForCausalLM`**,
只对 DS-V4 生效。要"通用",应该改成 vLLM 的 **CPU experts 后端**:

- 目标接口:`vllm.model_executor.layers.fused_moe.experts.cpu_moe.CPUExperts*`
  (主线已有 `CPUExpertsMxfp4` / `CPUExpertsFp8` / `CPUExpertsInt4`,但都要求 AMX);
- 我们的做法:提供 `XiaotuCPUExperts`(`mixed_experts.py` 已有雏形),把
  `_supports_current_device()` 放宽到"x86 且引擎变体可用",然后在
  `FusedMoEExperts` 的 oracle 里注册;
- 这样**任何** MoE 模型(DS-V4、Qwen、Mixtral、GLM…)只要权重量化格式在支持列表里,
  都能自动走 CPU 专家 + GPU 注意力/长 prefill;
- 工作量:**1–2 周**(接口适配 + 权重映射 + 回归)。

这一步比"再加 5 个格式"对"通用"的贡献大得多。

---

## 4. 测试与 CI

- **格式 × ISA 矩阵**:每个格式在每个 ISA 变体上都要有 golden(现在只有
  AVX-512 BF16 一条线)。可以用 `_run_one_variant.py`(已存在)驱动。
- **没有 AMX / AVX2 机器怎么办**:本机是 AMD(有 AVX-512 BF16,无 AMX、无 AVX-512 FP16)。
  可选:①自托管 runner;②QEMU 功能级验证(只能证明"能跑对",不能测性能);
  ③按 ISA 分 wheel + 用户侧自检(启动时打印选中的变体,和 `loader._chosen`)。
- **性能基线**:每个 ISA 变体跑 `scripts/bench_cpu_engine.py`,把结果落盘对比。

## 5. 打包与发布

- 一个 wheel 带全部 ISA 变体(5–6 个 .so × ~1.2 MB ≈ 7 MB,完全可接受);
- wheel tag 已经是 `cp312-cp312-manylinux_2_34_x86_64`(见 `setup.py`);
- 建议加一个 `xiaotu_moe.self_check()`:打印 CPU flags、选中的变体、可用的格式,
  便于用户报 bug。

---

## 6. 建议的执行顺序(按"收益/成本")

| 顺序 | 事项 | 为什么 |
|---|---|---|
| **1** | 打包全部 ISA 变体(T1) | 几乎零成本,立刻让 AVX2-only 机器能用 |
| **2** | 通用 CPU experts 后端(§3) | "通用"的核心;格式再多,只有 DS-V4 能用也没意义 |
| **3** | INT4 组量化 + INT8(§2.2 P1) | 对齐主线 CPU 的格式集,覆盖面最大 |
| **4** | 格式 × ISA 测试矩阵 + self_check(§4/§5) | 保证上面两步不回归 |
| **5** | AMX 接线(T2) | 性能上限最高,但需要硬件 + 我们已测出瓶颈不在算力 |
| **6** | 长尾格式(MXFP8/GGUF/Marlin 布局) | 按用户需求增量 |

**总计(粗估)**:做到"主线 CPU 格式集 + 全 ISA 可用 + 通用后端"约 **4–6 周**;
再加上 AMX 与长尾格式约 **再 3–5 周**。其中**第 1、2 步就覆盖了 80% 的"通用"诉求**。

---
## 修订记录

- **2026-09-09(第 2 版)** — 更正 §1.2 的 AMX 说明:删掉"瓶颈是反量化 ALU"的说法
  (已被证伪),换成真实路由下的实测(1.7–2.0 TFLOP/s、99.7% 在两个 GEMM 相位、
  非带宽受限、反量化占比未定)。依据:`EXPERIMENT_REPORT.md` §7.2b。
- **2026-09-09(第 1 版)** — 新建。基于对 `loader.py` / `build_variants.sh` /
  `moe_v2*.hpp` / `binding.cpp` / `amx_gemm.cpp` / 主线 `cpu_moe.py` 的实际盘点,
  给出"已有基础 + 缺口 + 工作量 + 优先级"。依据:用户 2026-09-09 提问 +
  仓库现状(见文中引用的文件与行)。
