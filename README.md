# vllm-xtu-moe

> **XTU = X Transformers Unity**(读音:汉语「小兔」)。本项目是 **vLLM 主线的混合推理加速插件**:
> CPU 专家 + GPU 注意力/长 prefill。内部标识符(`vllm_xiaotu_moe` / `xiaotu_moe` 包名、
> `XIAOTU_*` 环境变量、目录名)保持不变,以免破坏既有配置与调用方。

将 [xiaotu-moe](https://github.com/…/xiaotu-moe)(Apache-2.0,CPU AVX512-VNNI/BF16 fp4 MoE 引擎,
lk_moe 的开源重实现)作为**计算内核**,适配到 **vLLM 主线**(非 fork)的模块化 MoE 架构,做成
`FusedMoEExperts` 计算模块,在**无 AMX 的 x86(AMD EPYC)**上跑 MXFP4/FP8 MoE。

- **路线**:走 vLLM 主线插件(第二条路)。编排(serving/KV/scheduling/TP)用主线;
  计算内核(无 AMX 的 MXFP4/FP8 AVX512-VNNI/BF16)是 xiaotu-moe 的核心,我们保留并挂进主线。
- **目标**:在本机(2×A100 + AMD EPYC 9654 + 24 通道 DDR5-4800)上达到与 fork(xiaotu/lk)
  相当的并发 decode ~100 tok/s,并让长 prefill 走 GPU。
- **License**:Apache-2.0(继承 xiaotu-moe;不复制 lk_moe 专有代码)。

## 背景(一句话)
vLLM 主线已把 MoE 模块化 + 按 CPU 架构自动分发;但主线自己的量化 CPU 内核(MXFP4/FP8/INT4/INT8)
**全部要求 AMX(Intel 专有)**,本机 AMD 9654 无 AMX → 主线在本机对 MXFP4/FP8 没有任何可用 CPU
内核。**xiaotu-moe 恰好提供"无 AMX x86 上的 MXFP4/FP8 AVX512 内核"** → 我们不是可有可无,
而是主线在本机唯一可用的 CPU 量化内核。

## 布局
```
vllm-xtu-moe/
├── csrc/                 # xiaotu-moe 核心 C++ 引擎(25文件,AVX512 kernels + NUMA pool + scheduler)
├── xiaotu_moe/           # Python 绑定(loader 动态选 ISA,moe.py)
├── integration/          # 既有 fork 集成参考(LVLLM_BRIDGE.md + routed_experts.swap_xiaotu.clean.py)
├── docs/                 # xiaotu-moe 既有报告/diff(Thead geometry, compute perf compare)
├── ref/                  # 本次主干线插件的策略与源码实证(fork_vs_mainline_plugin_decision.md,
│                         #   xiaotu_cpu_expert_mainline.md)
└── src/vllm_xiaotu_moe/  # (WIP) 主线 FusedMoEExperts 适配层
```

## 核心机制(已源码实证,见 ref/fork_vs_mainline_plugin_decision.md §9)
- **线程绑定**:主线 `VLLM_CPU_OMP_THREADS_BIND`(auto)→ 每 TP rank 绑一个 NUMA node 全部核 +
  OMP env(KMP_AFFINITY/GOMP/OMP_PLACES+PROC_BIND)。插件 `apply()` 在进程内自动继承该绑核池。
- **内核线程**:主线 CPU kernel(`csrc/cpu/sgl-kernels/moe_fp8.cpp`)用 `at::parallel_for`(torch
  线程池/OpenMP)。我们的内核照此或自管线程,但尊重 rank 的 NUMA 分区。
- **权重分片**:`_load_w13/_load_w2` 按 `tp_rank` 本地分片(= single-copy,无跨 socket 复制);
  CPU MoE 各量化类 `_supports_parallel_config=True`(EP/TP 都行,`apply()` 带 `expert_map`)。
- **结论**:主线能把 NUMA/绑核控制权交给插件层 → 到 fork ~100 tok/s 的前提成立(325GB/s 可堆满)。

## 适配:主线 `FusedMoEExperts` 契约(进行中)
仿 `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py` 的 `CPUExpertsMxfp4/FP8`,
写 `XiaotuMxfp4CPUExperts`/`XiaotuFp8CPUExperts`:
- `_supports_quant_scheme`:同 MXFP4 / FP8;
- `_supports_current_device`:放宽为 `current_platform.is_cpu() and arch==X86`(**不要求 AMX**);
- `apply()`:按契约传 weights/topk,底层调 xiaotu 引擎;
- 权重预处理:尽量复用主线 `prepare_mxfp4_moe_layer_for_cpu` / `prepare_fp8_moe_layer_for_cpu`,
  必要时自写 repack(类比 gpu_prefill 撞的 marlin repack)。

## 环境(见 ref/setup_mainline_vllm.sh)
- conda env `vllm-xiaotu-moe`(Python 3.12)+ 主线 vLLM 0.28(torch==2.13.0)。torch 从 aliyun
  —— **环境名沿用旧名(内部标识符,改名会打断既有脚本)**;对外项目名是 `vllm-xtu-moe`。
  cu130 镜像装(pytorch-wheels 已镜像 cuda13 全栈,直连不走 proxy)。
- 并行/编译 ≤96 核(本机 192 核,生产占 96)。

## 测试
- 仅 GPU2/8071 + 离线;不碰 prod 8070/GPU0+1。目标:长 prefill/full decode 走 GPU,~100 tok/s。

---
## 修订记录
- **2026-09-08(本版)** — 新建项目骨架:拷入 xiaotu-moe 核心引擎(csrc 25 文件 + xiaotu_moe py
  绑定 + docs + integration 参考 + ref 策略/实证文档)。背景=主线量化 CPU 内核全要求 AMX,本机
  无,唯我们需要;NUMA 控制权主线已实证可给插件。后续:写适配层,装 torch(vllm-xiaotu-moe env,
  aliyun 镜像直连),适配主线联调。
