# vllm-xtu-moe

> 本仓库的根目录就是 **插件项目本身**(`vllm-xtu-moe`)。英文简版见 [`README_EN.md`](README_EN.md)。
>
> **XTU = X Transformers Unity**(读音:汉语「小兔」)。本项目是 **vLLM 主线的混合推理加速插件**:
> CPU 专家 + GPU 注意力/长 prefill。内部标识符(`vllm_xiaotu_moe` / `xiaotu_moe` 包名、
> `XIAOTU_*` 环境变量、目录名)保持不变,以免破坏既有配置与调用方。

---

## ⚠️ 项目关系(务必先读,别把两个项目搞混)

| 项目 | 是什么 | 现在怎么用 |
|---|---|---|
| **`vllm-xtu-moe`**<br>(本仓库) | **vLLM 主线插件**:混合推理(专家权重放 CPU、注意力与长 prefill 留 GPU),按量化格式把 CPU 计算后端挂到主线的 `FusedMoEFactory` | **对外发布的项目**。用户入口:`pip install vllm-xtu-moe` + `VLLM_EXPERTS_LOAD_DEVICE=cpu` |
| **`xiaotu-moe`**<br>(另一个独立仓库) | 独立的 **CPU MoE 引擎项目**:闭源 `lk_moe` 的开源重实现,当年的目标是在 `Lvllm` / `Lvllmds4-x` fork 里做 **drop-in 替换** | **已转私有,仅作参考**。它的引擎源码作为本仓库的**内置计算内核**(`xiaotu_moe/` + `csrc/`)继续维护 |
| **`Lvllmds4-x`**<br>(第三方 vLLM fork) | 别人的 vLLM fork | 只作为 **PR2/PR3 的代码来源**(Apache-2.0,署名保留);**不是本项目的运行依赖** |

一句话:**引擎来自 xiaotu-moe,项目本身是 vllm-xtu-moe,运行环境是 vLLM 主线(不是 fork)。**

> 因此:本仓库里凡是描述 **fork / lk_moe / drop-in 替换 / lvllm 集成** 的材料
> (`integration/`、`docs/XIAOTU_MOE_REPORT_*.md`、`docs/THREAD_GEOMETRY.md`、
> `docs/compute_perf_compare.md`、`docs/TODO_LONGTERM.md` 等)都是 **xiaotu-moe 时期的历史材料**,
> 只作参考,**不代表本项目路线**。本项目路线与决策见 [`docs/BACKLOG.md`](docs/BACKLOG.md)。

---

## 这个插件解决什么

vLLM 主线已把 MoE 模块化并按 CPU 架构自动分发;但主线自带的量化 CPU 内核
(MXFP4 / FP8 / INT4 / INT8)**全部要求 AMX(Intel 专有)**。本机是 AMD EPYC 9654,
没有 AMX → 主线在本机对 MXFP4/FP8 没有任何可用的 CPU 内核。
本插件内置的 **xiaotu 引擎**提供「无 AMX x86 上的 MXFP4 / FP8 / INT4 / BF16
AVX512-VNNI/BF16 内核」,并在混合模式下接管主线 `FusedMoEFactory` 的 CPU 后端槽位。

**为什么需要它**:专家权重动辄几十到几百 GB(DS-V4 ~137 GiB、Qwen3.8-Flash-Next 185 GB),
单张 40 GB 卡放不下;把专家放 CPU、注意力放 GPU 是唯一可行路径,而这条路径在 AMD 上
只有本插件提供可用的 CPU 内核。

## 现状(2026-09-09 夜)

| 能力 | 状态 |
|---|---|
| 通用 CPU experts 后端 | ✅ **BF16 / FP8(block 128) / MXFP4 / INT4(WNA16)** 四种格式;任意 MoE 模型只要格式在列表里就会自动被 oracle 选中(DS-V4 / GLM / Qwen 系) |
| 路由 | ✅ 复用主线 router:softmax / sigmoid+noaux_tc / sqrtsoftplus / grouped-topk / custom routing function |
| 激活 | ✅ packed 布局 gated 激活 + `swiglu_limit/alpha/beta`(GLM-5.x、DS-V4、MiniMax-M3 的 clamped SwiGLU) |
| 原生 vLLM 可用性 | ✅ 插件自带 4 处 mainline shim(专家权重建在 CPU / oracle 前置 / 跳过 AMX 重打包 / 通知 experts),**无需打补丁的 vLLM**(见 `docs/BACKLOG.md` L8、T29) |
| 长 prefill GPU 流式 | ✅ 逐层权重流式 + 阈值切换(见 `docs/GPU_PREFILL.md`、`docs/EXPERIMENT_REPORT.md`) |
| 上游 PR | 🧊 #56118 / #56119 / #56120 按 **D9=A 冻结**中(等通用后端稳定后统一重排) |
| 已知阻塞 | GLM-5.3-Flash 在 A100/SM80 起不来(其 MLA 维度无 attention 后端,与本插件无关);FP8 内核性能待优化 |

## 布局

```
vllm-xtu-moe/
├── vllm_xiaotu_moe/        # ★ 插件本体:主线集成层
│   ├── mixed_experts.py    #   通用 CPU experts 后端(BF16/FP8/MXFP4/INT4 + 主线 router)
│   ├── mainline_shims.py   #   在原生 vLLM 上启用混合模式的 4 处 monkey-patch
│   ├── hybrid_model.py     #   DS-V4 的 OOT 覆盖(历史路径,仍可用)
│   ├── gpu_prefill.py      #   长 prefill 逐层 GPU 流式
│   └── adapter.py          #   引擎 ↔ vLLM 的薄适配
├── xiaotu_moe/             # 内置 CPU 引擎(Python 绑定 + loader + 预编译 .so)
├── csrc/                   # 内置 CPU 引擎(C++ 内核,AVX512-VNNI/BF16 + NUMA pool)
├── patches/                # 上游 PR 快照 / 主线补丁 / RFC 草稿
├── scripts/                # 基准、数值验证、端到端测试
├── docs/                   # 报告与设计(见下方导航;含 xiaotu-moe 历史材料)
├── integration/            # ⚠️ xiaotu-moe 时期的 fork 集成材料(历史)
└── ref/                    # 路线决策与源码实证(历史 + 现状)
```

## 快速开始(原生 vLLM)

```bash
pip install vllm-xtu-moe                 # 插件(含预编译引擎 .so)
export VLLM_EXPERTS_LOAD_DEVICE=cpu      # 专家权重放 CPU
export XIAOTU_MOE_SINGLECOPY=1           # 单份权重(省内存)
python -m vllm_xiaotu_moe.mainline_shims # 自检:shim / 引擎变体 / 后端注册
# 然后正常启动 vllm serve / LLM(...) 即可,MoE 模型会自动走 CPU 专家
```

GPU 平台 + 混合模式的完整说明见 `docs/gpu_cpu_mixed_mode.md`;
长 prefill 的阈值与实测见 `docs/EXPERIMENT_REPORT.md` §5.9。

## 文档导航

| 文档 | 内容 |
|---|---|
| [`docs/BACKLOG.md`](docs/BACKLOG.md) | **待办与决策总账(先读这个)**;含项目关系、D1–D12、L1–L9、证据索引 |
| [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md) | 论文式实测报告(硬件、开发摘要、主线改动、上游策略、全部测量) |
| [`docs/ROADMAP_GENERALITY.md`](docs/ROADMAP_GENERALITY.md) | 通用化路线图(ISA 阶梯、格式矩阵、通用后端计划) |
| [`docs/GPU_PREFILL.md`](docs/GPU_PREFILL.md) | 长 prefill GPU 流式的设计与实测 |
| [`docs/gpu_cpu_mixed_mode.md`](docs/gpu_cpu_mixed_mode.md) | 混合模式设计稿 |
| [`docs/MAINLINE_CHANGES.md`](docs/MAINLINE_CHANGES.md) | 主线改动清单(SM80 支持 + 混合模式) |
| [`patches/UPSTREAM_PRS.md`](patches/UPSTREAM_PRS.md) | 上游 PR 计划与描述快照(冻结中) |
| `integration/`、`docs/XIAOTU_MOE_REPORT_*.md`、`docs/THREAD_GEOMETRY.md`、`docs/compute_perf_compare.md`、`docs/TODO_LONGTERM.md` | ⚠️ **xiaotu-moe 时期的历史材料**(fork / lk_moe 路线),仅作参考 |

## 环境

- conda env `vllm-xiaotu-moe`(Python 3.12)+ **vLLM 主线**(本机为源码树 `6c73b08`,
  torch 2.13.0+cu130)。**环境名沿用旧名**(内部标识符,改名会打断既有脚本);
  对外项目名是 `vllm-xtu-moe`。
- 本机:2×A100-PCIE-40GB(SM80,无 NVLink)+ AMD EPYC 9654(2×96 核,无 AMX)+ 1.5 TB DDR5-4800。
- 纪律:只用 GPU2 / 离线测试,不碰生产 8070。

## License

Apache-2.0(本项目与内置引擎)。第三方代码(Lvllmds4-x 的 SM80 移植)按其 Apache-2.0
保留 SPDX 署名;不复制闭源 `lk_moe` 的任何代码。

---

## 修订记录

- **2026-09-09(第 2 版)** — **明确项目边界,修正与 `xiaotu-moe` 的混淆**:新增「项目关系」章节
  (vllm-xtu-moe = 主线插件;xiaotu-moe = 独立引擎项目,已转私有;Lvllmds4-x = 第三方 fork,
  仅 PR2/PR3 代码来源);把「将 xiaotu-moe 作为计算内核适配」改为「内置引擎 + 主线插件」的准确表述;
  更新布局/现状/快速开始/文档导航到当前实际状态。依据:用户 2026-09-09 澄清 + 仓库现状。
- **2026-09-08(第 1 版)** — 新建项目骨架:拷入 xiaotu-moe 核心引擎(csrc 25 文件 + xiaotu_moe py
  绑定 + docs + integration 参考 + ref 策略/实证文档)。背景=主线量化 CPU 内核全要求 AMX,本机
  无,唯我们需要;NUMA 控制权主线已实证可给插件。后续:写适配层,装 torch(vllm-xiaotu-moe env,
  aliyun 镜像直连),适配主线联调。
