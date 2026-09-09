# vllm-xtu-moe

> **XTU = X Transformers Unity**(读音:汉语「小兔」)。
> 一个 **vLLM 主线插件**:让 **MoE 专家权重住在 CPU、注意力与长 prefill 留在 GPU**,
> 从而在单卡显存放不下专家权重的场景下把模型跑起来。

[**English**](README_EN.md) · 中文(默认)

---

## 它解决什么问题

现代 MoE 模型的**专家权重**动辄几十到几百 GB(DeepSeek-V4 ≈ 137 GiB、
Qwen3.8-Flash-Next 185 GB),远超单卡显存;而注意力、KV cache、路由、共享专家
这些"每 token 都要算"的部分留在 GPU 最划算。把两者拆开是唯一现实的部署方式。

vLLM 主线已经支持"专家放 CPU"的**接口**(`FusedMoEFactory` + 量化后端槽位),
但它自带的量化 CPU 内核(MXFP4 / FP8 / INT4 / INT8)**全部要求 Intel AMX**。
在没有 AMX 的 x86 机器(例如 AMD EPYC)上,这些格式**没有任何可用的 CPU 内核**。

本插件把 **xiaotu CPU 引擎**(AVX-512 VNNI/BF16,无 AMX 依赖)接进主线的
CPU 后端槽位,于是:

- 任意使用 `FusedMoEFactory` 的 MoE 模型,只要量化格式在支持列表里,
  就能在 GPU 上跑注意力、在 CPU 上算专家;
- 不需要 fork vLLM,也不需要改模型代码;
- 支持 GLM / DeepSeek / Qwen / Mixtral 等不同**路由方式**的模型。

## 支持矩阵

| 权重格式 | 引擎内核 | 状态 |
|---|---|---|
| **BF16 / FP16**(无量化) | `MOE_BF16` / `MOE_FP16` | ✅ |
| **FP8 e4m3 + block 128×128**(vLLM `kFp8Static128BlockSym`) | `MOE_FP8` | 🟡 层内数值已验证(引擎 vs torch 参考,真实权重);真实模型端到端一致性排查中;内核性能待优化 |
| **MXFP4**(e2m1 + e8m0 block 32) | `MOE_MXFP4` | ✅ |
| **NVFP4** | `MOE_NVFP4` | ✅ 引擎侧 |
| **INT4 / WNA16**(GPTQ / AWQ 组量化) | `MOE_WNA16` | 🟡 引擎侧可用,组大小/零点适配进行中 |
| INT8 W8A8 | — | ❌ 引擎尚未实现 |

**路由**:直接复用主线 router(softmax、sigmoid + `noaux_tc`、
`sqrtsoftplus`、grouped-topk、自定义路由函数),因此 GLM、DeepSeek、Qwen 等
不同路由风格都能正确选专家。

**激活**:支持 packed 布局的 gated 激活(SILU、`SWIGLUOAI_UNINTERLEAVE`)
以及 `swiglu_limit` / `alpha` / `beta`(GLM-5.x、DeepSeek-V4、MiniMax 使用的
clamped SwiGLU)。**不支持** gate/up 交错布局(`SWIGLUOAI`,如 gpt-oss)——
此时插件会拒绝该后端而不是给出错误结果。

**指令集**:内置引擎按 `/proc/cpuinfo` 在运行时选择最高可用变体:
`scalar → avx2 → avx512_base → avx512_vnni → avx512_bf16`。
不需要 AMX;Intel AMX 机器上主线自带的 CPU 内核也可继续使用。

## 安装与快速开始

```bash
# 1) 主线 vLLM(本插件是插件,不需要 fork)
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130

# 2) 本插件(源码安装时先构建引擎原生扩展)
git clone https://github.com/yeungtuzi/vllm-xtu-moe.git
cd vllm-xtu-moe
PYTHON=$(which python) bash scripts/build_engine_variants.sh   # 约 90 秒
pip install -e .

# 3) 跑一个专家权重放不进显存的 MoE 模型
export VLLM_EXPERTS_LOAD_DEVICE=cpu     # 专家权重放 CPU
export XIAOTU_MOE_SINGLECOPY=1          # 权重只存一份(省内存)
vllm serve <模型目录> \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

自检(打印引擎 ISA 变体、生效的集成补丁、后端注册情况):

```bash
VLLM_EXPERTS_LOAD_DEVICE=cpu python -m vllm_xiaotu_moe.mainline_shims
VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py   # 后端选择探测,秒级
```

完整安装、环境变量与各模型的参考命令见 **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**。

## 工作原理(一句话版)

```
GPU:  attention · KV cache · router · shared experts · 长 prefill 的专家计算(可选)
CPU:  routed experts 的权重与计算(xiaotu 引擎,AVX-512)
```

- 专家权重在构造期就建在 CPU(避免显存 OOM),其余部分照常在 GPU;
- 主线 oracle 在混合模式下优先选中本插件的 CPU 后端;
- 长 prefill 可以按阈值切换为"逐层把该层专家权重流式搬上 GPU 计算"
  (见 [`docs/GPU_PREFILL.md`](docs/GPU_PREFILL.md));
- 上述集成点在上游 vLLM 尚未合并,插件用 `vllm_xiaotu_moe/mainline_shims.py`
  以 monkey-patch 方式提供等价行为,因此**原生 vLLM 即可使用**。

细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | 安装、环境变量/参数、各模型运行命令、排错 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 混合模式与主线集成设计 |
| [`docs/GPU_PREFILL.md`](docs/GPU_PREFILL.md) | 长 prefill 逐层 GPU 流式 |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | 实测:硬件、吞吐、时延、消融 |
| [`docs/UPSTREAM.md`](docs/UPSTREAM.md) | 需要上游配合的改动与对应 PR |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 格式/指令集/后端通用化计划 |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | 已知限制与不支持的情形 |
| [`patches/`](patches/) | 上游 PR 描述与补丁快照、RFC 草稿 |
| [`report/`](report/) | 基准数据与图表 |

## 仓库结构

```
vllm-xtu-moe/
├── vllm_xiaotu_moe/     # 插件:主线集成层
│   ├── mixed_experts.py #   通用 CPU experts 后端(BF16/FP8/MXFP4/INT4)
│   ├── mainline_shims.py#   在原生 vLLM 上启用混合模式的集成补丁
│   ├── hybrid_model.py  #   DeepSeek-V4 的模型级覆盖(可选路径)
│   └── gpu_prefill.py   #   长 prefill 逐层 GPU 流式
├── xiaotu_moe/          # 内置 CPU 引擎(Python 绑定 + C++ 内核 + 运行时 ISA 选择)
├── scripts/             # 构建、基准、数值验证、端到端冒烟
├── patches/             # 上游 PR / 补丁 / RFC
├── docs/                # 文档
└── report/              # 基准数据与图表
```

## 已知限制

- **GLM-5.3-Flash 需要 SM90+**:其 MLA 维度(`qk_nope=256 / rope=0 / v=256`)
  在 vLLM 主线没有任何 attention 后端支持,与 MoE 后端无关。
  专家层的数值已用真实权重验证。详见
  [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)。
- **FP8 CPU 内核性能待优化**:当前是"先正确、后提速"的阶段。
- **不支持专家并行(expert_map)**:TP>1 走权重分片,EP 尚未支持。
- **不支持 gate/up 交错布局**(`SWIGLUOAI`,gpt-oss 系)。

## 许可与第三方

本项目与内置引擎均为 **Apache-2.0**。第三方代码(取自 `Lvllmds4-x` 的 SM80
移植等)按其 Apache-2.0 保留 SPDX 署名;不包含闭源 `lk_moe` 的任何代码。
详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 作者

**大河马 (BigHippo)** `<dahema@me.com>`,由 DeepSeek Harness 辅助。
