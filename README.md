# vllm-xtu-moe

> **XTU = X Transformers Unity**(读音:汉语「小兔」)。
> 一个 **vLLM 主线插件**:让 **MoE 专家权重住在 CPU、注意力与长 prefill 留在 GPU**,
> 从而在单卡显存放不下专家权重的场景下把模型跑起来。

[**English**](README_EN.md) · 中文(默认)

> **📌 当前版本:v0.2**(2026-09-17)——**支持 DeepSeek-V4.1-Flash(748B)**:
> **1M 上下文 + GPU 预填充 + 投机解码**全链路验收通过;其中 **GPU 预填充快 2.0-2.8×**,
> 并补齐官方 `vllm bench serve` 的完整验收数据(6 种 prompt 长度 × C=1/2/4/8)。
> 发行说明:[`RELEASE_NOTES_v0.2.md`](RELEASE_NOTES_v0.2.md) ·
> Releases: <https://github.com/yeungtuzi/vllm-xtu-moe/releases/tag/v0.2>
>
> 历史:另一条**把 lvllm 编排链移植进 vLLM fork** 的路线(两卡 A100-40GB 上解码 C=1 20.29 t/s、
> 预填充 1287 t/s、1M 上下文)见 [`docs/MILESTONE_lk_port.md`](docs/MILESTONE_lk_port.md)。

---

## 它解决什么问题

现代 MoE 模型的**专家权重**动辄几十到几百 GB(DeepSeek-V4 ≈ 137 GiB),
远超单卡显存;而注意力、KV cache、路由、共享专家
这些"每 token 都要算"的部分留在 GPU 最划算。把两者拆开是唯一现实的部署方式。

vLLM 主线已经支持"专家放 CPU"的**接口**(`FusedMoEFactory` + 量化后端槽位),
但它自带的量化 CPU 内核(MXFP4 / FP8 / INT4 / INT8)**全部要求 Intel AMX**。
在没有 AMX 的 x86 机器(例如 AMD EPYC)上,这些格式**没有任何可用的 CPU 内核**。

本插件把 **xiaotu CPU 引擎**(AVX-512 VNNI/BF16,无 AMX 依赖)接进主线的
CPU 后端槽位,于是:

- 任意使用 `FusedMoEFactory` 的 MoE 模型,只要量化格式在支持列表里,
  就能在 GPU 上跑注意力、在 CPU 上算专家;
- 不需要 fork vLLM,也不需要改模型代码;
- 支持 GLM / DeepSeek / Mixtral 等不同**路由方式**的模型。

## 支持矩阵

| 权重格式 | 引擎内核 | 状态 |
|---|---|---|
| **BF16 / FP16**(无量化) | `MOE_BF16` / `MOE_FP16` | ✅ |
| **FP8 e4m3 + block 128×128**(vLLM `kFp8Static128BlockSym`) | `MOE_FP8` | ✅ 层内数值已验证(48 层自校验 ≤1.4e-4) |
| **MXFP4**(e2m1 + e8m0 block 32) | `MOE_MXFP4` | ✅ |
| **NVFP4** | `MOE_NVFP4` | ✅ 引擎侧 |
| **INT4 / WNA16**(GPTQ / compressed-tensors 组量化) | `MOE_WNA16` | ✅ 对称量化(zero point 8)已通:检查点布局在引擎构造时一次性重排;非对称零点与 AWQ 的 N-packed 布局会**显式报错**,详见 `docs/KNOWN_LIMITATIONS.md` |
| INT8 W8A8 | — | ❌ 引擎尚未实现 |

**路由**:直接复用主线 router(softmax、sigmoid + `noaux_tc`、
`sqrtsoftplus`、grouped-topk、自定义路由函数),因此 GLM、DeepSeek 等
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
vllm serve <模型目录> \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85

# 4)(可选,推荐)把**长 prefill 也交给 GPU** —— DeepSeek-V4.1-Flash 实测快 2.0-2.8×
#    两个前提:①阈值要 ≥4096(低于此 CPU 更划算);②**必须显式封顶 KV 池**,
#    否则 vLLM 会把显存填满、GPU 预填充会被逐层静默拒绝(退回 CPU)。
vllm serve <模型目录> \
  --tensor-parallel-size 2 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 8192 \
  --kv-cache-memory 4294967296 \
  --gpu-memory-utilization 0.55 \
  --speculative-config dspark
# 显存配方(TP=2/MBT=8192):非KV 10 + KV 4 + staging 7.2 + 首请求一次性增长 7.7 + 激活 ≈ 33 GiB
# 完整参数表与依据见 docs/RUNBOOK.md §5.9;判定是否真走了 GPU:日志出现 `GPU prefill ACTIVE`
```

自检(打印引擎 ISA 变体、生效的集成补丁、后端注册情况):

```bash
VLLM_EXPERTS_LOAD_DEVICE=cpu python -m vllm_xiaotu_moe.mainline_shims
VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py   # 后端选择探测,秒级
```

> ⏸️ **暂不声明支持**:该模型的实测数据早于 v0.2 的改动(执行模型 / 小 batch 路径 / EP 存储分片),**未在当前代码上复验**。复验计划见 `docs/HANDOFF_v0.2pre.md` §5.1。

DeepSeek-V4-Flash 的**推荐参数与实测数据**见
**[`docs/TUNING_REPORT.md`](docs/TUNING_REPORT.md)**。

完整安装、环境变量见 **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**;
**DeepSeek-V4-Flash 的逐步使用指南 + 初步性能实测**见
**[`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md)**。

> ### ⚠️ 验证时点(重要)
>
> 本文档里的**实测数据不是同一时点的**,请按此判读:
>
> | 结论 | 验证于 | 在 **v0.2** 代码上复验? |
> |---|---|---|
> | **DeepSeek-V4-Flash 主线端到端**(服务、`bench_lat` C=1/2/4、数值门禁 `OK=7 BAD=1`、启动自检) | **0.2pre**(2026-09-14) | ⏸️ **未复验**(清单见 [`docs/HANDOFF_v0.2pre.md`](docs/HANDOFF_v0.2pre.md) §5.1) |
> | **DeepSeek-V4.1-Flash 全链路**(1M 上下文 / GPU 预填充 / 投机 / ShareGPT 端到端 / 长 prompt 回放) | **v0.2**(2026-09-18) | ✅ **是**(即本页数据) |
>
> 原因:v0.2 改动了**所有模型都会走的路径**(CPU 预填充的 scratch 布局、执行模型
> `XIAOTU_MOE_ASYNC=0`、小 batch 路径 `NSLICE_SMALL=0`、**EP 存储分片**)。
> **V4.1-Flash 已在 v0.2 上完整复验**;**其余模型仍需复验**,不要默认沿用旧数字。

### 已验证模型

| 模型 | 状态 | 关键数据(均为本机实测) |
|---|---|---|
| **DeepSeek-V4.1-Flash**(748B) | ✅ **v0.2** | **1M 上下文 + GPU 预填充 + 投机**全链路;主机峰值 **629.4 GiB(−42%)**;KV **6.72M tokens(1M 并发 6.41×)**;ShareGPT C=1 **16.64 output tok/s,TTFT 1339 ms**(out≤128,prompt 均值 227);**TPOT 与上下文强相关:17-token prompt 33.5 ms → 438-token 53.5 ms**(见下方「TPOT 不是常数」);greedy ×2 **5/5 逐字节相同** |
| **DeepSeek-V4-Flash** | ✅ 0.2pre | 主线端到端(服务 / `bench_lat` C=1/2/4 / 数值门禁 `OK=7 BAD=1` / 启动自检) |

* **V4.1-Flash 为什么能支持**:38.5% 的权重是**纯查找表**(Engram,183 GiB,每 token 只需 ~12 KB
  主机流量),官方生产栈也把它放在**主机内存**里 —— 与本项目 `XIAOTU_PLE_CPU=1` 思路一致;
  其余(CED / CSA2 / FP4 KV / DSpark 投机)由 vLLM 主线 `deepseek_v41` 提供,本插件负责 **MoE 层**。
  发布当天的可行性分析见 [`docs/V41_FLASH_ANALYSIS.md`](docs/V41_FLASH_ANALYSIS.md);
  **落地实测与配置**见 [`RELEASE_NOTES_v0.2.md`](RELEASE_NOTES_v0.2.md) 与
  [`docs/RUNBOOK.md`](docs/RUNBOOK.md) §5.9。

## 性能实测(v0.2)

> 机器:2×AMD EPYC 9654(192 核)/ 3×A100-40GB / DDR5-4800 24 通道。
> 口径:**TP=2**、充分预热、唯一 prompt(不命中前缀缓存)、官方 `vllm bench serve`。
> 原始数据在 [`report/tuning/logs/bench_serve_acc2/`](report/tuning/logs/bench_serve_acc2/),
> 完整表见 [`report/tuning/BENCH_REFERENCE.md`](report/tuning/BENCH_REFERENCE.md) §7 与 §9。

### ⭐ ShareGPT 摘要(v0.2;`output tok/s` 与 `TTFT` **两个独立量,不折算**)

口径:`--dataset-name sharegpt`、`--sharegpt-output-len 128`、`--num-prompts 16`、前缀缓存开、
`--backend openai`(**裸补全,不触发思考** —— 见下方「口径警告」)。

| 并发 | Plain:output tok/s | Plain:TTFT (ms) | dspark:output tok/s | dspark:TTFT (ms) |
|---|---|---|---|---|
| **C=1** | **16.64** | **1339** | _待补_ | _待补_ |
| C=4 | 42.50 | 745 | _待补_ | _待补_ |
| C=8 | 50.03 | 1100 | _待补_ | _待补_ |

**v0.2 相对上一版(撤回的 `v0.21.0`)同口径的变化(Plain)**:C=1 output tok/s **13.75 → 16.64(+21%)**、
TTFT **2224 → 1339 ms(−40%)**;而 **mean TPOT 43.32 → 42.46 ms 不变** ——
**涨的全部是 TTFT 那一项,不是解码**。

### ⭐ TPOT 不是常数:它随**上下文长度**变化(这也解释了"到底 36 还是 60 ms")
**同一台服务、同一组参数,只换 prompt 集**(C=1、`--ignore-eos` 强制 128 输出、前缀缓存关):

| prompt 均值 | 解码时上下文 | mean TPOT | p50 TPOT | TTFT | output tok/s |
|---|---|---|---|---|---|
| 17 tok | ~145 | **33.5 ms** | 33.4 | 162 ms | 28.98 |
| 165 tok | ~293 | **39.8 ms** | 37.3 | 947 ms | 21.31 |
| 438 tok | ~566 | **53.5 ms** | 56.5 | 2456 ms | 13.84 |

* **输出长度不是主因**:同样 438-token prompt,`--ignore-eos` 开/关(输出 128 vs 65 token)
  TPOT 是 53.5 vs 58.4 ms —— 一个量级内,方向还不一致(噪声/内容)。
* **prompt 长度才是主因**:17 → 438 token,TPOT **33.5 → 53.5 ms(+60%)**。
  本页下面的"end-to-end"表其实早就显示了同一件事(32-token prompt 36.2 ms → 1024-token 64.6 ms)。
* **机制**:TPOT = **CPU MoE(与上下文无关,≈30-42 ms)** + **GPU 稀疏注意力(随上下文增长、然后饱和)**。
  引擎实测解码桶 `[NS-PROF]` = 755-1056 µs/层 × 40 层 = **30-42 ms** ⇒ 短上下文时
  TPOT 几乎**全是 CPU MoE**,`1/33.5ms ≈ 30 tok/s` 就是当前解码地板。
* ⇒ **所以"36.75 ms"和"60 ms"都对**,只是量在不同上下文上;**报 TPOT 必须同时给上下文/prompt 长度**。
* ⇒ `output_throughput` 里 TTFT 的占比由 **output 长度**决定:
  `output_tput ≈ 1/(TTFT/L + TPOT)`;实测 L=128 只到纯解码上限的 **74%**,L≥1024 达 **96-98%**。

### ⚠️ 口径警告:`--backend openai` **不触发思考**
`--backend openai` 打 `/v1/completions` 并原样发 prompt,而 V4.1 的思考只在 prompt encoder
(经 chat template)里渲染 ⇒ **裸补全路径不开思考**。DSH 的真实路径是 `/v1/chat/completions` + 默认开思考。
上表是**非思考**口径(与历史数字可比);`openai-chat` 列待补。
判断依据是"template 有没有被套上",不是 backend 的名字 —— `custom` 数据集不加
`--skip-chat-template` 时会在客户端套一次,**那条路径也会开思考**。

### GPU 预填充 vs CPU 预填充(DeepSeek-V4.1-Flash,客户端 TTFT)

| prompt | CPU 预填充 | **GPU 预填充** | 加速 |
|---|---|---|---|
| ~3.7K | 24.17 s(151 tok/s) | **12.11 s(302 tok/s)** | **2.0×** |
| ~7.0K | 42.17 s(165 tok/s) | **15.19 s(460 tok/s)** | **2.8×** |

成本模型(实测):**每个 chunk ≈ 8.9 s 固定 + 0.79 ms/token**。固定项 = 每个 chunk 都要把
**143.6 GiB/rank** 的专家权重搬一遍(TP=2,40 层 × 3.589 GiB)⇒ **chunk(`--max-num-batched-tokens`)越大越划算**。

### DSH 开发会话回放(长 prompt;**这不是 ShareGPT**)

ShareGPT 平均 prompt 只有 ~227 token,**永远触发不到 GPU 预填充**(阈值 4096)。
用本机 DSH transcript 的末尾若干轮原文按 token 截断,得到真实的长 prompt 曲线
(C=1、out=128、**CPU 预填充**;`python scripts/make_dsh_replay.py` 现场重建):

| prompt | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|
| TTFT | 11.43 s | 21.99 s | 42.58 s | 85.25 s | 172.56 s |
| 预填充速率 | 179 tok/s | 186 | 192 | 192 | 190 |

⇒ 预填充速率**线性且稳定在 ~190 tok/s**(固定成本已被摊平)。GPU 预填充的同一曲线见 `RELEASE_NOTES_v0.2.md` §3.6。

### 端到端(官方 `vllm bench serve`,TP=2 / MBT=8192 / GPU 预填充 / KV 封顶 4 GiB / 关前缀缓存)

| prompt | TTFT C=1 | TTFT C=8 | 总吞吐 C=1 | 总吞吐 C=8 | TPOT C=1 |
|---|---|---|---|---|---|
| 32 | 0.47 s | 3.49 s | 37 tok/s | 95 tok/s | 36.2 ms |
| 256 | 1.98 s | 10.9 s | 55 | 134 | 43.6 ms |
| 1024 | 10.0 s | 23.6 s | 65 | 242 | 64.6 ms |
| 4096 | 12.8 s | 46.4 s | 200 | 379 | 66.3 ms |
| 16384 | 34.5 s | 170 s | 399 | 455 | 54.4 ms |
| 32768 | 70.9 s | 333 s | 420 | 454 | 58.4 ms |

**怎么读**:①长 prompt 的总吞吐**饱和在 ~420-455 tok/s**(上界由 GPU 预填充的固定成本决定);
②**TTFT 与长度线性、与并发强相关** —— 预填充是串行共享资源,并发只增加排队;
③短 prompt(32/256)走 CPU,TTFT 亚秒级。

---

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
| [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) | **DeepSeek-V4-Flash 使用指南与初步性能** |
| [`docs/TUNING_REPORT.md`](docs/TUNING_REPORT.md) | **调参报告:DeepSeek-V4-Flash 的推荐运行参数 + 全部实测** |
| [`docs/PERFORMANCE_OPTIMIZATION.md`](docs/PERFORMANCE_OPTIMIZATION.md) | **性能优化手册:每层归因、NUMA 权重布局(单卡 +38%)、CPU-TP 设计、硬件评估与实验矩阵** |
| [`docs/V41_FLASH_ANALYSIS.md`](docs/V41_FLASH_ANALYSIS.md) | DeepSeek-V4.1-Flash 资源账与可行性分析 |
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
- **GPU 预填充的 chunk 上限是 `--max-num-batched-tokens 8192`**(DeepSeek-V4.1-Flash / A100-40GB):
  `16384` 会 OOM,且 OOM 点是 **attention** 的 `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`
  (不是 MoE)——16K chunk 的激活 + staging 放不下。想再往上需要更大显存或把激活压下来。
- **开了 GPU 预填充就必须显式封顶 KV 池**(`--kv-cache-memory`):vLLM 默认把显存填到
  `--gpu-memory-utilization` 为止,预填充的 staging 会**逐层被拒并静默退回 CPU**。见 `docs/RUNBOOK.md` §5.8。
- **③ CED 预填充捷径:已实现机制链但默认关闭,当前拿不到收益**。
  实测把冲突逐个定位到**两个上游不变量**:`mhc/tilelang.py:345` 断言
  `x.shape == (num_tokens, hidden_size)`(层不能返回比本步 `num_tokens` 更短的张量),
  以及 `positions` 来自 runner 全局缓冲、插件层够不到。
  ⇒ 要真正拿到收益应把"本步有效 token 数"作为一等量在**上游 vLLM** 里下传。
  全过程与证据见 [`report/tuning/NOTES.md`](report/tuning/NOTES.md) §571-§604。

## 许可与第三方

本项目与内置引擎均为 **Apache-2.0**。第三方代码(取自 `Lvllmds4-x` 的 SM80
移植等)按其 Apache-2.0 保留 SPDX 署名;不包含闭源 `lk_moe` 的任何代码。
详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 作者

**大河马 (BigHippo)** `<dahema@me.com>`,由 DeepSeek Harness 辅助。
