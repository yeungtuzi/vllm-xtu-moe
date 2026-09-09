# 运行手册 RUNBOOK — 装 vLLM 主线 + 装插件 + 跑 DeepSeek-V4-Flash / Qwen3.8-Flash-Next

> 本文给出**可直接复制粘贴**的命令:①装主线 vLLM(简要)②装 `vllm-xtu-moe`
> ③环境变量与命令行参数 ④跑 DS-V4-Flash 与 Qwen3.8-Flash-Next-FP8 的参考配置
> ⑤自检与排错。
>
> 本机验证过的组合(2026-09-09):Python 3.12.14 · torch 2.13.0+cu130 ·
> vLLM 主线 `0.1.dev1+g6c73b08de`(commit `6c73b08`)· 2×A100-PCIE-40GB(SM80)·
> AMD EPYC 9654(无 AMX)。

---

## 0. TL;DR(最短路径)

```bash
# 1) 主线 vLLM(任意较新的 mainline 都可以;本机用 6c73b08 验证)
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130   # 或源码安装见 §1
# 2) 插件(含内置 CPU 引擎)
pip install vllm-xtu-moe          # 或源码安装见 §2
# 3) 跑一个放不进显存的 MoE 模型
export VLLM_EXPERTS_LOAD_DEVICE=cpu     # ★ 必须:专家权重放 CPU
export XIAOTU_MOE_SINGLECOPY=1          # 推荐:权重单份(省一半内存)
vllm serve <模型目录> --tensor-parallel-size 1 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --enforce-eager \
  --kernel-config.enable_jit_warmup=false
```

装完先自检(会打印引擎 ISA 变体、已生效的 shim、后端注册情况):

```bash
python -m vllm_xiaotu_moe.mainline_shims
```

---

## 1. 安装主线 vLLM(简要)

> 本插件是**主线插件**(通过官方 `vllm.general_plugins` 入口加载),
> **不需要 fork**。混合模式所需的 4 处主线配合由插件自带的
> `vllm_xiaotu_moe/mainline_shims.py` 以 monkey-patch 形式提供
> (见 `docs/BACKLOG.md` L8/T29),所以在原生 vLLM 上即可工作。

**方式 A:源码安装(本机用的方式,可控)**

```bash
conda create -n vllm-xiaotu-moe python=3.12 -y && conda activate vllm-xiaotu-moe

# torch(本机用 aliyun 镜像直连装 cu130 全栈)
pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu130

git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout 6c73b08          # 本机验证过的提交;更新的 main 一般也可以
VLLM_USE_PRECOMPILED=1 pip install -e .  # 复用预编译内核,避免几小时编译
```

**方式 B:pip 安装(最简)**

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130
```

> 注意:`VLLM_USE_PRECOMPILED=1` 会从 vLLM 的发布渠道取预编译算子,速度最快;
> 若目标平台没有预编译包,则退回源码编译(数小时)。

---

## 2. 安装 vllm-xtu-moe

```bash
git clone https://github.com/yeungtuzi/vllm-xtu-moe.git
cd vllm-xtu-moe

# 2a) 先构建内置 CPU 引擎的原生扩展(生成 5 个 ISA 变体,约 90 秒)
PYTHON=$(which python) bash scripts/build_engine_variants.sh
#    产物: xiaotu_moe/build/_xiaotu_moe_C_{scalar,avx2,avx512_base,avx512_vnni,avx512_bf16}.so
#    运行时由 xiaotu_moe/loader.py 按 /proc/cpuinfo 自动选最高可用变体

# 2b) 安装插件(editable 或普通安装都行)
pip install -e .
#    或 pip install vllm-xtu-moe   (wheel 里已打包引擎 .so,无需 2a)
```

**自检**(应看到 `mixed_mode_enabled = True`、引擎变体、4 个后端被换成 `XiaotuCPUExperts*`):

```bash
VLLM_EXPERTS_LOAD_DEVICE=cpu python -m vllm_xiaotu_moe.mainline_shims
# 再确认 oracle 会选中我们的后端(不加载权重,秒级):
VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py
```

期望输出(节选):

```
[OK ] bf16      -> CPU  mixed_experts.XiaotuCPUExpertsBF16
[OK ] fp8-glm53 -> CPU  mixed_experts.XiaotuCPUExpertsFp8
[OK ] fp8-dsv4  -> CPU  mixed_experts.XiaotuCPUExpertsFp8
[OK ] wna16-int4-> is_supported_config=True
```

---

## 3. 环境变量与命令行参数

### 3.1 必开 / 常用

| 变量 | 推荐值 | 作用 |
|---|---|---|
| `VLLM_EXPERTS_LOAD_DEVICE` | `cpu` | ★ 混合模式开关:routed-expert 权重建在 CPU(否则几百 GB 专家在构造时就 OOM) |
| `XIAOTU_MOE_SINGLECOPY` | `1` | 权重只存一份(NUMA 分片);不设会按 socket 复制,内存约 2× |
| `XIAOTU_MAINLINE_SHIMS` | `1`(默认) | 在原生 vLLM 上启用混合模式的 4 处 monkey-patch;`0` 关闭 |
| `CUDA_VISIBLE_DEVICES` | 例 `2` | 选卡(本机 GPU2 为测试卡) |
| `HF_HUB_OFFLINE` | `1` | 离线(权重已在本地) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | 关掉 FlashInfer 采样器(避免额外依赖/开销) |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600`–`7200` | 大模型 + CPU 专家初始化慢,启动等待要放大 |

### 3.2 长 prefill 走 GPU(可选)

| 变量 | 推荐值 | 作用 |
|---|---|---|
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | `384`(默认)或 `512`–`768` | 单层 prefill token 数达到阈值时,该层 MoE 改为**逐层流式 GPU 计算**;`0` 关闭(纯 CPU) |
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE` | 路径 | 客户端进程设 env 不生效时的替代(从文件读阈值) |
| `XIAOTU_GPU_PREFETCH_AHEAD` | `1` | 预取下一个层的权重(双槽流水) |

阈值推荐表(实测)见 `docs/EXPERIMENT_REPORT.md` §5.9;机制见 `docs/GPU_PREFILL.md`。

### 3.3 观测 / 调试(默认关闭,不影响性能)

| 变量 | 作用 |
|---|---|
| `XIAOTU_MOE_PROFILE=1` | 引擎相位计时(A gate/up、B down、C 组合),每 40 次调用打印 |
| `XIAOTU_VERIFY_LAYER=1` | **层内数值自校验**:每层第一次调用时,用 torch 参考(同一批已加载权重)算一遍并打印 `rel_rms` |
| `XIAOTU_MOE_THREADS=<n>` | 引擎线程数(默认用共享 NUMA 池的全部核心) |
| `XIAOTU_MOE_NOSHARD=1` | 关闭权重 NUMA 分片(调试用) |
| `XIAOTU_NVTX=1` / `XIAOTU_TORCH_PROFILE=<dir>` | NVTX / torch profiler 埋点 |

### 3.4 常用命令行参数

| 参数 | 建议 | 说明 |
|---|---|---|
| `--tensor-parallel-size` | `1`(单卡)/`2`(TP=2 需 `XIAOTU_MOE_SINGLECOPY=1`) | 本插件当前**不支持 expert_map(EP)**,TP>1 走的是权重分片 |
| `--max-model-len` | 先 `4096`,再按 KV 容量放大 | KV ≈ 400 KiB/token(DS-V4) |
| `--gpu-memory-utilization` | `0.85` | 非专家权重 + KV 在显存 |
| `--enforce-eager` | 建议先开 | 避免 CUDA graph 与 CPU 引擎 host 回调的额外变量;稳定后可试关 |
| `--kernel-config.enable_jit_warmup=false` | 建议 | 跳过 JIT 预热,加快启动 |
| `--trust-remote-code` | 一般不需要 | DS-V4/GLM/Qwen3.8 的配置已进主线 |

---

## 4. 跑 DeepSeek-V4-Flash(单卡,已验证)

模型目录(本机):`/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master`

### 4.1 HTTP 服务

```bash
export CUDA_VISIBLE_DEVICES=2
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export XIAOTU_MOE_SINGLECOPY=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master \
  --host 0.0.0.0 --port 8071 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --kernel-config.enable_jit_warmup=false \
  --served-model-name DeepSeek-V4-Flash-xiaotu
```

等价脚本:`bash scripts/serve_8071.sh`(可用 `MODEL=/PORT=/MAXLEN=/SEQS=` 覆盖)。

### 4.2 离线 Python API

```python
import os
os.environ.update(CUDA_VISIBLE_DEVICES="2", VLLM_EXPERTS_LOAD_DEVICE="cpu",
                  XIAOTU_MOE_SINGLECOPY="1", HF_HUB_OFFLINE="1")
from vllm import LLM, SamplingParams
llm = LLM(model="/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master",
          tensor_parallel_size=1, gpu_memory_utilization=0.85,
          max_model_len=4096, max_num_seqs=16, enforce_eager=True,
          kernel_config={"enable_jit_warmup": False})
print(llm.generate(["1+1 等于几?"], SamplingParams(max_tokens=16, temperature=0))[0].outputs[0].text)
```

参考脚本:`scripts/dsv4_longprefill_test.py`、`scripts/dsv4_real_test.py`、`scripts/bench_llm.py`。

### 4.3 长 prefill 走 GPU(16K 上下文起明显)

```bash
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384     # 0 = 全 CPU
```

实测(1×A100,DS-V4):2 091 token 5.64 s、4 137 token 5.76 s、8 229 token 8.64 s、
16 039 token 18.24 s(≈880 tok/s);TP=2 + 专家分片:16 039 token 13.96 s(≈1 149 tok/s)。
纯 CPU prefill 对照:2 091 token 112.9 s。

---

## 5. 跑 Qwen3.8-Flash-Next-FP8(185 GB 专家,混合模式的目标用例)

> 仓库 id:`Qwen/Qwen3.8-Flash-Next-FP8`(bf16 版是 `Qwen/Qwen3.8-Flash-Next`)。
> 注意:线上**没有**叫 `Qwen/Qwen3.8-Flash` 的仓库(2026-09-09 核实)。
> 架构 `Qwen4ExpForConditionalGeneration`(48 层 / 512 专家 / top-10 / hidden 2560 /
> moe_inter 640 / fp8 e4m3 block-128),主线已支持该架构。

```bash
# 下载(hf-mirror 直连,不要走代理;约 185 GB)
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen3.8-Flash-Next-FP8

export CUDA_VISIBLE_DEVICES=2
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export XIAOTU_MOE_SINGLECOPY=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --kernel-config.enable_jit_warmup=false
```

离线脚本(自带计时与连贯性检查):

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_MOE_SINGLECOPY=1 \
  SMOKE_MODEL=Qwen/Qwen3.8-Flash-Next-FP8 python scripts/fp8_moe_smoke.py
```

> **状态(2026-09-09)**:该模型的端到端验证**正在进行**。已知:主线支持该架构;
> QSA 稀疏注意力是 Triton 实现、GDN 线性注意力也是 Triton,未发现 SM90 硬门槛。
> 若 mm processor 报错,可加 `--limit-mm-per-prompt '{"image":0,"video":0}'` 只跑文本。

---

## 6. GLM-5.3-Flash(仅作参考:本机 A100 **跑不了**)

`zai-org/GLM-5.3-Flash`(306 GB)需要 SM90+:
其 MLA 维度 `qk_nope=256 / rope=0 / v=256` 在主线没有任何 attention 后端支持
(证据:`docs/evidence/glm53_a100_blocker.txt`)。**这不是插件问题** ——
本插件对该模型的**专家层**数值已验证(真实权重 rms_rel 9.3e-5,见
`scripts/test_glm53_fp8_layer.py`)。在 H100/B200 上按 §5 同样方式启动即可。

---

## 7. 自检 / 冒烟 / 排错

| 目的 | 命令 |
|---|---|
| 插件与 shim 自检 | `VLLM_EXPERTS_LOAD_DEVICE=cpu python -m vllm_xiaotu_moe.mainline_shims` |
| oracle 是否选中我们的后端 | `VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py` |
| CPU 专家 vs GPU 专家数值等价(微型模型,分钟级) | `python scripts/make_tiny_mixtral.py` 然后 `MOE_MODE=gpu/cpu python scripts/tiny_moe_equiv.py` |
| 真实 fp8 模型端到端 | `python scripts/fp8_moe_smoke.py` |
| 层内数值自校验(真实模型) | 加 `XIAOTU_VERIFY_LAYER=1` 跑上面的脚本 |
| FP8 引擎吞吐微基准 | `python scripts/bench_fp8_engine.py 128 2048 768 8` |

常见问题:

| 现象 | 原因 / 处理 |
|---|---|
| 构造期 OOM / `b_q_weight is not on GPU` | 没设 `VLLM_EXPERTS_LOAD_DEVICE=cpu`;或没装插件(见 §2 自检) |
| `AttributeError: '_OpNamespace' '_C' object has no attribute 'convert_weight_packed'` | 主线 CPU 后端的 AMX 重打包被调用了 → 确认 `XIAOTU_MAINLINE_SHIMS=1` 且 `VLLM_EXPERTS_LOAD_DEVICE=cpu` |
| `RuntimeError: XiaotuCPUExperts.apply called before process_weights_after_loading` | 同上(shim 4 没生效);检查 vLLM 版本是否被 shim 覆盖 |
| 启动超时 `Engine core initialization failed` | `VLLM_ENGINE_READY_TIMEOUT_S=3600+`;首次加载 306 GB 级模型需要几分钟 |
| 内存翻倍 / node 0 满 | 设 `XIAOTU_MOE_SINGLECOPY=1` |
| 输出不连贯 | 加 `XIAOTU_VERIFY_LAYER=1` 看每层 `rel_rms`;同时用 `XIAOTU_MOE_PROFILE=1` 看耗时 |
| 想让长 prefill 快 | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384`(见 §3.2) |

---

## 8. 修订记录

- **2026-09-09(第 1 版)** — 新建。按用户要求给出:主线 vLLM 简要安装、插件安装、
  环境变量/命令行参数说明、DS-V4-Flash 与 Qwen3.8-Flash-Next-FP8 的参考运行命令、
  自检与排错表。依据:本机实测(`scripts/serve_8071.sh`、`scripts/fp8_moe_smoke.py`、
  `docs/EXPERIMENT_REPORT.md`、`docs/BACKLOG.md` L1–L9)。
