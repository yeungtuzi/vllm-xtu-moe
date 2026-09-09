# RUNBOOK — 安装、配置与运行

本手册给出:①安装主线 vLLM ②安装本插件 ③环境变量与命令行参数 ④常见模型的
参考启动命令 ⑤自检与排错。

参考环境(验证组合):Python 3.12 · torch 2.13.0+cu130 · vLLM 主线
`0.1.dev1+g6c73b08de` · NVIDIA A100-PCIE-40GB(SM 8.0)· AMD EPYC 9654(无 AMX)。
较新的 vLLM mainline 一般同样可用;若集成点发生变化,插件会打印明确的告警。

---

## 1. 安装主线 vLLM

本插件通过官方 `vllm.general_plugins` 入口加载,**不需要 fork**。混合模式所需的
几处主线配合由插件自带(`vllm_xiaotu_moe/mainline_shims.py`)以 monkey-patch
方式提供(见 [`ARCHITECTURE.md`](ARCHITECTURE.md))。

**方式 A:pip(最简)**

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130
```

**方式 B:源码(可控,推荐与已验证提交对齐)**

```bash
python -m venv .venv && source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout 6c73b08          # 已验证提交;更新的 main 通常也可以
VLLM_USE_PRECOMPILED=1 pip install -e .  # 复用预编译算子,避免长时间编译
```

> `VLLM_USE_PRECOMPILED=1` 会下载官方预编译算子包;若目标平台没有对应的
> 预编译包,则退回源码编译(耗时较长)。

---

## 2. 安装 vllm-xtu-moe

```bash
git clone https://github.com/yeungtuzi/vllm-xtu-moe.git
cd vllm-xtu-moe

# 2a) 构建内置 CPU 引擎的原生扩展(生成 5 个 ISA 变体,约 90 秒)
PYTHON=$(which python) bash scripts/build_engine_variants.sh
#    产物: xiaotu_moe/build/_xiaotu_moe_C_{scalar,avx2,avx512_base,avx512_vnni,avx512_bf16}.so
#    运行时由 xiaotu_moe/loader.py 按 /proc/cpuinfo 自动选择最高可用变体

# 2b) 安装插件
pip install -e .
#    也可以 pip install vllm-xtu-moe(发布包已内含预编译引擎 .so,无需 2a)
```

**自检**:

```bash
VLLM_EXPERTS_LOAD_DEVICE=cpu python -m vllm_xiaotu_moe.mainline_shims
VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py     # 不加载权重,秒级
```

期望输出(节选):

```
mixed_mode_enabled = True
xiaotu_moe variant = _avx512_bf16
[vllm-xtu-moe/shims] mainline shims applied (...)
[OK ] bf16       -> CPU  mixed_experts.XiaotuCPUExpertsBF16
[OK ] fp8-glm53  -> CPU  mixed_experts.XiaotuCPUExpertsFp8
[OK ] fp8-dsv4   -> CPU  mixed_experts.XiaotuCPUExpertsFp8
[OK ] wna16-int4 -> is_supported_config=True
```

---

## 3. 环境变量与命令行参数

### 3.1 必开 / 常用

| 变量 | 推荐值 | 作用 |
|---|---|---|
| `VLLM_EXPERTS_LOAD_DEVICE` | `cpu` | ★ 混合模式开关:routed-expert 权重在 CPU 上构造与计算 |
| `XIAOTU_MOE_SINGLECOPY` | `1` | 权重只保留一份(NUMA 分片);不设时可能按 socket 复制,内存约 2× |
| `XIAOTU_MAINLINE_SHIMS` | `1`(默认) | 在原生 vLLM 上启用混合模式所需的集成补丁;`0` 关闭 |
| `CUDA_VISIBLE_DEVICES` | 例如 `0` | 选择使用的 GPU |
| `HF_HUB_OFFLINE` | `1` | 离线环境(权重已在本地) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | 关闭 FlashInfer 采样器(避免额外依赖) |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600`–`7200` | 大模型 + CPU 专家初始化较慢,启动等待时间需放大 |

### 3.2 长 prefill 走 GPU(可选)

| 变量 | 推荐值 | 作用 |
|---|---|---|
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | `384`(默认)或 `512`–`768` | 单层 prefill token 数达到阈值时,该层专家改为**逐层流式 GPU 计算**;`0` 关闭(全部 CPU) |
| `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE` | 文件路径 | 客户端进程设置环境变量不生效时,从文件读取阈值 |
| `XIAOTU_GPU_PREFETCH_AHEAD` | `1` | 预取下一层的权重(双槽流水) |

阈值推荐与实测见 [`BENCHMARKS.md`](BENCHMARKS.md);机制见
[`GPU_PREFILL.md`](GPU_PREFILL.md)。

### 3.3 观测 / 调试(默认关闭)

| 变量 | 作用 |
|---|---|
| `XIAOTU_MOE_PROFILE=1` | 引擎相位计时(gate/up、down、组合),周期性打印 |
| `XIAOTU_VERIFY_LAYER=1` | 层内数值自校验:每层首次调用时用 torch 参考实现复算并打印 `rel_rms` |
| `XIAOTU_MOE_THREADS=<n>` | 引擎线程数(默认使用共享 NUMA 池的全部核心) |
| `XIAOTU_MOE_NOSHARD=1` | 关闭权重 NUMA 分片(调试用) |
| `XIAOTU_SYNC_DECODE` | 默认 `0`;设 `1` 时在引擎调用前后与设备同步(排查时序问题用,会降低吞吐) |
| `XIAOTU_NVTX=1` / `XIAOTU_TORCH_PROFILE=<dir>` | NVTX / torch profiler 埋点 |

### 3.4 常用命令行参数

| 参数 | 建议 | 说明 |
|---|---|---|
| `--tensor-parallel-size` | `1`(单卡)或 `2`(配合 `XIAOTU_MOE_SINGLECOPY=1`) | 目前**不支持 expert_map(EP)**,TP>1 走权重分片 |
| `--max-model-len` | 先 `4096`–`8192`,再按 KV 容量放大 | KV 占用随模型不同(DeepSeek-V4 约 400 KiB/token) |
| `--gpu-memory-utilization` | `0.85` | 非专家权重 + KV cache 在显存 |
| `--enforce-eager` | 建议先开 | 避免 CUDA graph 与 CPU 引擎 host 回调的额外变量;稳定后可尝试关闭 |
| `--kernel-config.enable_jit_warmup=false` | 建议 | 跳过 JIT 预热,加快启动 |
| `--trust-remote-code` | 通常不需要 | 主流模型配置已进入主线 |

---

## 4. 参考启动命令

### 4.1 DeepSeek-V4-Flash(单卡)

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export XIAOTU_MOE_SINGLECOPY=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve <DEEPSEEK_V4_FLASH_DIR> \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --kernel-config.enable_jit_warmup=false \
  --served-model-name DeepSeek-V4-Flash
```

离线 Python API:

```python
import os
os.environ.update(VLLM_EXPERTS_LOAD_DEVICE="cpu", XIAOTU_MOE_SINGLECOPY="1",
                  CUDA_VISIBLE_DEVICES="0")
from vllm import LLM, SamplingParams

llm = LLM(model="<DEEPSEEK_V4_FLASH_DIR>",
          tensor_parallel_size=1, gpu_memory_utilization=0.85,
          max_model_len=4096, max_num_seqs=16, enforce_eager=True,
          kernel_config={"enable_jit_warmup": False})
print(llm.generate(["1+1 = ?"], SamplingParams(max_tokens=16, temperature=0))[0].outputs[0].text)
```

长 prefill 走 GPU:

```bash
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384     # 0 = 全部 CPU
```

### 4.2 Qwen3.8-Flash-Next-FP8(专家权重 185 GB,混合模式的典型用例)

该模型的权重构成:**专家 ~120 GB(fp8)** + **非专家 ~65 GB**,其中非专家里有一张
约 **51 GB 的 PLE n-gram 嵌入表**——因此 2×A100-40GB 上**必须**用 TP=2 专家并行,
并把一部分非专家权重 offload 到 CPU(`--cpu-offload-gb`),否则显存放不下 KV cache。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export XIAOTU_MOE_SINGLECOPY=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --cpu-offload-gb 12 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --kernel-config.enable_jit_warmup=false
```

> 单卡(40 GB)放不下非专家权重,会报 `No available memory for the cache blocks`;
> 此时增大 `--gpu-memory-utilization` 或加大 `--cpu-offload-gb`。
> CPU 侧内存需求:每 rank 约 60 GB 专家权重 + 引擎快照一份(共约 240 GB)。

离线冒烟脚本(自带计时与连贯性检查):

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_MOE_SINGLECOPY=1 \
  SMOKE_MODEL=Qwen/Qwen3.8-Flash-Next-FP8 python scripts/fp8_moe_smoke.py
```

> 该架构为 `Qwen4ExpForConditionalGeneration`(48 层 / 512 专家 / top-10 /
> hidden 2560 / moe_inter 640 / fp8 e4m3 block-128),主线已支持。
> 若多模态处理器报错,可加 `--limit-mm-per-prompt '{"image":0,"video":0}'` 只跑文本。

### 4.3 其它 MoE 模型

任何使用 `FusedMoEFactory` 的 MoE 模型(BF16 / FP8 / MXFP4 / INT4)都可以用同样
方式启动,只需把 `--max-model-len`、`--max-num-seqs` 按显存与 KV 需求调整。
后端是否被正确选中,用 `scripts/probe_oracle.py` 或启动日志中的
`Using CPU ... MoE backend` 行确认。

---

## 5. 自检 / 冒烟 / 排错

| 目的 | 命令 |
|---|---|
| 插件与集成补丁自检 | `VLLM_EXPERTS_LOAD_DEVICE=cpu python -m vllm_xiaotu_moe.mainline_shims` |
| 后端选择探测(不加载权重) | `VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py` |
| CPU 专家 vs GPU 专家数值等价(微型模型,分钟级) | `python scripts/make_tiny_mixtral.py`,再 `MOE_MODE=gpu` / `MOE_MODE=cpu python scripts/tiny_moe_equiv.py` |
| 真实 fp8 模型端到端 | `python scripts/fp8_moe_smoke.py` |
| 层内数值自校验 | 在上面的命令前加 `XIAOTU_VERIFY_LAYER=1` |
| FP8 引擎吞吐微基准 | `python scripts/bench_fp8_engine.py 128 2048 768 8` |
| CPU 引擎吞吐微基准 | `python scripts/bench_cpu_engine.py` |

| 现象 | 原因 / 处理 |
|---|---|
| 构造期 OOM 或 `b_q_weight is not on GPU` | 未设置 `VLLM_EXPERTS_LOAD_DEVICE=cpu`,或插件未安装/未加载 |
| `AttributeError: '_OpNamespace' '_C' object has no attribute 'convert_weight_packed'` | 主线 CPU 后端的 AMX 重打包被触发 → 确认 `VLLM_EXPERTS_LOAD_DEVICE=cpu` 且 `XIAOTU_MAINLINE_SHIMS=1` |
| `RuntimeError: XiaotuCPUExperts.apply called before process_weights_after_loading` | 集成补丁未生效(检查 vLLM 版本是否被 `mainline_shims` 覆盖) |
| 启动超时 `Engine core initialization failed` | 放大 `VLLM_ENGINE_READY_TIMEOUT_S`;超大模型首次加载需要数分钟 |
| 内存占用翻倍 / 某个 NUMA 节点被占满 | 设置 `XIAOTU_MOE_SINGLECOPY=1` |
| 输出不连贯 | 用 `XIAOTU_VERIFY_LAYER=1` 查看每层 `rel_rms`,并用 `XIAOTU_MOE_PROFILE=1` 看耗时分布 |
| 想加速长 prefill | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384` |
