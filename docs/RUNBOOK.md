# RUNBOOK — 安装、配置与运行

本手册给出:①安装主线 vLLM ②安装本插件 ③环境变量与命令行参数 ④常见模型的
参考启动命令 ⑤自检与排错。

参考环境(验证组合):Python 3.12 · torch 2.13.0+cu130 · vLLM 主线
**`0.29.1rc1.dev95+gdabc4362b`(commit `dabc4362b`,2026-09-14)** ·
NVIDIA A100-PCIE-40GB(SM 8.0)· AMD EPYC 9654(无 AMX)。

> 上一版针对 `0.1.dev1+g6c73b08de`(2026-09-08)。**本机不能从源码编译**
> (nvcc 12.1 与 torch cu130 不匹配、且无 Rust 工具链),所以升级主线**必须挑一个
> 已发布 precompiled wheel 的 commit**;`scripts/check_upstream_drift.sh` 会直接告诉你
> 当前上游 HEAD 有没有轮子。若集成点变化,插件会打印明确的告警(见 IRON_RULES R10)。

---

## 1. 安装主线 vLLM

本插件通过官方 `vllm.general_plugins` 入口加载,**不需要 fork**。混合模式所需的
几处主线配合由插件自带(`vllm_xiaotu_moe/mainline_shims.py`)以 monkey-patch
方式提供(见 `ARCHITECTURE.md`)。

**方式 A:pip(最简)**

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130
```

**方式 B:源码(可控,推荐与已验证提交对齐)**

```bash
python -m venv .venv && source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout dabc4362b       # 已验证提交(需有 precompiled wheel;见上方说明)
VLLM_USE_PRECOMPILED=1 pip install -e .  # 复用预编译算子,避免长时间编译
```

> `VLLM_USE_PRECOMPILED=1` 会下载官方预编译算子包;若目标平台没有对应的
> 预编译包,则退回源码编译(耗时较长)。

---

## 2. 安装 vllm-xtu-moe

**方式 A(推荐,v0.2.2 起):直接装 whl,不需要本地编译器** —— GitHub Release 页面
([`v0.2.2`](https://github.com/yeungtuzi/vllm-xtu-moe/releases/tag/v0.2.2))附带

```
vllm_xtu_moe-0.2.2-cp312-cp312-manylinux_2_34_x86_64.whl
```

```bash
pip install ./vllm_xtu_moe-0.2.2-cp312-cp312-manylinux_2_34_x86_64.whl   # vLLM 需已安装
```

* wheel 里**已经打包好 6 个 ISA 变体**(`scalar/avx2/avx512_base/avx512_vnni/avx512_bf16/
  avx512_bf16_vbmi`,g++-16 构建),运行时按 `/proc/cpuinfo` 自动选,所以省掉 2a/2a' 的构建;
* 约束:**Python 3.12 + Linux x86-64**(glibc ≥ 2.34),且环境里要有 `libcudart.so.12`
  (装了 vLLM/torch 的环境天然满足;`.so` 只依赖 CUDA runtime 与驱动);
* 想改内核开关(2a' 里那些)或换编译器时,再走下面的源码方式。

**方式 B:源码安装(需要构建引擎)**

```bash
git clone https://github.com/yeungtuzi/vllm-xtu-moe.git
cd vllm-xtu-moe

# 2a) 构建内置 CPU 引擎的原生扩展(生成 6 个 ISA 变体,约 3-6 分钟)
PYTHON=$(which python) bash scripts/build_engine_variants.sh
#    产物: xiaotu_moe/build/_xiaotu_moe_C_{scalar,avx2,avx512_base,avx512_vnni,avx512_bf16,avx512_bf16_vbmi}.so
#    运行时由 xiaotu_moe/loader.py 按 /proc/cpuinfo 自动选择最高可用变体

# 2a')【§630/§631】编译器与调优开关(都可选,留空用默认)
#    CXX=g++-16           编译器。**§630 实测:g++-16 比 g++-11 快 ~5%(大预填充形状)**;
#                         clang++-23 无优势(大形状反而慢 1.6%)⇒ 中选用 g++-16(trunk,
#                         实验性版本;若不可用退回 g++-11,行为正确但慢 ~5%)。
#    MTUNE=znver4         `-mtune` 只改调度、不改 ISA(绝不能改成 `-march=znver4`,
#                         那会把 AVX-512 塞进 scalar/avx2 兜底变体)。**§631 实测单独用会更差,
#                         故默认不中选**;仅在 `XIAOTU_MOE_FOLD_SCALE=1` 时对小结/BS=227
#                         有 −3.4% 额外收益。`g++-11` 不认识 `znver4`。
#    EXTRA_FLAGS=...      透传宏,例如 `-DXIAOTU_MOE_FOLD_SCALE=0` 回到 §631 之前的数值路径。
#
#    内核开关(源码默认值,见 xiaotu_moe/csrc/moe/moe_v2_packed4.hpp 顶部)。**v0.2.1 中选值**:
#      XIAOTU_MOE_FOLD_SCALE  默认 **1**。把每 (列,K 组) 的 block scale 折到权重侧,
#                             消掉每个 (行,列,组) 多出来的第 3 条 FMA 端口指令
#                             ⇒ 大预填充 −8.2~−8.4%。**只在 MRT>=2 时生效**(mr=1 时旧式更省一条
#                             FMA 端口指令)。数值门 `OK=7 BAD=1` / `me=1 max_rel=1.873e-02`
#                             **逐位不变**,确定性 10/10。
#      XIAOTU_MOE_GEMM_MR     默认 **6**。行块高度 = "一次解码出来的权重喂给几行 token"。
#                             真实负载每专家 M≈6 ⇒ MR=6 让权重**只读一遍**(MR=4 要两遍)
#                             ⇒ BS=227 −13.6%。
#      XIAOTU_MOE_GEMM_NR_DEFAULT 默认 **2**;`XIAOTU_MOE_GEMM_NR` 仍可运行时覆盖(=0 回退 GEMV 路径)。
#      XIAOTU_MOE_GEMM_NR_CT  默认 **2**。把 NR 变成**编译期**常量 ⇒ 所有 `acc[MR][N]` 数组
#                             从 `[6][8]`(48 个 zmm)收紧到 `[6][2]`(12 个)⇒ 通用回退路径不再溢出。
#      XIAOTU_MOE_TILE_ALLMR  默认 **1**。tile 特化覆盖**所有** mr(循环边界写成字面量,
#                             数组才有资格进 zmm)。`=3` 时 mr=4/5 掉进通用回退路径,
#                             M=4/5 实测 1.02/1.09 ms/层;`=1` 后 0.63/0.70(−38%/−36%)。
#      ⚠️ M(=每专家行数 = batch token×top_k/活跃专家数)决定最优 MR:`MR=6` 对
#         V4.1(M≈B/64)、V4(M≈B/42.7)、GLM-5.3(M≈B/36)三者都最优或近似最优;
#         推导与按模型定制的命令见内部《GLM-5.3-Flash 分析》§3(不随仓库发布)。
#  示例:`CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh`
#  验证:`bash scripts/ab_compilers.sh gcc16`(内部调优记录见《与 lk_moe 的 A/B》轮 4-7)

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
| `VLLM_EXPERTS_LOAD_DEVICE` | `cpu` | ★ 混合模式开关:routed-expert 权重在 CPU 上构造与计算。**取值只有 `cpu` / `gpu`**(写 `cuda` 会被 vLLM 直接拒绝);`gpu` 用于同模型对照基线 |
| (无开关) 权重布局 | **固定** | 权重按 NUMA node 分片,每个 node 的 worker 只读写本 node 绑定的一份(page-local);node 间只交换很小的激活切片/部分和。原先的 `XIAOTU_MOE_SINGLECOPY` 单拷贝模式**已从代码中删除** |
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
`GPU_PREFILL.md`。

### 3.3 观测 / 调试(默认关闭)

| 变量 | 作用 |
|---|---|
| `XIAOTU_MOE_PROFILE=1` | 引擎相位计时(gate/up、down、组合),周期性打印 |
| `XIAOTU_VERIFY_LAYER=1` | 层内数值自校验:每层首次调用时用 torch 参考实现复算并打印 `rel_rms` |
| `XIAOTU_MOE_THREADS=<n>` | 引擎线程数(默认使用共享 NUMA 池的全部核心) |
| `XIAOTU_MOE_NSLICE_SMALL=0` | 关闭小批量 N-切片(退回逐 token 路径,用于 A/B 对比;两条路径结果逐位一致) |
| `XIAOTU_MOE_NOSHARD=1` | 关闭权重 NUMA 分片(调试用) |
| `XIAOTU_SYNC_DECODE` | 默认 `0`;设 `1` 时在引擎调用前后与设备同步(排查时序问题用,会降低吞吐) |
| `XIAOTU_NVTX=1` / `XIAOTU_TORCH_PROFILE=<dir>` | NVTX / torch profiler 埋点 |

### 3.4 常用命令行参数

| 参数 | 建议 | 说明 |
|---|---|---|
| `--tensor-parallel-size` | `1`(单卡)或 `2` | 目前**不支持 expert_map(EP)**,TP>1 走权重分片 |
| `--max-model-len` | GLM-5.3:**交付默认 262144(256K)**,实测可起 512K/704K;DeepSeek-V4 系列先 `4096`–`8192` 再放大 | KV 单价差异极大:**GLM-5.3 ≈ 11.9-12.3 KB/token**,DeepSeek-V4.1 ≈ 40 KB/token |
| `--gpu-memory-utilization` | `0.85` | 非专家权重 + KV cache 在显存 |
| `--enforce-eager` | 建议先开 | 避免 CUDA graph 与 CPU 引擎 host 回调的额外变量;稳定后可尝试关闭 |
| `--kernel-config.enable_jit_warmup=false` | 建议 | 跳过 JIT 预热,加快启动 |
| `--trust-remote-code` | 通常不需要 | 主流模型配置已进入主线 |
| `JITCACHE=1`(env,默认) | **保持开启** | 把编译缓存钉到固定目录,见 §3.5 |

### 3.5 JIT 固定缓存目录(`JITCACHE`)

> **背景(v0.1 起)**:每次启动都重新 JIT 太慢,所以把编译缓存固定到一个目录,跨重启复用。

**症状**:用 `--compilation-config '{"mode":"VLLM_COMPILE",…}'` 时,**每换一个旗标、每改一行我们
的代码,启动后第一个请求就把逐形状的 Triton 内核从头编一遍**(`jit_monitor` 警告 20-60 s/形状)。

**原因**(安装好的 vLLM 源码里逐行核实):缓存目录名 = 四个 hash 拼出来的
`$VLLM_CACHE_ROOT/torch_compile_cache/<hash10>`,而其中 `env_hash` 覆盖**每一个 `VLLM_*` 环境变量**、
`code_hash` 覆盖**被 trace 的源码(含我们插件替换的模型类)**;并且
`CompilerInterface.initialize_cache()` 会把 `TRITON_CACHE_DIR` **重定向**到那个 hash 目录里,
所以 `~/.triton/cache` 攒下的内核用不上。

**做法**:三个启动器(`scripts/serve_v41.sh`、`scripts/serve_mainline.sh`、
`<内部探针>/xtu_own_v41_mem.sh`)都 source `scripts/lib_jitcache.sh`:

| 旋钮 | 默认 | 含义 |
|---|---|---|
| `JITCACHE` | `1` | `1` = 钉住固定目录;`0` = 回到 vLLM 默认的 hash 行为 |
| `XIAOTU_JIT_CACHE_DIR` | `~/.cache/vllm/torch_compile_cache` | 固定目录的**根**(默认就是 vLLM 自己的根,即参考实现 lk 的设定) |
| `JITCACHE_STAMP` | 空 | 追加到目录名,用来**强制重建** |
| `JITCACHE_COMPILING` | 由 `COMPILE` 决定 | 只有走编译时才真的改 `TRITON_CACHE_DIR` 等 |

目录名形如
`…/torch_compile_cache/xtu-DeepSeek-V4.1-Flash-tp2-vllm_compile-6b5ef34f7b-82747171`,
即 **模型 + TP + 编译模式 + vLLM commit + 我们源码的 sha1**。所以
* 重复启动 / 换旗标扫描 ⇒ **命中同一目录**;
* 我们自己改了插件或引擎源码、上游 vLLM 动了 ⇒ **自动换新目录**(不会读到陈旧计算图)。

日志里会出现 `[jitcache] 固定缓存目录 = …`;若该行显示"本次不编译 ⇒ 不改 Triton 目录",
说明这一跑是 `mode=NONE`,Triton 用的仍是 `~/.triton/cache`(本来就固定持久)。

---

> **DeepSeek-V4-Flash 的逐步指南 + 初步性能数据**见 [`MODEL_GUIDES.md`](MODEL_GUIDES.md);
> 本节只给最简参考命令。

## 4. 参考启动命令

### 4.1 DeepSeek-V4-Flash(单卡)

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_EXPERTS_LOAD_DEVICE=cpu
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
os.environ.update(VLLM_EXPERTS_LOAD_DEVICE="cpu", CUDA_VISIBLE_DEVICES="0")
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

### 4.2 目标 FP8 模型(暂不声明支持)

> ⏸️ **暂不声明支持**:该模型的实测数据早于 v0.2 的改动(执行模型 / 小 batch 路径 / EP 存储分片),**未在当前代码上复验**。复验计划见 `内部交接 HANDOFF_v0.2pre.md` §5.1。

### 4.3 其它 MoE 模型

任何使用 `FusedMoEFactory` 的 MoE 模型(BF16 / FP8 / MXFP4 / INT4)都可以用同样
方式启动,只需把 `--max-model-len`、`--max-num-seqs` 按显存与 KV 需求调整(GLM-5.3-Flash 见 `scripts/serve_glm53_mainline.sh`:默认 **256K × 2 路** + GPU 预填充)。
后端是否被正确选中,用 `scripts/probe_oracle.py` 或启动日志中的
`Using CPU ... MoE backend` 行确认。

---

## 4b. 生产服务(8070)—— 两种模式,按"能不能交互"选

```bash
bash scripts/serve_prod_8070.sh                       # MODE=1m :TP=2 + 1M 上下文 + DSpark(默认)
MODE=fast PORT=8090 bash scripts/serve_prod_8070.sh   # MODE=fast:单卡 + 256K + DSpark
SPEC_OFF=1 bash scripts/serve_prod_8070.sh            # 关投机:高并发吞吐优先
```

### 实测对比(2026-09-10,全部本机实跑)

| 口径 | **fast**(单卡 256K + DSpark) | **1m**(TP=2 1M + DSpark) | 单卡 256K 无投机(CG) | lk-moe 生产(双卡+DSpark) |
|---|---|---|---|---|
| **C=1 单路 TPOT** | **85 ms(11.8 tok/s/路)** | 266 ms(3.7 tok/s/路) | ~128 ms | — |
| C=1 TTFT | 0.63 s | 2.2 s | 2.85 s(C=4) | — |
| C=4 Output / Total | 23.60 / 52.82 | 待测 | 25.91 / 57.84 | **46.67 / 105.74** |
| C=4 TPOT | 171 ms | — | 128.6 ms | **40.2 ms** |
| C=64 Output | — | 48 左右(无投机) | 90.60 | — |
| C=128 Output | — | — | **106.15** | — |
| 上下文 | 256K | **1M(KV 1,876,112 tokens)** | 256K | — |

**怎么选**:
- **日常交互**(聊天/写代码):用 `MODE=fast`,单路 85 ms 出词,手感接近可用;
- **长文本/1M 上下文**:用默认的 1M 模式,但要接受 3.7 tok/s/路(比 fast 慢 3 倍);
- **批处理/高吞吐**:`SPEC_OFF=1` + 高并发(C≥64),单卡可到 106 tok/s。

**为什么 1M 一定要 TP=2**:KV = 29.5 KB/token,1M 就是 29.5 GiB;单卡扣掉 ~19.6 GB 非专家
权重后放不下;FP4 KV(`nvfp4_ds_mla`)在 A100/SM80 被主线拒绝,512K 单卡也会 OOM。

**已知问题**:TP=2 的每层跨 rank 同步使低并发延迟变差(EP=0 时 qlen=1 的每层
`period 4.63 ms = compute 0.86 + rest 3.77`);解法见内部《性能优化》§7
(CPU-TP:把合并放进引擎内部,而不是每层一次集合通信)。

**健康检查**:

```bash
curl -s http://127.0.0.1:8070/v1/models | head -c 200
curl -s http://127.0.0.1:8070/metrics | grep -E "num_requests_running|spec_decode_num_accepted"
```

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
| 内存占用翻倍 / 某个 NUMA 节点被占满 | 检查是否有残留进程占着分片内存;分片布局不可关闭(单拷贝模式已删除) |
| 输出不连贯 | 用 `XIAOTU_VERIFY_LAYER=1` 查看每层 `rel_rms`,并用 `XIAOTU_MOE_PROFILE=1` 看耗时分布 |
| 想加速长 prefill | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384` |

---

## 5. DeepSeek-V4.1-Flash 运行手册(v2026-09-18;数据出处见 `内部调优记录 NOTES.md` §564-§578)

### 5.1 中选默认(不要凭记忆改,以下为实测过的组合)

| 项 | 默认 | 依据 |
|---|---|---|
| TP | **2** | 单卡放不下;TP=1 需显式说明差异(§507) |
| 分片单位 | **自适应 node**(NPS4 上 = 8 片) | socket 分片在 NPS4 上慢 +29%(§505/§518/§519) |
| `XIAOTU_MOE_SPIN_IDLE_US` | **300** | 5000 = 正反馈灾难;0 更慢(§356/R14;本机 1.84 vs 1.14 ms/层) |
| `XIAOTU_ENGRAM_LAST` | **1**(2026-09-17 起) | 峰值 **1087.6 → 642 GiB(−41%)**;正确性已闭环(§563-565) |
| `XIAOTU_ENGRAM_VERIFY` | **1** | 表内容逐 chunk 抽验、失败 fail-closed(§564) |
| GPU 预填充 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | 由 `vram_policy` 给(**4096**;§603 按实测盈亏平衡 ~2860 token 定) | GPU 路径非逐位确定(§572),但更快;低于 ~2860 反而更慢 |
| 投机解码 | `--speculative-config dspark` | R-VRAM 优先级 3 |

### 5.2 R-VRAM 策略(权威输出)

```bash
python -m vllm_xiaotu_moe.vram_policy --maxlen 1048576 --max-num-seqs 2 --tp 2 --gpu-mem-util 0.55
```
```
[vram-policy] maxlen=1048576 × 2 并发  util=0.55  ⇒ 填池后净空 17.8 GiB(预填充 preflight 要 8.4)
✅ 1. 1M 上下文(KV)        4.4 GiB/卡(maxlen=1048576 × 2 并发)
✅ 2. GPU 预填充          8.4 GiB(staging 6.72 × 1.25)
✅ 3. GPU 投机解码        3.7 GiB
❌ 4. 专家层常驻          0 层(§584 实测:常驻收益为负 ⇒ 显存改投 KV)
```
任一项不足即按优先级 fallback;**绝不额外多占系统内存**。

**输入口径(§592 全部改成"不用填"或"实测值")**
| 量 | 值 | 来源 |
|---|---|---|
| 卡可用显存 | 39.49 GiB | A100-40G 实测 |
| 非 KV 常驻 | **9.78 GiB/卡** | 启动日志 `Actual usage is 9.78 GiB for consumed memory (weights + non-torch)` |
| KV | **2.1 GiB / Mtoken / 卡** | maxlen=1M 三次实测互洽(1.79~2.25)|
| KV 需求量 | `maxlen × 并发数` | 服务承诺"每请求都能用满 maxlen" |
| GPU 预填充预留 | **模型维度现算** = `2×(w13+w2+s13+s2) × 1.25` | 与 preflight 的 `staging_bytes()` 同式 |
| 投机(draft) | 7.388/TP GiB | checkpoint `mtp.0/1/2` |
| 单层专家权重 | staging/2(TP=2 时 3.36) | — |

⚠️ **GPU 预填充要求 `--tensor-parallel-size >= 2`**:TP=1 时 staging 不摊薄(6.72→13.45 GiB,
preflight 要 16.8),单卡必然放不下 ⇒ 策略直接判否(**不指望 TP=1 跑它**)。

### 5.3 启动与验收命令

```bash
# 启动(TP=2/1M/投机/常驻层,真实权重)
TAG=acc1m PORT=8315 MAXLEN=1048576 SEQS=8 GPUS=0,1 SPEC=1 RESIDENT=20-21 GP_MIN=1024 THREADS=60 \
  bash <内部探针>/xtu_own_v41_mem.sh        # 自带内存峰值采样

# 1M 配置的实测(§570):READY 388 s;峰值 629.4 GiB;KV 6,724,586 tokens(1M 并发 6.41×);
# 显存 35.98 GiB/卡(**总占用**;其中 KV 池 12.01 GiB —— 别把总占用当 KV,§592(b));greedy 5/5。

# 三门禁(引擎)
bash scripts/check_engine_aligned.sh
#   数值门:OK=7 BAD=1(max_rel=1.873e-02,me=1 既有偏差)
#   性能门:DEDUP=12 0.70 ms/层(216 GB/s);DEDUP=23 0.85 ms/层(296 GB/s、2.47 GB/s·线程)
#   同日 lk 比值:1.17 / 1.23(阈值 1.30/1.40)
python scripts/test_engine_determinism.py 11     # 11 次运行 10/10 逐位相同
```

### 5.4 环境变量速查(本会话新增/变更)

| 变量 | 默认 | 作用 |
|---|---|---|
| `XIAOTU_ENGRAM_LAST` | **1** | Engram 大表最后加载(峰值 −41%) |
| `XIAOTU_ENGRAM_VERIFY` | 1 | 表内容抽验(fail-closed) |
| `XIAOTU_ENGRAM_ABLATE_FILE` | 未设 | 设了才装"禁注入"消融开关;`touch/rm` 运行时切换(§565) |
| `XIAOTU_MOE_POOL_DEADLINE_MS` | 300000 | flat 路径看门狗截止(调试用毫秒级) |
| `XIAOTU_MOE_SHARD_WD` | 300 | **sharded** 路径看门狗截止(秒) |
| `XIAOTU_MOE_PUB_WINDOW_US` | 0 | 发布窗口放大器(诊断;§567) |
| `PREFIX_CACHE`(探针) | 1 | 0 = `--no-enable-prefix-caching` |
| `KVSHARE`(探针) | 0 | 1 = `--kv-sharing-fast-prefill` |
| `XIAOTU_CED_FASTPREFILL` | **0(未验证完,勿开)** | ③ CED 预填充捷径(§571-578) |
| `XIAOTU_CED_WINDOW` | 255 | CED 尾部长度(255=精确档论证,128=omlx 近似档) |
| `XIAOTU_MOE_VARIANT` | 未设 | 强制 ISA 变体(变体是 pybind11 全局注册,**一变体一进程**) |

### 5.5 验收/诊断工具(内部探针,不随仓库发布)

| 工具 | 用途 |
|---|---|
| `capture_prefill_golden.py` | 抓/比对 **prefill logits 黄金基线**(8 例,含 3 条长 prompt) |
| `ced_ab.py` | ③ CED 的**生成口径** A/B(文本 + 耗时) |
| `prof_capture.py` + `trace_kernels.py` | torch profiler 抓取 + 按 kernel 归因 |
| `engram_ablation.py` | Engram 运行时消融(困惑度 ×5.15 @ 知识密集文本;×1.4 @ 普通文本) |
| `stress_pool.py` / `.sh` | 线程池丢票竞态复现器(修复前 4/4 挂;修后 4×600 s 0 挂) |

### 5.6 已知陷阱(踩过的)

1. **长 prompt 的 prefill logits 本就不逐位可复现**(native 噪声:1K→0.12、4K→0.38 nats;
   首 token 稳定)。**不是**引擎问题,也**不是** prefix cache(§572 两次否证)。⇒ 长序列验收用
   "首 token/贪心文本一致"而不是逐位 logprob。
2. **`--kv-sharing-fast-prefill` 开着会声明 prompt logprobs 不正确**(runner 有硬 assert)。
3. **`XIAOTU_MOE_VARIANT` 只能进程级**:同进程加载两个变体会
   `ImportError: generic_type ... already registered`。
4. **多进程压测要关自旋**(`XIAOTU_MOE_SPIN_IDLE_US=0`),否则 SPIN=5000 会把 CPU 烧在自旋上。
5. **`pkill -f` 会命中自己的命令行** ⇒ 用字符类(`port 832[2]`)。
6. **⭐ `--gpu-memory-utilization` 会把 GPU 预填充"饿死"**(§592,最容易踩的一个):
   vLLM 用 KV 池把显存**填到 util 为止**,所以启动后净空 ≈ `(1−util)×39.49 GiB`
   (util=0.90 ⇒ **3.95 GiB**;util=0.55 ⇒ 17.8 GiB)。而预填充 preflight 要 8.4 GiB
   ⇒ **util ≥ 0.8 时 GPU 预填充必然逐层被拒、静默退回 CPU(且混合模式比纯 CPU 更慢)**。
   解法见 §5.8:**显式给 `--kv-cache-memory` 把 KV 池封顶**。
6b. **kill 服务后必须确认 GPU 显存真的释放**:vLLM 的 worker 常变成**僵尸**(`Zl`, ppid=1)
   并**继续占着显存**(实测两个 worker 各占 37.9 GiB)⇒ 下一个服务会在启动时 OOM。
   做法:`nvidia-smi --query-compute-apps=pid,used_memory` 查到 pid 后**反复 `kill -9`** 并等待,
   直到 `memory.used` 归零再启下一个。(脚本 `/tmp/xtu_killall.sh` 是这套清理。)
7. **长 prompt 的"预热后只要 ~1 秒"是 prefix-cache 命中,不是预填充吞吐**(§591):
   用**完全相同的 prompt** 连发,第 2 次起整段 KV 命中,量到的是"缓存命中 + 首步"。
   凡是报"预填充 tok/s"的行,必须用**每次从头就不同**的 prompt(首 token 就不同)
   且核对 `cached_tokens=0`。正确尺子:`scripts/probe_ttft.py`(默认 `UNIQUE=1`)。

### 5.7 ③ CED 预填充捷径状态(实验,默认关)

* 上游 vLLM **只有通用机制**(`kv_sharing_fast_prefill`),对 V4.1 **零接线**;omlx PR#3607 做了
  "CED prefill skip with SWA bounded replay"(声称 prefill 快 74-79%,长上下文精度仍在评估)。
* 我们已实现(§576/§578,全部 env 门控):①eligible=**19 层(21..39)**;②attention metadata
  窗口改写;③**层循环级隐藏状态切片**;④返回前零填充回全长。判据修正后实测
  `追加 V4.1 eligible 19 层` ✓。
* **尚未取得验收数据**(§570 式三门禁 + 生成等价性 + 预填充提速)。**未验证前不要开**。

### 5.8 ⭐ 让 GPU 预填充真的拿到显存(必读)

**规则:只要开了 GPU 预填充(`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS>0`),
就一定要显式传 `--kv-cache-memory`,不要把 KV 池交给 `--gpu-memory-utilization` 去填。**

```bash
# 1) 先问策略层要那个数(它会按 maxlen×并发 算出**刚好够**的 KV,并把余量留给预填充/投机)
python -m vllm_xiaotu_moe.vram_policy --maxlen 1048576 --max-num-seqs 2 --tp 2 --gpu-mem-util 0.90
#   ⚠️ ... 解法:显式传 `--kv-cache-memory 4729960528` ... 剩下的 15.3 GiB 才留给预填充/投机。
#   ⇒ 直接把这行 emit 出来:eval "$(python -m vllm_xiaotu_moe.vram_policy ... --emit-env)"
#     (serve_v41.sh 已自动接线:`XIAOTU_KV_CACHE_BYTES` → `--kv-cache-memory`)

# 2) 探针里也可以直接钉:
TAG=gpf KV_CACHE_BYTES=4729960528 GP_MIN=1024 MAXLEN=1048576 SEQS=2   bash <内部探针>/xtu_own_v41_mem.sh
```

**为什么要这样**:vLLM 的 KV 池是"把 util 填满"来定尺寸的,与 `maxlen` 无关
⇒ `maxlen=8192` 也会分到 ~11-25 GiB 的池(能装 70 万~160 万 token,**用不到**),
而 GPU 预填充的 staging 需要 8.4 GiB **空闲**显存。池封顶后:
| 配置 | KV 池 | 启动后净空 | GPU 预填充 |
|---|---|---|---|
| 1M×2 / util=0.90(默认) | 35.5 | **3.9** | ❌ 逐层被拒 |
| 1M×2 / `--kv-cache-memory 4.4G` | 4.4 | **15.3** | ✅ + 投机 ✅ |
| 8K×8 / `--kv-cache-memory 0.5G` | 0.5 | **24.3** | ✅ + 投机 ✅ |
| 1M×8 / 封顶后 | 17.6 | 2.1 | ❌ KV 优先(按 R-VRAM 优先级 1 让路)|

**若封顶后仍 OOM**(显存实在不够同时放 KV + 预填充 + 投机),按 R-VRAM 顺序二选一:
1. **关掉 GPU 预填充**:`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=0`(上下文不变,预填充回 CPU);或
2. **缩小上下文**:`--max-model-len` 降到策略给出的建议值(策略会直接打印"≤ N")。

**性能量级(§592/§593 实测,必须按 chunk 大小读)**:GPU 预填充每个 chunk 都要把**该步 40 层的
专家权重 H2D 搬一遍**(TP=2 每 rank **143.6 GiB** = 40 × 3.589)⇒ 单 chunk 成本**近似固定**。
分解(§593,已排除硬件:PCIe 实测 26.8 GB/s = Gen4 x16 线速):
```
纯 pinned 拷贝        5.74 s/chunk     ← 已经是线速
+ GPU 侧 K-major 转置 7.40 s/chunk     ← 多 1.66 s
= 服务当前实测        11.6 s/chunk     ← 另有 4.2 s 未解释(疑似 host 回调串行、未预取)
```
| chunk | 当前 11.6 s | 修掉转置 7.4 s | 线速+重叠 5.74 s |
|---|---|---|---|
| 2048 | 174 | 277 | 357 |
| 8192 | 694 | 1107 | 1427 |
| **17400** | **1500** | 2351 | 3031 |

⇒ **chunk 越大越划算**:`--max-num-batched-tokens` 建议 ≥ 8192;要摸到 1500+ tok/s
当前需 ≥ ~17400,把上面两项修掉后只需 **~8.6K~11K**。
短 chunk 下 GPU 预填充**比 CPU 还慢**(CPU 约 3.9 ms/token ≈ 258 tok/s)——
盈亏平衡在 **chunk ≈ 3000 token**。
⚠️ **`--max-num-batched-tokens=32768` 目前会 hang**(§593(f),启动后 GPU 0%、日志刷
`shm_broadcast...60 seconds`)⇒ 先别用,修好再上。

### 5.8b ⚠️ 发行流程(踩过:只 push tag ≠ 发布 Release)

**"发行一个版本" = 版本号 + git tag + `gh release create` 三件事,缺一不可。**
2026-09-17 的 v0.2 只做了前两件 ⇒ GitHub 的 Releases 页面仍显示旧的 v0.2.0(Latest),
客户端"看不到新版本"。正确流程:

```bash
# 1) 版本号
sed -i 's/^version = .*/version = "0.2"/' pyproject.toml
# 2) 写发行说明 RELEASE_NOTES_v0.2.md,然后 commit + push + 打 tag
git add -A && git commit -m "release: v0.2"
git push origin main
git tag -a v0.2 -m "..." && git push origin v0.2
# 3) **关键一步:创建 GitHub Release**(否则页面上看不到)
gh release create v0.2 --title "vllm-xtu-moe v0.2 — <主题>" \
  --notes-file RELEASE_NOTES_v0.2.md
gh release list --limit 5        # 复核 Latest 是否已切到新版本
```

**版本口径(2026-09-18 定案)** —— "真正支持 V4.1 + 预填充性能优化"的那次作为 `v0.2` 发布;
更早那次"主线化"的 `0.2.0` 全仓库改名 **`0.2pre`**(它的 git tag `v0.2.0` 保留作归档,
但**文档里一律称 `0.2pre`**)。开发过程中曾有一个只含 GPU 预填充修复的中间构建,
**从未正式发布**,其内容已并入 `v0.2`,不作为任何版本号出现。
```bash
git tag -l        # 复核:v0.1.0 / v0.2.0(0.2pre 的历史 tag)/ v0.2 / v0.2.1
```

### 5.9 ⭐ 推荐的生产/挂 harness 配置(最稳定且高效;2026-09-17 定稿)

**一句话:TP=2 / 1M 上下文 / 显式 12 GiB KV / 投机开 / **GPU 预填充关** / 常驻层 0 / 编译关。**

```bash
cd /home/user/lvllm/vllm-xiaotu-moe
VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=0 \
TAG=harness PORT=8700 GPUS=0,1 TP=2 MAXLEN=1048576 SEQS=8 MBT=8192 LOAD=auto GPU_UTIL=0.90 \
SPEC=1 COMPILE=0 THREADS=60 SPIN=300 KV_CACHE_BYTES=12884901888 \
  bash scripts/serve_v41.sh
# READY ≈ 390 s(Engram 189 GiB 流式 + 逐层释放)
# 健康检查
curl -s localhost:8700/v1/models | head -c 120
curl -s localhost:8700/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":8,"temperature":0}'
```

**每个旋钮的依据(都是实测,不是猜)**

| 旋钮 | 值 | 依据 |
|---|---|---|
| `TP=2` | 2 | R-VRAM 默认;**GPU 预填充要求 TP≥2**(TP=1 staging 不摊薄 6.72→13.45 GiB,§592(e)) |
| `MAXLEN` | 1048576 | R-VRAM 优先级 1,**不可降级** |
| `KV_CACHE_BYTES` | **12 GiB** | 复现 §570 验收过的 KV 容量(**6,724,586 token = 1M 并发 6.41×**);同时**刻意留出 ~13 GiB 空闲**做长序列工作区(§513c 的 32K OOM 根因就是这里被 KV 吃光)。**必须显式给**,理由见 §5.8 |
| `SPEC=1` | DSpark | 实测 **16.64 vs 13.85 t/s**(+20%),TPOT 36.75 vs 42.86 ms;greedy 5/5 逐字节一致。<br>⚠️**TPOT 必须连上下文一起读**:同一服务的受控实测(只换 prompt 集)是 **33.5 / 39.8 / 53.5 ms** 对应 17 / 165 / 438 token 的 prompt(README「TPOT 不是常数」)。上表那两个值取自**短 prompt** 口径,不要拿它跟长 prompt 的数字比 |
| **`GPU_PREFILL_MIN_TOKENS=0`** | **关** | §592/§593/§594:MBT=8192 以下 GPU 预填充的**每 chunk 固定成本 ~11.6 s**(80% 是 staging),默认 chunk(~2048)下只有 **155 tok/s vs CPU 258 tok/s**;要赢需 chunk ≥3000 且先修 §594 的 4 条。**先保证稳定** |
| `SEQS=8` | 8 | 挂 harness 要并发;KV 12 GiB 足够(短 prompt 下容量以 Mtoken 计) |
| `MBT=8192` | 8192 | 与 §570 一致;⚠️ **别用 32768**(§593(f) 会 hang) |
| `COMPILE=0` | 关 | 实测 CUDA graph 无收益(43.19 → 44.73 ms;我们的 step 被 CPU↔GPU 边界切成 41 段) |
| `GPU_RESIDENT_LAYERS`(空) | 0 层 | §584 实测**负收益**(TPOT 43.19 vs 42.86;C=4 −12%);这块显存改投 KV 收益确定 |
| `THREADS=60` | 每 rank 60 | 本机 24 CCD 拐点;TP=2 下 184/rank 会 368 线程超订 |
| `SPIN=300` | 300 | SPIN=0 时每层白付 ~0.5 ms futex(0.94-1.00 vs 0.44 ms/层) |
| `ENGRAM_LAST=1` | 默认 | 峰值主机内存 **629.4 GiB vs 1087.6 GiB(−42%)** |
| `RELEASE_SOURCE=1` | 默认 | 逐层 load/slice/release,主机专家内存 522→253 GiB |

**预期性能(TP=2,单流,TPOT ≈ 0.92 ms/层)**
| 场景 | 数值 | 出处 |
|---|---|---|
| 单流解码 | **16.64 tok/s,TPOT 36.75 ms** | ShareGPT 16 题 C=1(§570) |
| C=4 | 30.47 tok/s,TPOT 111.19 ms | 同上 |
| 短 prompt 首 token | ~30 ms + 调度 | §587 |
| **冷预填充(不命中缓存)** | **~258-285 tok/s**(4K prompt ≈ 14 s) | §591 |
| **prefix-cache 命中** | **~600-700 ms**(首步+调度) | §591 ⇒ **多轮/agent 复用同一上下文时几乎免费** |
| 1M 上下文并发 | 6.41× | §570 |
| 主机内存峰值 | 629.4 GiB | §570 |
| 数值/确定性 | `OK=7 BAD=1 max_rel=1.873e-02`;greedy ×2 逐字节 5/5 | §570 |

**⭐ GPU 预填充(§594-§602 修完后,推荐开启)**:`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=4096`
+ **必须**按 §5.8 封顶 KV 池 + `MBT=8192`(见下表的显存账)。

| 项 | 值 | 依据 |
|---|---|---|
| 实测收益(prompt 3655 / 6980) | **12.1 s → 302 tok/s** / **15.2 s → 460 tok/s** | §602,**2.0× / 2.8×** 于同期 CPU 路径(24.2 s / 42.2 s)|
| 成本结构 | `≈8.9 s 固定 + 0.79 ms/token` / chunk | 固定项 = 每 chunk 搬 **143.6 GiB/rank** 专家权重(TP=2,40 层×3.589)|
| staging 真实峰值 | **7.2 GiB/卡**(槽 3.59 + raw 3.16 + 暂存 0.45) | §600/§601;**旧口径 6.72 只算了槽**,会放过注定 OOM 的配置 |
| preflight 门槛 | `staging × 1.10` ≈ **7.9 GiB 空闲** | §600b |
| **MBT 上限** | **8192**(16384 **会 OOM**) | §602(d):16K chunk 的激活+staging 在 40 GB 卡上放不下,OOM 点是 attention 的 `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` |
| 显存配方(TP=2/MBT=8192) | 非KV 10 + KV 4 + staging 7.2 + 首请求增长 7.7 + 激活 ≈ **33 GiB** | §599/§600 |
| ⚠️ 别开 `--enforce-eager` | 它让 free 从 12.95 掉到 **1.4 GiB**,预填充被 preflight 拒(§600c) | 用 CUDA graph 的默认 |

**阈值取 4096,不要用 1024**(§603 实测):成本模型是
`GPU ≈ 8.9 s/chunk 固定 + 0.79 ms/token` vs `CPU ≈ 3.9 ms/token` ⇒ **盈亏平衡 ~2860 token**;
用 1024 会让 1-3K 的 prompt 白付 ~9 s(L=1024 的 TTFT 从 CPU 的 ~4 s 变成 **10.03 s**)。
`vram_policy --emit-env` 现在输出 **4096**(可用 `XIAOTU_GPU_PREFILL_SWITCH_TOKENS` 覆盖)。

**判定"是否真的走了 GPU"**:日志应出现 `GPU prefill ACTIVE`(每个模块一次,40 层×rank 数);
若出现 `GPU prefill DISABLED ... only N GiB is free` ⇒ 按 §5.8 封顶 KV 或降 MBT。

**最终性能验收口径(v0.2 起)**:所有问题收口后用 **官方 `vllm bench serve`**(非自研探针),
`TP=2` + 充分预热,覆盖超短~32K prompt × C=1/2/4/8;详见内部《未来规划》。

### 5.10 ⭐ 两个"给客户端用"的服务端参数(2026-09-18 定稿;**默认都开**)

两条都是**为了 DSH 这类客户端能显示/选择**才加的,已对着本机 vLLM 源码
(`0.29.1rc1.dev95+gdabc4362b`)逐条确认,不是凭记忆。

#### (a) `--enable-prompt-tokens-details` —— 让客户端显示**前缀缓存命中率**
* 源码:`vllm/entrypoints/launchers/cli_args.py:132`
  `enable_prompt_tokens_details: bool = False` — *"If set to True, enable
  prompt_tokens_details in usage."*;`vllm serve --help=all` 里确认为
  `--enable-prompt-tokens-details, --no-enable-prompt-tokens-details`。
* 作用:chat/completions 的 `usage` 里出现
  `"prompt_tokens_details": {"cached_tokens": N}` ⇒ DSH 才能算出**缓冲命中率**。
  不打开时该字段缺失,客户端只能显示 0 或干脆不显示。
* 我们的默认:`TOK_DETAILS=1`(在 `<内部探针>/xtu_own_v41_mem.sh` 里),
  `TOK_DETAILS=0` 可关。

#### (b) 思考强度(**reasoning effort**)
* **不需要服务端 flag 就能被客户端选择**:`reasoning_effort` 是 OpenAI
  chat-completions 的**一等字段**(`chat_completion/protocol.py:245`),vLLM 会把它
  并进 chat template 的 kwargs(`protocol.py:589`)。
* **但服务端要能定一个默认值**,这就是 `--default-chat-template-kwargs`
  (`cli_args.py:93`,JSON;请求级值优先,见 `protocol.py:603` `merge_kwargs`)。
* DeepSeek-V4.1 的取值(`vllm/tokenizers/deepseek_v41.py`):
  `none`(关思考)/ `low` / `high` / `xhigh` / `max`,或 **1..100 的整数**;
  非法值会直接报错。
  ⚠️ **`off` 要真正关思考必须送 `"none"`** —— 什么都不送等于**默认开思考**
  (tokenizer 里 `if "thinking" not in kwargs and "enable_thinking" not in kwargs: thinking = True`)。
* 我们的默认:`THINK_EFFORT=high`(即 `--default-chat-template-kwargs
  '{"reasoning_effort":"high"}'`);留空 `THINK_EFFORT=` 则回到 vLLM 内建默认
  (thinking=True, effort=high)。

**DSH 侧还要配这一小段**(否则选择器根本不出现 —— 能力是**客户端目录里声明的**,
不会向网关查询;见 `packages/llm/llm-pi-ai/src/catalog.ts` 的 `reasoningEfforts` /
`compat.supportsReasoningEffort`)。`~/.dsh/settings.yaml` 里给自建路由的 model 条目加:

```yaml
llm-pi-ai:
  providers:
    epyc-a100-server:
      apiKeyEnv: EPYC_A100_SERVER_API_KEY
      api: openai-completions
      baseURL: http://127.0.0.1:8070/v1
      models:
        - id: DeepSeek-V4-Flash-0731
          contextWindow: 1024000
          maxTokens: 128000
          # ① 声明"可选思考强度" ⇒ 选择器出现;键是 UI 档位,值是**线上拼写**
          reasoningEfforts: { off: "none", low: "low", high: "high", xhigh: "xhigh", max: "max" }
          compat:
            # ② 这个端点接受 `reasoning_effort`(openai-completions)
            supportsReasoningEffort: true
```
* 档位键只能取 `off|minimal|low|medium|high|xhigh|max`(`catalog.ts:74` 的 drift gate);
  值必须是 **vLLM 认得的拼写**,否则 400。
* ⚠️ **V4.1 只认 5 个拼写**:`REASONING_EFFORT_MAPPINGS = {low:25, high:50, xhigh:75, max:100}`
  加 `none`(关思考),另外还接受 **1..100 的整数**
  (`tokenizers/deepseek_v41_encoding.py:183`;`DEFAULT_REASONING_EFFORT = "high"`)。
  **`minimal` / `medium` 会被 V4.1 直接 `ValueError`** —— 它们只在 OpenAI 协议的 `Literal`
  里合法(是给别的模型用的)⇒ **DSH 的 `reasoningEfforts` 里绝不要声明 `minimal`/`medium`**,
  否则客户端一点就是 400。
* 正因为 effort 是 **1..100 的数值预算**,客户端若能送整数,就得到"细粒度思考强度"
  (例如 `reasoning_effort: 60`)。
* `off` 的**值不能留空**:留空 = "支持,但不送参数",对 V4.1 等于**仍然开思考**。
* 只声明 `reasoningEfforts` 而不开 `compat.supportsReasoningEffort`,选择器会出现但
  请求里不带 `reasoning_effort`(`openai-completions` 下该开关才决定发送)。
* 这一路**不需要额外服务端 flag**:本模型的 `architectures=["DeepseekV41ForCausalLM"]`
  ⇒ vLLM 自动把 `tokenizer_mode` 设成 `deepseek_v41`(`config/model.py:709`),
  走 V4.1 prompt encoder。

**自检命令**
```bash
# ① 缓存命中率字段
curl -s localhost:8700/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-xtu","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
# 期望能看到 "prompt_tokens_details": {"cached_tokens": 0} 这样的键

# ② 思考强度(同一句话,low vs none)
for e in low none; do
  curl -s localhost:8700/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"dsv41-xtu\",\"messages\":[{\"role\":\"user\",\"content\":\"9.11和9.9哪个大?\"}],\"max_tokens\":64,\"reasoning_effort\":\"$e\"}" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("reasoning_effort='$e'", "reasoning_content_len=", len(m.get("reasoning_content") or ""), "content=", (m.get("content") or "")[:60])'
done
# 期望:low 有 reasoning_content;none 的 reasoning_content 为空/缺失
```
