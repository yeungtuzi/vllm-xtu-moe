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
cd vllm && git checkout dabc4362b       # 已验证提交(需有 precompiled wheel;见上方说明)
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
[`GPU_PREFILL.md`](GPU_PREFILL.md)。

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
| `--max-model-len` | 先 `4096`–`8192`,再按 KV 容量放大 | KV 占用随模型不同(DeepSeek-V4 约 400 KiB/token) |
| `--gpu-memory-utilization` | `0.85` | 非专家权重 + KV cache 在显存 |
| `--enforce-eager` | 建议先开 | 避免 CUDA graph 与 CPU 引擎 host 回调的额外变量;稳定后可尝试关闭 |
| `--kernel-config.enable_jit_warmup=false` | 建议 | 跳过 JIT 预热,加快启动 |
| `--trust-remote-code` | 通常不需要 | 主流模型配置已进入主线 |
| `JITCACHE=1`(env,默认) | **保持开启** | 把编译缓存钉到固定目录,见 §3.5 |

### 3.5 JIT 固定缓存目录(`JITCACHE`,用户 2026-09-17 要求)

**症状**:用 `--compilation-config '{"mode":"VLLM_COMPILE",…}'` 时,**每换一个旗标、每改一行我们
的代码,启动后第一个请求就把逐形状的 Triton 内核从头编一遍**(`jit_monitor` 警告 20-60 s/形状)。

**原因**(安装好的 vLLM 源码里逐行核实):缓存目录名 = 四个 hash 拼出来的
`$VLLM_CACHE_ROOT/torch_compile_cache/<hash10>`,而其中 `env_hash` 覆盖**每一个 `VLLM_*` 环境变量**、
`code_hash` 覆盖**被 trace 的源码(含我们插件替换的模型类)**;并且
`CompilerInterface.initialize_cache()` 会把 `TRITON_CACHE_DIR` **重定向**到那个 hash 目录里,
所以 `~/.triton/cache` 攒下的内核用不上。

**做法**:三个启动器(`scripts/serve_v41.sh`、`scripts/serve_mainline.sh`、
`report/tuning/probes/xtu_own_v41_mem.sh`)都 source `scripts/lib_jitcache.sh`:

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

> ⏸️ **暂不声明支持**:该模型的实测数据早于 v0.2 的改动(执行模型 / 小 batch 路径 / EP 存储分片),**未在当前代码上复验**。复验计划见 `docs/HANDOFF_v0.2.md` §5.1。

### 4.3 其它 MoE 模型

任何使用 `FusedMoEFactory` 的 MoE 模型(BF16 / FP8 / MXFP4 / INT4)都可以用同样
方式启动,只需把 `--max-model-len`、`--max-num-seqs` 按显存与 KV 需求调整。
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
`period 4.63 ms = compute 0.86 + rest 3.77`);解法见 `docs/PERFORMANCE_OPTIMIZATION.md §7`
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

## 5. DeepSeek-V4.1-Flash 运行手册(v2026-09-18;数据出处见 `report/tuning/NOTES.md` §564-§578)

### 5.1 出货默认(不要凭记忆改,以下为实测过的组合)

| 项 | 默认 | 依据 |
|---|---|---|
| TP | **2** | 单卡放不下;TP=1 需显式说明差异(§507) |
| 分片单位 | **自适应 node**(NPS4 上 = 8 片) | socket 分片在 NPS4 上慢 +29%(§505/§518/§519) |
| `XIAOTU_MOE_SPIN_IDLE_US` | **300** | 5000 = 正反馈灾难;0 更慢(§356/R14;本机 1.84 vs 1.14 ms/层) |
| `XIAOTU_ENGRAM_LAST` | **1**(2026-09-17 起) | 峰值 **1087.6 → 642 GiB(−41%)**;正确性已闭环(§563-565) |
| `XIAOTU_ENGRAM_VERIFY` | **1** | 表内容逐 chunk 抽验、失败 fail-closed(§564) |
| GPU 预填充 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` | 由 `vram_policy` 给(**1M 时 1024**) | 实测交点 384;GPU 路径非逐位确定(§572),但更快 |
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

**输入口径(§592 全部改成"用户不用填"或"实测值")**
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
preflight 要 16.8),单卡必然放不下 ⇒ 策略直接判否(用户裁决:不指望 TP=1 跑它)。

### 5.3 启动与验收命令

```bash
# 启动(TP=2/1M/投机/常驻层,真实权重)
TAG=acc1m PORT=8315 MAXLEN=1048576 SEQS=8 GPUS=0,1 SPEC=1 RESIDENT=20-21 GP_MIN=1024 THREADS=60 \
  bash report/tuning/probes/xtu_own_v41_mem.sh        # 自带内存峰值采样

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

### 5.5 验收/诊断工具(都在 `report/tuning/probes/`)

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
TAG=gpf KV_CACHE_BYTES=4729960528 GP_MIN=1024 MAXLEN=1048576 SEQS=2   bash report/tuning/probes/xtu_own_v41_mem.sh
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

**性能量级(§592 实测,必须按 chunk 大小读)**:GPU 预填充每个 chunk 都要把**该步 40 层的专家权重
H2D 搬一遍**(TP=2 每 rank ≈144 GiB)⇒ 单 chunk 成本**近似固定**:
```
吞吐(chunk) ≈ chunk_tokens / 11.8 s        # 2048 → 174 tok/s;8192 → 694;32768 → 2777
```
⇒ **chunk 越大越划算**:`--max-num-batched-tokens` 建议 ≥ 8192,要摸到 1500+ tok/s 需 ≥ 16384~32768。
短 chunk 下 GPU 预填充**比 CPU 还慢**(CPU 约 3.9 ms/token ≈ 258 tok/s),这正是 §592 的实测结论。
