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
#    也可以直接装 GitHub Release 附件里的 whl(已内含预编译引擎 .so,无需 2a);
#    本项目(发行名 vllm-xtu-moe)**不发布到 PyPI**。
#    引擎包 xiaotu_moe 另有独立的 PyPI 发行名 `xiaotu-moe`(engine-only,由引擎自身源码
#    发布);本插件的 wheel 已把引擎一起打进去,所以装本插件**不需要**再装它。
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

#### ⚠️ 显存契约:staging 是**进程级持久**的,必须从 KV 里让出来(§601,真实生产事故)

预检口径(`vllm_xiaotu_moe/gpu_prefill.py:fits_device`)是

```
要求空闲显存  >=  staging 峰值 × 1.10  +  XIAOTU_GP_ACT_RESERVE_GIB(默认 3.0 GiB)
```

* **staging 有多大**:FP8 路径(GLM-5.3、TP=2、每 rank、`ns=4`)=
  `2×(w13+w2+s13+s2) + raw + 单份 DMA 暂存` = **7.59 GiB**(启动日志里的权威数字:
  `GPU prefill ACTIVE ... preflight: staging ~7.59 GiB`);这些 `_buf` **按名字缓存、进程内永不释放**。
  注意 `E` 是**每 rank 的 288 个专家**(不是 144):`fp8_staging_bytes(288,4096,1024,ns=4)`;
  按 144 算是 3.7 GiB,**少一倍**。
* **为什么要那 3.0 GiB**:vLLM 是拿 `util×总显存 − (权重 + **激活峰** + CUDA graph)` 去**定 KV 大小**的
  (启动日志里 `peak activation: 2.9 GiB`)—— staging 一进场,吃的正是这份激活峰。只比
  "分配 staging 的瞬间够不够",就是**预检通过、跑起来 OOM**。
* **事故现场(2026-09-19 11:10:56,util 0.90,28,553-token 的真实 DSH 请求)**:
  `chunk_kda_with_fused_gate` 申请 52 MiB 失败,GPU 0/1 各剩 45 MiB(现场还有
  PyTorch 已分配 37.90 GiB + **695 MiB reserved-unallocated**),**两个 worker 一起崩、服务退出**
  (`logs/glm53_prod.log`)。请求本身只有 6528 token 的 prefill chunk,别无异常 ——
  7.59 GiB 持久 staging + 激活峰在 util 0.90 下本来就**装不下**。
* **现在会怎样**:预检不过 ⇒ 打印 `GPU prefill DISABLED ...` 并**优雅退回 CPU 预填充**(慢,但绝不崩);
  通过时 `GPU prefill ACTIVE ...` 会带上四个数:
  `staging ~X GiB, required >= Y GiB free, had Z GiB (slack +W GiB)`
  —— **以后调 util 就看这一行的 slack**,不要凭感觉。
* **生产默认因此从 util 0.90 降到 0.85**(实测 KV 池 988,081 → **971,949** token,只掉 1.6%;
  2 路 256K 占 54%),并加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 治那份 695 MiB 碎片
  (本服务没有 KV connector,不触发 vLLM 对该 env 的兼容性报错)。
  > **2026-09-20 更新**:那次 0.90 装不下的**真因是"没有封顶 KV 池"** —— 池把预算填满,
  > 7.59 GiB 的持久 staging 与激活峰自然挤不下。**现在 `serve_glm53_mainline.sh` 默认
  > `GPU_UTIL=0.90` + 自动按 `MAXLEN×SEQS` 封顶 KV(见 §5.1b)** ⇒ 池不再顶满预算,
  > 0.90 的前提已经变了。**若仍要保守,可用 `GPU_UTIL=0.85`(池与 cap 都由脚本算)。**
  > ⚠️ **该新默认尚未在本机做过 A/B**(cap 固定、扫 util 0.82/0.86/0.90);
  > 未经 A/B 前,生产可先用 0.85 兜底。
* **验收(已在 8070 实跑)**:32,077-token 真实长请求(**TTFT 108.4 s**)、两路并发
  (14,084 + 15,423 token)全部跑通、零 OOM;4096/64 C=1 基准 22,650 ms / 2.50 tok/s
  (旧 util 0.90:22,764 ms / 2.49)—— 换配置**没有性能回退**。
* **【0.2.3 rebase 后,2026-09-20 重标定】GLM 生产 util 必须降到 `0.82`**:rebase 到上游
  `133b71e0b` 后,上游改了激活/KV 定容 ⇒ 同 util 下 KV 池从 971,949 涨到 **1,018,328**,
  激活余量少 ~1.1 GiB ⇒ **util 0.85 下两路 14k+15k 并发 OOM、引擎整进程死亡**(实测两次)。
  `0.82` 下 32k + 并发全部通过(KV 915,487)。权威判据仍是启动日志那一行 `slack`。
* **【0.2.3】开 MTP(`SPEC_K>0`)另有一份显存账单**:启动 profile 的 `peak activation`
  从 2.9 GiB 掉到 **0.84 GiB**(draft 只在 decode 跑),vLLM 据此把 KV 池定得**更大**;
  再叠加第 45 层 draft 常驻 **3.38 GiB/rank** ⇒ **必须 `GPU_UTIL=0.82` 或 `GP_PREFILL=0`**,
  否则 32k 预填充 OOM。完整表格见 `KNOWN_LIMITATIONS.md` §8.2。
* **【0.2.3 已知问题,未修】GPU 预填充偶发 `Triton Error [CUDA]: illegal memory access`**
  (报错点 `byte_transpose._ktranspose_bytes_kernel`,非确定;同配置另一次能跑过)。
  规避:`GP_PREFILL=0`(长 prompt 退 CPU,慢但稳)。定位建议见 `KNOWN_LIMITATIONS.md` §8.3。

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
| `--gpu-memory-utilization` | **默认 `0.90`**(2026-09-20 起);**必须与 `KV_CACHE_BYTES` 同时设** | **原则:留给 staging+激活 = util×VRAM − 非专家 − KV 池** ⇒ **util 尽量高、KV 池按需最小**。⚠️ 只提 util 不封顶 ⇒ 池填满预算、staging 被逐层拒(比不开还慢)、长 prompt 激活 OOM(**0.2.3 的 0.85 回归**)。**顺序:先定 cap,再提 util。** 若启动报 `Free memory ... less than desired`,按实际空闲把 util 降下来 |
| `SPEC_K`(env,`serve_glm53_mainline.sh`) | **默认 `0`(关)** —— 2026-09-20 用户裁定关闭 | `1..4` ⇒ `--speculative-config method=mtp`。**收益/代价严重不成比例,故默认关**:收益只有 decode **+3%**(21.9→22.6 tok/s,噪声内;ShareGPT 上只有 +2%),代价却是 ①draft 第 45 层常驻 **3.38 GiB/rank** ②KV 池 **−27%**(915,487→666,366,256K 并发 3.49×→2.54×)③ITL 脉冲化 46→64 ms。**而且它是长 prompt OOM 的元凶** —— 详见 §3.7。`k>1` 更差(k=4 掉到 11.2 tok/s)。数据见 `MODEL_GUIDES.md` §2.6 |
| `SPEC_CONFIG`(env,`serve_glm53_mainline.sh`) | 默认空 | 透传任意 vLLM 投机配置 JSON(覆盖 `SPEC_K`),用于 **外挂 draft** 方案:`ngram` / `suffix` / `eagle` / `eagle3` / `medusa` / `draft_model`。⚠️ `ngram` 需要 `numba`(本环境未装,启动会 `ModuleNotFoundError: numba`) |
| `GP_PREFILL`(env,`serve_glm53_mainline.sh`) | 默认 `1`(开);**不稳时置 `0`** | `0` ⇒ 把 `XIAOTU_GP_ACT_RESERVE_GIB` 拉到 99 ⇒ GPU 流式预填充永不放行(长 prompt 退 CPU,慢但稳) |
| `--enforce-eager` | 建议先开 | 避免 CUDA graph 与 CPU 引擎 host 回调的额外变量;稳定后可尝试关闭 |
| `--kernel-config.enable_jit_warmup=false` | 建议 | 跳过 JIT 预热,加快启动 |
| `--enable-auto-tool-choice` + `--tool-call-parser` | **GLM-5.3 必开:`glm47`** | 客户端带 `tools` 且 `tool_choice:"auto"` 时,缺这两项 vLLM 直接 **400**;合法值里 GLM 系是 `glm45` / `glm47`(实现是 `glm47_moe_tool_parser`) |
| `--enable-prompt-tokens-details` | **默认已开**(`PROMPT_TOKENS_DETAILS=1`) | 让客户端能读到**前缀缓存命中率**;vLLM 默认**不上报**,不开时客户端只看到 `cached_tokens: null`(见 §3.6) |
| `--reasoning-parser` | **GLM-5.3 必开:`glm47`** | 把思考内容从 `content` 里拆出来,单独返回(**本树字段名是 `reasoning`**,不是旧的 `reasoning_content`) |
| `--trust-remote-code` | 通常不需要 | 主流模型配置已进入主线 |
| `JITCACHE=1`(env,默认) | **保持开启** | 把编译缓存钉到固定目录,见 §3.7 |

### 3.5 工具调用与思考等级(GLM-5.3-Flash,**生产默认已开**)

`scripts/serve_glm53_mainline.sh` 的默认参数里现在包含(可用同名环境变量覆盖,置空即关闭):

```
TOOL_PARSER=glm47        -> --enable-auto-tool-choice --tool-call-parser glm47
REASONING_PARSER=glm47   -> --reasoning-parser glm47
```

**为什么要开**:

* **缺 `--enable-auto-tool-choice` / `--tool-call-parser` 会 400** —— agent 框架(DSH 等)默认就发
  `tools` + `tool_choice:"auto"`,服务端会直接拒:
  `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`。
  实测开启后:`tools=[get_weather]` + `tool_choice:"auto"` → 200 且返回
  `tool_calls=[get_weather({"city": "Tokyo"})]`。
* **缺 `--reasoning-parser` 时思考内容混在 `content` 里**,客户端拿不到独立的思考字段。
  实测开启后同一道题:`reasoning_effort=low` → 思考 0 字 / `high` → 22 字 / `max` → **917 字**。

**思考等级的语义(来自检查点 `chat_template.jinja` 第 2 行,已离线渲染核实)**:

```jinja
{%- set effective_reasoning_effort = reasoning_effort
      if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
```

| 客户端传的 `reasoning_effort` | 模板实际渲染 |
|---|---|
| 不传 / `max` | `Reasoning Effort: Max`(**默认档**) |
| `low` | `Reasoning Effort: Low` |
| `high` | `Reasoning Effort: High` |
| `medium` / `none` / 其它 | **`Max`**(模板不区分 —— 也就是说 **无法借此关闭思考**) |

⇒ 模型真正区分的只有 **low / high / max 三档**。另有 `clear_thinking`(默认 `false`,
控制是否清掉历史里的思考)可用 `chat_template_kwargs` 传入。

**客户端怎么打通**:vLLM 侧由 `ChatCompletionRequest.build_chat_params()` 把**请求顶层的
`reasoning_effort`** 合并进 `chat_template_kwargs` 再渲染模板(模板不认识的变量会被
`resolve_chat_template_kwargs` 过滤掉),所以服务端**只需上面两个 parser 开关**,不需要别的参数。

**客户端侧(以 DSH 为例)**:UI 里的思考等级来自**客户端自己的模型能力声明**,服务端无法替代。
为什么连官方 API 都不用配、我们这条 route 却必须显式声明(2026-09-19 逐行核实):

| 路径 | 能力从哪来 | 要不要写配置 |
|---|---|---|
| 官方 DeepSeek(`llm-deepseek`) | 插件里**写死**:`modelInfoFor()` 直接返回 `reasoning:{efforts:[off,low,high,max]}` | 不用 |
| pi-ai 目录里的 provider(如 `zai`) | pi-ai 自带目录:`providers/data/zai.json` 的 `glm-5.3-flash` 就带 `reasoning:true` + `thinkingLevelMap {off:null,low:low,high:high,max:max}` | 不用 |
| **我们的 `epyc-a100-server`** | 自定义 route:**pi-ai 目录里没有这条 route,也不存在任何"探测端点能力"的逻辑** ⇒ `base?.reasoning ?? false` = **false** | **必须写** |

DSH 里这个字段叫 **`reasoningEfforts`**(`~/.dsh/settings.yaml`,`llm-pi-ai.providers.<route>.models[]`):

```yaml
models:
  - id: GLM-5.3-Flash
    name: GLM-5.3-Flash
    contextWindow: 262144
    # key = UI 里可选档位;value = 真正发给 API 的拼写;只有 off 允许留空。
    reasoningEfforts: { low: low, high: high, max: max }
    # ⚠️ 必须一起钉住这两项,理由见下面两条。
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: true, thinkingFormat: openai }
```

⚠️ **`supportsDeveloperRole: false` 不是可选项**(2026-09-19 实测发现):自定义 route 的 baseURL
检测会给 `supportsDeveloperRole: true`,而 pi-ai 只在 **reasoning 模型**上用它
(`openai-completions.js:910` `useDeveloperRole = model.reasoning && compat.supportsDeveloperRole`)
⇒ **一旦开了思考等级,system 消息的 role 就变成 `developer`**。GLM 的 `chat_template.jinja`
只认 `user/assistant/tool/system`(system 在 255 行),`developer` 分支不存在 ⇒
**整段 system prompt 被静默丢弃**,实测:

```python
tok.apply_chat_template([{"role":"system","content":"SYS-PROMPT-MARKER"},{"role":"user","content":"hi"}], tokenize=False)
# -> '[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>SYS-PROMPT-MARKER<|user|>hi'
# 把 role 换成 developer:
# -> '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi'   ← MARKER 消失
```

* ⚠️ **不要写 `reasoning: true` / `thinkingLevelMap`** —— 那是 pi-ai 的**内部**字段名,不是
  `llm-pi-ai` 的 profile 字段名;写了会被 schema **静默丢弃**(实测:重启后依旧没有档位可选)。
  正确名字是 `reasoningEfforts`,可用
  `node -e "import('<…>/dsh-llm-pi-ai/lib/index.js').then(m=>console.log(m.Config(yaml.load(fs.readFileSync(process.env.HOME+'/.dsh/settings.yaml','utf8'))['llm-pi-ai'])))"`
  就地校验字段有没有被吃掉。
* 未声明的档位会被解析成 `null`(= 不提供),所以上面的字典解析出来与官方目录里 `glm-5.3-flash`
  的那份 **逐档完全一致**:`{off:null,minimal:null,low:low,medium:null,high:high,xhigh:null,max:max}`
  ⇒ UI 只给 **Low / High / Max**(`getSupportedThinkingLevels()` 会丢掉所有 `null` 档)。
* 不要写裸 `off:`(YAML 会解析成布尔 `false`,键变成 `"false"`);要留空必须写带引号的 `"off"`。
* **实际发出去的线格式**:自定义 route 的 baseURL 检测结果是 `thinkingFormat:"openai"` +
  `supportsReasoningEffort:true` ⇒ 只是**请求顶层的 `reasoning_effort: low|high|max`**,
  不选档位时**该字段不出现** ⇒ 模板落到默认档 `Max`(想要别的默认值,可在 route 级加
  `reasoning: high`,它是 profile 的默认档字段)。
* **整条链路可以离线验证**(不需要 GPU / 不需要真服务):起一个假 endpoint 收
  `/v1/chat/completions` 并回一段最小 SSE,再用 pi-ai 自己的 `createModels()`+`createProvider()`
  发一次请求,把 body 打出来。实测(DSH 走的就是这条路径:适配器把等级作为 pi-ai 的
  **`options.reasoning`** 传下去,`dsh-llm-pi-ai/lib/index.js:1664`):

  | `options.reasoning` | 请求体里的 `reasoning_effort` | system 消息 role |
  |---|---|---|
  | `low` / `high` / `max` | `"low"` / `"high"` / `"max"` | `system` |
  | 不传 | **字段不出现**(模板按 `Max` 渲染) | `system` |

  同一个探测在 `supportsDeveloperRole: true` 下打出的是 `developer,user` —— 也就是上面那个坑。
  脚本存在 `dev-docs/dsh_wire_probe.mjs`(不进仓库,随 dev-docs 一起被 gitignore)。
* **生效方式**:DSH 用 chokidar 监听 `settings.yaml`,改完**热重载**;前端刷新一次页面(拉模型目录)
  就会出现档位。若仍没有,说明运行中的 DSH 用它启动时的旧内存文档把文件**回写覆盖**了
  (踩过一次,11:10 的文件改动被抹掉)—— 先确认文件里 `reasoningEfforts` 还在,不在就重写,
  再**完全停掉 DSH 后启动**。

**DSH 的自动压缩 ≠ 万能:阈值必须排在"真上限"之前(2026-09-19 踩坑,DSH 侧配置)**

DSH(`dsh-compaction-basic`)有两层保护,**都会真的触发**,但都要靠配置对齐:

1. **压力压缩**(`agent/pre-step`):DSH 自己的 token 计量 ≥ `contextWindow × thresholdRatio`
   (默认 `0.8`)时,摘要 + 保留尾部(默认占 16%);
2. **超限恢复**(`agent/request-error` 且 code = `CONTEXT_WINDOW_EXCEEDED`):剪枝 + 摘要 + **自动重试**
   (默认 `maxOverflowRetries: 1`)。

⚠️ **但服务商把 `max_tokens` 也算进上下文**(DeepSeek 报错原文:
`maximum context length is 1048576 tokens. However, you requested 1050282 tokens
(794282 in the messages, 256000 in the completion)`),而 `dsh-llm-deepseek` 的默认
`maxTokens` 是 **256000** ⇒ 真正的输入上限 = `1048576 − 256000 = 792,576`,**低于** 0.8 阈值
`800,000` ⇒ 会话可以涨到 794K 仍不触发压缩、却已经超限,**必然失败**。更糟的是超限恢复要再调一次
模型做摘要,而那次输入同样超限 ⇒ 恢复也失败(`compaction/end` 里记
`DeepSeek API stream from https://api.deepseek.com failed`),整轮报错;要等**下一轮**才压得下去。

**修法**(`~/.dsh/settings.yaml`,热重载,无需重启):
`llm-deepseek` 路由级加 `maxTokens: 131072` ⇒ 输入上限 `917,504` > 阈值 `800,000`(余量 117K)。
想保留 256K 输出就换成把 `models[].contextWindow` 降到 ~`800000`(阈值随之 640K),或给
`dsh web` 加 profile patch 把 `compaction-basic` 的 `thresholdRatio` 改成 `0.7`(该插件没走
settings,只能走 `~/.dsh/profiles/web/cordis.patch.yml`)。
本插件自身的 GLM route 是 pi-ai 侧,默认只发 `max_completion_tokens: 8192`,阈值 209,715,
余量 44K,暂时不用动(**注意 `contextWindow` 与 `maxTokens` 相加不能超过服务端 `--max-model-len`**)。

### 3.6 前缀缓存命中率上报(默认已开)与 **2176 块粒度**

`scripts/serve_glm53_mainline.sh` 默认带 `PROMPT_TOKENS_DETAILS=1`,即
`--enable-prompt-tokens-details`(置 `PROMPT_TOKENS_DETAILS=0` 关闭)。

**为什么必须显式开**:vLLM 里 `FrontendArgs.enable_prompt_tokens_details` **默认 `False`**
("If set to True, enable prompt_tokens_details in usage"),且
`_make_prompt_tokens_details()` 在该开关为假时**直接返回 `None`** ⇒ 客户端在 `usage` 里
**看不到任何缓存信息**(表现为 `cached_tokens: null`)。开了之后每个响应都带:

```json
"usage": {"prompt_tokens": 5117,
          "prompt_tokens_details": {"cached_tokens": 4352, "created_cache_tokens": 0}}
```

* 流式请求要带 `"stream_options": {"include_usage": true}`,`usage` 才会出现在最后一个 chunk 里
  (**DSH 走的就是这条路径**,已实测);
* `cached_tokens` 单位是 token,不是请求数;命中率 = `cached_tokens / prompt_tokens`。

⚠️ **命中按 `block_size = 2176` 的整块计算**(该值由 KDA/线性注意力的 mamba page 反推,见
`platforms/interface.py`)。实测(同一段共享前缀、只换末尾提问):

| 共享前缀 token | 第 1 次 | 第 2 次起 |
|---|---|---|
| **5117** | `cached_tokens=0`,`created_cache_tokens=4352` | **`cached_tokens=4352`**(= 2 × 2176) |
| 1715(< 1 块) | `cached_tokens=0` | **`cached_tokens=0`** |

⇒ **共享前缀不足 2176 token 时永远显示 0 命中**,这是块粒度的必然结果,不是缓存坏了;
每次命中至少 2176 token。看 DSH 命中率时按这个粒度解读。

### 3.6b GLM 的投机解码(MTP):**默认关闭**(2026-09-20 用户裁定)

**一句话**:GLM-5.3-Flash 的 MTP **收益只有 +3%,代价却是 27% 的 KV 池 + 3.38 GiB 显存**,
而且**它是长 prompt 显存不足的元凶** ⇒ **默认 `SPEC_K=0`**,不再默认开。

**实测账本**

| 维度 | 数字 |
|---|---|
| 收益:decode(256/128 C=1) | 21.9 → **22.6 tok/s(+3%,在噪声内)**;ShareGPT 上只有 **+2%**(21.9→22.3) |
| 接受率 | **1.46**(p0≈0.46,逐位衰减极快:p1≈0.14、p2≈0.06、p3≈0.03) |
| 代价①draft 常驻 | 第 45 层强制 GPU 常驻 **3.38 GiB/rank** |
| 代价②KV 池 | **915,487 → 666,366 token(−27%)**;256K 并发 **3.49× → 2.54×** |
| 代价③ITL | **脉冲化 46 → 64 ms**(一步吐 1–2 个 token,流式体验变差) |
| `k>1` | **明确更差**:k=4 时 accept 1.63、吞吐掉到 11.2 tok/s |

**为什么这是"长 prompt 的元凶"** —— 显存账本(util 0.82 / 2×A100-40GB):

```
非专家权重 ~7.5 GiB
+ KV 池(封顶 6 GiB)
+ GPU 预填充 staging 8.35 GiB   (7.59 × 1.10)
+ MTP draft 常驻 3.38 GiB      ← 这一项
= ~25.2 GiB   ⇒ 留给「激活工作区」的只剩 ~7 GiB
```
而**长 prefill 的激活正比于 chunk 大小**(MBT),4,148-token 级的请求会超过这 7 GiB
⇒ **实测 `torch.OutOfMemoryError` → EngineCore 死**。诊断见 `dev-docs/HANDOFF_PERF_TOPN.md` §10。

⇒ **GLM 上"GPU 预填充 + 投机 + 长上下文"三者不可兼得**,而投机只值 +3%,**最不值得保**。

**要开的话**(不建议):`SPEC_K=1 bash scripts/serve_glm53_mainline.sh`,并接受上面的代价;
若同时要长 prompt,必须再让出一项(降 KV 池 / 降 `MBT` / 关 GPU 预填充)。

**对照组 —— MiMo 不适用此结论**:MiMo-V2.5 的 draft 是 **dense 3 层**(极小),
GPU 预填充与投机**可以共存**:实测 4,148-token × C=4 拿到 **1,845 tok/s prefill 且投机开着**。
⇒ **"投机与预填充二选一"是 GLM 的特例,不是通用规律。**


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
方式启动,只需把 `--max-model-len`、`--max-num-seqs` 按显存与 KV 需求调整(GLM-5.3-Flash 见 `scripts/serve_glm53_mainline.sh`:默认 **256K × 2 路** + GPU 预填充 + 工具调用/思考解析 `glm47`,端口默认 **8070**,见 §3.5 与 §4.4)。
后端是否被正确选中,用 `scripts/probe_oracle.py` 或启动日志中的
`Using CPU ... MoE backend` 行确认。

---

## 4.4 GLM-5.3-Flash 生产服务(8070)—— 当前占用 8070 的服务

```bash
bash scripts/serve_glm53_mainline.sh          # 默认 PORT=8070,256K × 2 路 + GPU 预填充 + glm47 解析
PORT=8080 bash scripts/serve_glm53_mainline.sh   # 换端口
TOOL_PARSER= REASONING_PARSER= bash scripts/serve_glm53_mainline.sh   # 关掉工具调用/思考解析
```

实测(2026-09-19,2×A100-40GB / TP=2):`tools` + `tool_choice:"auto"` → 200 且正确返回
`tool_calls`;`reasoning_effort` **low/high/max** 三档生效(思考长度 0 / 22 / 917 字符);
KV 池 **971,949** token(两路 256K 占 54%,util 0.85)。**客户端的思考等级需要客户端自己声明**
(DSH 见 §3.5 的 `reasoningEfforts` + `compat.supportsDeveloperRole: false`),服务端只需上面两个 parser 开关。

**事故与恢复(2026-09-19,必读)**:上一版默认 util 0.90 的服务在 **11:10:56 被一个
28,553-token 的真实请求 OOM 打死**(`chunk_kda_with_fused_gate` 申请 52 MiB 失败,
GPU 0/1 各剩 45 MiB)。根因是 GPU 预填充的持久 staging 吃掉了 vLLM 的激活预留,
详见 §3.2 的显存契约。修复后需验收的两件事(已跑过,数字如下):

| 验收项 | 结果 |
|---|---|
| util 0.85 启动 | KV 池 **971,949 token**(3.71×)/ 预检余量 `slack +3.85 GiB` |
| 32,077-token 真实长请求(崩溃复现) | ✅ **TTFT 108.4 s,零 OOM** |
| 两路并发 14,084 + 15,423 token | ✅ 全部完成,零 OOM |
| 4096/64 C=1 基准(对比旧 util 0.90) | ✅ 22,650 ms(vs 22,764),out 2.50 tok/s(vs 2.49) |

### 4.4.1 日志在哪 / 怎么保证存活 / 死了怎么重启

**① 日志**(两种启动方式,位置不同):

| 启动方式 | 日志 | 就绪判定 |
|---|---|---|
| **手动后台**(`nohup`,当前 8070 就是这个) | `logs/$TAG.log`(现为 `logs/glm53_prod.log`)+ 启动器自己的 `nohup` 输出(我起的时候是 `/tmp/glm53_launch.log`) | `curl -sf http://127.0.0.1:8070/v1/models` |
| **systemd 用户单元**(见下) | `journalctl --user -u glm53 -f`(前台模式,stdout 进 journal) | 同上 |

⚠️ 手动模式的日志行是 `> "$LOG"` ⇒ **每次启动都截断旧日志**(11:10 那次崩溃的现场就是这么丢的:我 11:29
重启后旧内容没了)。要留存历史就显式给带时间戳的 TAG:`TAG=glm53_$(date +%m%d_%H%M%S) bash scripts/serve_glm53_mainline.sh`。

**一页纸状态**(端口/进程/显存/关键日志行,只读):

```bash
bash scripts/glm53_status.sh          # 退出码 0 = /v1/models 有响应
```

**② 现在怎么"存活"的(说实话:没有任何守护)**:

* 它是 `nohup ... &` 起的,启动器退出后进程被 init 收养(实测 `PID 520307, PPID=1`)⇒ **能扛住终端关闭、tmux/DSH 退出**;
* 但**不抗崩溃、不抗重启**:OOM 被杀就是死了(§601 那次就是这样);机器重启也不会自己起来;
* 启动器脚本打印 `[glm53] READY` 后自己退出(它只是**启动+等待**,不是 supervisor);
* 所以"存活"目前靠的是:① 显存契约不再让长 prefill OOM(§3.2);② 有人看着。

**③ 死了怎么重启**(~5 min 才 READY,期间 `curl` 会失败是正常的):

```bash
cd /home/user/lvllm/vllm-xiaotu-moe
# 1) 先确认死透了、显存已释放(不然端口冲突/显存不够,起不来)
bash scripts/glm53_status.sh || true
kill "$(cat logs/glm53_prod.pid)" 2>/dev/null; sleep 20; nvidia-smi    # GPU 0/1 应归零
# 2) 起(推荐:带时间戳 TAG,日志留档)
TAG=glm53_prod nohup bash scripts/serve_glm53_mainline.sh >/tmp/glm53_launch.log 2>&1 &
sleep 300; curl -sf http://127.0.0.1:8070/v1/models && echo READY
```

**④ 想让"崩了自动起来 + 开机自起"就用 systemd 用户单元**(不需要 root;模板在
`deploy/systemd/glm53.service`,安装脚本 `scripts/install_glm53_systemd.sh`):

```bash
bash scripts/install_glm53_systemd.sh            # 已在本机安装(未启用、未启动)
sudo loginctl enable-linger "$USER"              # 开机自起才需要(需 sudo)
# 切换:先停手动实例 → 再交给 systemd
kill "$(cat logs/glm53_prod.pid)" && sleep 20 && nvidia-smi
systemctl --user start glm53
systemctl --user status glm53 ; journalctl --user -u glm53 -f
```

单元里的策略:`Restart=always` + `RestartSec=30`(OOM/异常退出 30 s 后自动拉起)、
`StartLimitIntervalSec=600 / StartLimitBurst=3`(10 分钟内最多 3 次,避免启动即失败的死循环)、
`TimeoutStartSec=1800`(加载 ~5 min,别被默认 90 s 判死)、`KillSignal=SIGINT`
(走 vLLM 优雅关停,释放显存)。⚠️ systemd 档与手动档**共用 8070 和 GPU 0/1,不能同时开**。

```bash
# 起服务脚本还有两个给排错用的开关(不碰 GPU):
DRYRUN=1 bash scripts/serve_glm53_mainline.sh        # 只打印最终命令行
FOREGROUND=1 PORT=8099 bash scripts/serve_glm53_mainline.sh   # 前台跑(日志到终端,不写 pidfile)
```

### 4.4.2 性能统计怎么取(**不用重启服务**)

| 来源 | 命令 | 能给什么 |
|---|---|---|
| **Prometheus `/metrics`** | `curl -s http://127.0.0.1:8070/metrics` | 90 个系列:TTFT / ITL 直方图、prefix cache 命中、KV 使用率、preemptions、请求结束原因、cache_config(KV 池大小/block=2176/util) |
| 日志里的 10 s 统计行 | `grep "Engine 000:" logs/glm53_prod.log` | prompt/generation 吞吐、Running/Waiting、GPU KV 使用率、**前缀缓存命中率**(逐窗口) |
| 单请求 | 响应里的 `usage.prompt_tokens_details` | `cached_tokens` / `created_cache_tokens`(要 `--enable-prompt-tokens-details`,已开) |
| 逐层预填充耗时 | `XIAOTU_GP_SPLIT=1`(**需重启**) | 每层 `asm/kernel` 毫秒数(`[gp-split]` 行) |

**按"一次运行"取增量**(累计量做差即可,不必重启):

```bash
curl -s localhost:8070/metrics > /tmp/m0          # 跑之前
# ... 客户端跑一轮 ...
curl -s localhost:8070/metrics > /tmp/m1          # 跑之后
python3 - <<'PY'
import re
def load(p):
    out={}
    for line in open(p):
        if line.startswith('#'): continue
        parts=line.split()
        if len(parts)<2: continue
        key=parts[0].split('{')[0]
        try: out[key]=out.get(key,0.0)+float(parts[1])
        except ValueError: pass
    return out
a,b=load('/tmp/m0'),load('/tmp/m1')
for k in ['vllm:time_to_first_token_seconds_sum','vllm:time_to_first_token_seconds_count',
          'vllm:inter_token_latency_seconds_sum','vllm:inter_token_latency_seconds_count',
          'vllm:prompt_tokens_total','vllm:generation_tokens_total',
          'vllm:prefix_cache_queries_total','vllm:prefix_cache_hits_total']:
    print(f"{k:52s} +{b.get(k,0)-a.get(k,0):,.1f}")
PY
```

> 实测参考(2026-09-19 11:29–14:34,15 次请求累计):平均 **TTFT 39.45 s**、平均 **ITL 83 ms
> (≈12 tok/s)**、平均端到端 48.6 s、**前缀缓存命中 26.5%**(50,048/188,653 token)、
> **preemptions=0 / error=0**;请求结束原因 5 `stop` + 10 `length`(后者是我用
> `max_tokens=8/16/64` 的探针与 `--ignore-eos` 基准确认,不是异常)。

**`/metrics` 里的 KV 池自证**:`vllm:cache_config_info{...}` 带
`block_size="2176"`、`kv_cache_size_tokens="971949"`、`kv_cache_max_concurrency="3.708"`、
`gpu_memory_utilization="0.85"` —— 和我写在 §4.4 的验收数字一致。


> ⚠️ **8070 现在归 GLM-5.3-Flash**。DeepSeek 的生产脚本 `scripts/serve_prod_8070.sh` 默认也用
> 8070,**两者不能同时起**(显存也只够一个 TP=2 服务)。要同时跑请显式换端口,例如
> `PORT=8090 bash scripts/serve_prod_8070.sh`(见 §4b)。

## 4b. DeepSeek 生产服务(两种模式,按"能不能交互"选)

> ⚠️ **端口**:该脚本默认 `PORT=8070`,而 **8070 现在归 GLM-5.3-Flash**(见 §4.4)⇒ 两者不能同时起,请显式换端口,例如 `PORT=8090 bash scripts/serve_prod_8070.sh`。

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

### 5.1b ⭐ 显存预算的正确设法学(2026-09-20 定案)

> **2026-09-21 补充**:如何在**固定上下文长度**(硬约束)下,把剩余显存分配给
> 「投机解码 / long prefill / 常驻 MoE」三者,见 **[`docs/TUNING_GUIDE.md`](TUNING_GUIDE.md)**。
> 该文给出预算恒等式、三个选项的**实测单位代价/收益**(例如常驻 MoE:
> **1.59 GiB/rank/层**换 **0.70 ms/token @C=1**、1.70 @C=4,TP=2 上限 11 层)、
> 决策算法,以及一个算好的 2×A100-40GB 例子(结果与实测的 11 层上限吻合)。
> **结论:每硬件 × 每模型都有自己的最优解,不是一组万能参数。**
> 常驻层曲线的本机复核用 `scripts/bench_resident_sweep.sh`。

```
留给 staging + 激活工作区 = util × VRAM − 非专家权重 − KV 池
```
* **`util` 是总预算上限,`KV_CACHE_BYTES` 是只给 KV 的上限**;
* **只提 util 不封 KV** ⇒ 池会**填满预算**(vLLM 按"填满 util"定池,与 maxlen 无关)
  ⇒ staging 被逐层拒(静默退回 CPU 预填充,**比不开还慢**)、长 prompt 激活 OOM;
* **只封 KV 不提 util** ⇒ 池合适了,但总预算小,留给激活的也少;
* ⇒ **两者同时设**:`util` 拉高做预算,`KV_CACHE_BYTES` 按需封池,**差额全给 staging + 激活**。

**KV 标尺(本机 GLM 实测,分毫不差)**:`6 GiB ↔ 293,651 token` ⇒ **48,942 token/GiB**。

| 目标 | 需要 token | 折合 cap |
|---|---|---|
| 256K × 2 | 524,288 | **≥ 11.8 GiB** |
| 256K × 3 | 786,432 | ≥ 17.6 GiB |
| 32K × 2 | 65,536 | ≥ 1.5 GiB |

**`serve_glm53_mainline.sh` 已内建**:不给 `KV_CACHE_BYTES` 时**按 `MAXLEN × SEQS × 1.10` 自动推算**
(262144×2 ⇒ 自动 11.8 GiB),`GPU_UTIL` 默认 **0.90**。

⚠️ **前提**:`util` 不得超过**启动时的实际空闲比例**,否则报
`Free memory on device cuda:N (...) is less than desired GPU memory utilization`。

### 5.2 R-VRAM 策略(权威输出)

> **2026-09-20 修订(用户裁定)**:优先级里**去掉"专家层常驻"** —— 实测收益为负(每层 3.36 GiB,
> 换来的收益抵不过它对 32K 预填充的挤压,见 §584),**不再作为一档**;
> 并**补上"激活工作区"** —— 它正比于 chunk(=MBT),是长 prompt OOM 的真凶,之前一直没被列出。
> (`vram_policy` 的输出里仍会打印一行"专家层常驻 0 层",那是**已退役项**的残留兜底,不再参与优先级。)

```bash
python -m vllm_xiaotu_moe.vram_policy --maxlen 1048576 --max-num-seqs 2 --tp 2 --gpu-mem-util 0.55
```
```
[vram-policy] maxlen=1048576 × 2 并发  util=0.55  ⇒ 填池后净空 17.8 GiB(预填充 preflight 要 8.4)
✅ 1. KV 池               4.4 GiB/卡(maxlen=1048576 × 2 并发)
✅ 2. GPU 预填充          8.4 GiB(staging 6.72 × 1.25)
✅ 3. GPU 投机解码        3.7 GiB        ← GLM 上已默认关(§3.6b)
  4. 激活工作区          ∝ chunk(=MBT)：**不预检,但必须留够** —— GLM 长 prompt OOM 就是栽在这里
  ❌ 已退役:专家层常驻   (收益为负,不再参与优先级)
```
任一项不足即按优先级 fallback;**绝不额外多占系统内存**。

⚠️ **激活工作区(第 4 项)是唯一"不预检却会致命"的项**:`vram_policy` 只算前 3 项,
而长 prefill 的激活正比于 chunk 大小。GLM 在 util 0.82 下前 3 项用完 ~19 GiB,
留给激活的 ~7 GiB 会被 4K 级 prompt 的 chunk 打穿 ⇒ **`torch.OutOfMemoryError`**。
**排查长 prompt OOM 时,先降 `MBT`,再考虑关投机/关 GPU 预填充。**

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
> **⚠️ 2026-09-20 更新:上面的分解已被分层插桩取代,结论变了。**
> 用 `XIAOTU_GPF_STAGE=1`(`gpu_prefill.py:954`)在**服务里**实测 DSV4.1(qlen≈4,274):
> ```
> [gpf-stage] n=80  dma=599.2ms  asm=0.0ms  tr=6.7ms  total=606.0ms/层
> ```
> | 子相 | 每层 | 占比 |
> |---|---|---|
> | **DMA(权重 H2D)** | **~600 ms** | **98.9%** |
> | asm(跨步组装) | 0.0 ms | 0% |
> | K-major 转置 | ~6.7 ms | **1.1%** |
>
> ⇒ ① **转置不是问题**(只占 1.1%),"离线预转 K-major"的收益上限就是这 1%;
> ② **真正的靶子是 DMA**:每层 3.86 GB/rank ÷ 0.600 s = **~6.4 GB/s**,而本机线速
> **26.86 GB/s** ⇒ **只跑到 24%,有 ~4.2× 空间**(地板 144 ms/层 ⇒ 6.0 s/chunk)。
> 待查:插桩自身的 `cuda.synchronize()` 会放大串行(需用 CUDA event 复核)、host 回调是否
> 真与 attention 重叠、环形槽只有 2 个是否退化成串行。诊断细节见
> `dev-docs/HANDOFF_PERF_TOPN.md` §12。
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
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":8,"temperature":0}'
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
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
# 期望能看到 "prompt_tokens_details": {"cached_tokens": 0} 这样的键

# ② 思考强度(同一句话,low vs none)
for e in low none; do
  curl -s localhost:8700/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"DeepSeek-V4.1-Flash\",\"messages\":[{\"role\":\"user\",\"content\":\"9.11和9.9哪个大?\"}],\"max_tokens\":64,\"reasoning_effort\":\"$e\"}" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("reasoning_effort='$e'", "reasoning_content_len=", len(m.get("reasoning_content") or ""), "content=", (m.get("content") or "")[:60])'
done
# 期望:low 有 reasoning_content;none 的 reasoning_content 为空/缺失
```

---

## 6. 跟进上游 vLLM(2026-09-19 预演数据)

**先纠正一个常见误解:我们这个栈已经在 Model Runner V2 上了。** 服务启动日志里就有
`[gpu_worker.py:441] Using V2 Model Runner`;V2 的实现是 **`vllm/v1/worker/gpu/` 这个包**
(`model_runner.py`),V1 是旧的单体 `vllm/v1/worker/gpu_model_runner.py`,只在上游判定
"V2 不支持该组合"时才回退(`config/vllm.py:_get_v2_model_runner_unsupported_features`:
ngram/ngram_gpu 投机、EAGLE 的 parallel drafting、stock torch.compile、UBatching、
mamba_cache_mode=all、自定义 logits processor …)。插件打的正是 V2 的命名空间
(`vllm.v1.worker.gpu.model_runner`、`vllm.v1.worker.gpu.spec_decode.speculator`)。
上游 MRv2 仍在按模型族扩默认(见 [MRv2 博客](https://vllm.ai/blog/2026-03-24-mrv2)、
[pooling 默认 MRv2 #48290](https://github.com/vllm-project/vllm/pull/48290)、
[Llama/Mistral #43458](https://github.com/vllm-project/vllm/pull/43458)),
**所以"跟进 V2"这件事我们已经跟上了,问题是继续往前跟。**

**我们与上游的距离(实测)**

| 项 | 值 |
|---|---|
| 我们补丁栈的上游 base | `dabc4362b4`(2026-09-14,= `origin/main~311`) |
| 本地补丁 | **10 个提交 / 39 个文件**(9 个特性补丁 + 1 个 30 文件的 snapshot),全是模型/注意力/kernel 层,不碰 runner 架构 |
| 上游今天 | `41c4a3ed4e`(2026-09-19),**311 个提交 / 5 天**(≈62 提交/天) |

**rebase 预演(在 `/tmp` 的临时 worktree 里 `cherry-pick`,不碰工作树、不碰 8070)**

* 10 个提交整体 cherry-pick:首个(snapshot)提交冲突 **2 个文件**;
* 只挑 9 个特性补丁:第 1 个(mHC)**冲突 1 个文件**——上游只对它做过 `ruff/pydocstyle`
  格式化(#52136)⇒ **纯格式冲突**;继续往下,DeepSeek-V4.1 的 fp8e4nv 移除补丁冲突
  **2 个文件**(`deepseek_v41/common/ops/{cache_utils,fused_compress_quant_cache}.py`,
  上游对这两个文件有 6/3 次改动);
* 其余大部分**自动合并**,包括 `envs.py`、`v1/engine/core.py`、`sparse_swa.py`、
  `oracle/{unquantized,int_wna16}.py`;
* **唯一需要"重指"的**:补丁 `a93931f11a` 改的 `vllm/models/glm5next/nvidia/attention.py`
  在上游已**被搬走**(现在注意力层在 `vllm/model_executor/layers/attention/sparse_mla_attention.py`
  这一带)⇒ 1 个文件重新接线;我们新增的 `flashmla_sparse_sm8x.py` / `sparse_mla_kernels.py`
  是新文件,不冲突。
* **结论:约 4-5 个文件需要手工解冲突 / 重指,不需要重构。**

**插件挂点健在性(逐条核对 upstream HEAD)**

| 挂点 | 现状 |
|---|---|
| `FusedMoEFactory` | 还在(`fused_moe/layer.py:88`,是**函数**不是类——`git grep "class FusedMoEFactory"` 会误判为"没了") |
| `oracle/{fp8,mxfp4,int_wna16,unquantized}.py` | 都在(新增了 `mxfp8/nvfp4/int8/w4a8`) |
| `experts/cpu_moe.py` | 在 |
| `profile_run` / `compile_or_warm_up_model` / `init_attn_backend` / `_initialize_kv_caches` | 都在(命名空间未变) |

**上游这 5 天在我们耦合面上的改动量(= 长期维护成本排序)**

```
vllm/model_executor/layers/fused_moe      26 次   ← 最勤:量化/MoE API 是我们的主要 shim 面
vllm/v1/attention                         23 次   ← 我们的 SM8x 补丁都在这里
vllm/v1/core                              16 次   ← KV 定容/调度:GPU 预填充放行时序依赖它
vllm/model_executor/layers/quantization   10 次
vllm/v1/worker/gpu/model_runner.py         9 次   ← V2 runner 本体(profile_run 挂点)
vllm/models/glm5next                       8 次
vllm/v1/worker/gpu_worker.py + gpu_model_runner.py  6+6 次
vllm/v1/worker/gpu/spec_decode             5 次
```

**性能优化要不要重做?——设计不用,标定和验收必须重跑**

* **可以原样复用**(与 runner 语义无关):CPU 引擎内核(引擎侧)、FP8 流式装配/DMA/side-stream、
  GPU 常驻层、ping/pong 预取、以及服务级的 parser / 前缀缓存开关;
* **必须重标定**(都是经验数):`--gpu-memory-utilization 0.85`、`XIAOTU_GP_ACT_RESERVE_GIB=3.0`
  (基准是启动日志里上游给的 `peak activation`)、`XIAOTU_GP_ACT_RESERVE_GIB`/GPU 预填充阈值
  1500、KV 池与并发;上游这 5 天就有 `FULL CUDA graph capture for microbatched steps (DBO)`
  (#51700)、MRv2 fast-prefill(#56145)、共享 token→request 映射(#57102)等**会改变激活/时序**的改动;
* **必须重跑的门禁**:`test_block23_equiv.py`、determinism 11/11、fp8 conformance、
  `check_engine_aligned.sh`、`test_gpu_prefill_fp8_assembly/vs_cpu`、以及服务验收(32K 崩溃复现、
  4096/64 TTFT、256/128 TPOT);
* **工作量**:rebase + 门禁 ≈ 半天;重标定 + 验收 ≈ 半天到一天(每次服务启动 ~5 min,要起 2-3 次)。

**跟进流程(推荐)**

```bash
cd /home/user/lvllm/process_data/ref/repos/vllm-mainline
git fetch origin main                              # 已完成:origin/main = 41c4a3ed4e
git worktree add --detach /tmp/vllm-up origin/main # 隔离预演,不动当前工作树(editable 安装正在被 8070 用)
cd /tmp/vllm-up && git cherry-pick <我们的 10 个提交>   # 解那 4-5 个冲突
# 通过后再跑门禁;起服务用**临时端口**(8071/8072)做新老对照;最后才决定是否替换 8070
```

⚠️ 两个坑:① `vllm.__version__` 报的是**构建时**的 base(`0.29.1rc1.dev95+gdabc4362b`),
不是当前 HEAD(`git describe` = `v0.29.1rc0-105-gaf3e7c14d7`)——判断代码版本看 `git describe`;
② 直接在当前目录 rebase 会改到正在被 8070 加载的代码(editable install),**必须用 worktree/分支隔离**。
