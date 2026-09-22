# Changelog

本文件记录 `vllm-xtu-moe` 的显著变更。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

---

## [Unreleased]

### Added

- **DeepSeek-V4.1-Flash 的 CED(decoder-side SWA bounded replay)验收数据**:承接上游
  [PR #56752](https://github.com/vllm-project/vllm/pull/56752)(`ced/pr56752` = 生产树 +3 commits)。
  同批**配对 A/B**(256K / L=16384 / MBT=8192 / TP=2 / 两臂同 seed,仅 `--no-swa-bounded-replay` 不同):
  * **prefill:C=1 434.5 → 903.9 tok/s(2.08×)、C=2 334.6 → 602.1 tok/s(1.80×)**;
  * **decode:C=1 持平(median TPOT 51.91 → 51.84 ms)、C=2 快 1.99×(257.3 → 129.4 ms)**;
  * **零失败(0/16,含两路并发)**;生成等价性:3 条自然长 prompt **首 token 全一致**。
  README 的 V4.1「长」两行已改用 CED 开的数据。详见 `docs/EXPERIMENTS.md` B84/B85。
- **③「~13.95 s 未归属时间」结案**:它不是"插件看不到的前向",而是**稀疏 MLA 注意力被记进了 MoE 的 `pre`**。
  逐层计时 `pre`(apply 内)= **98.9%**、`other`(apply 外,含注意力/层间)= **1.1%**;慢层**精确等于 DSA 层**
  (`layer_types` 周期 4)。chrome trace 给出**无残差**的闭合账:稀疏 MLA **24.80 s(59.8%)**、
  MoE GEMM 5.80 s、NCCL 2.08 s、其余核 2.11 s、**非核空隙(H2D DMA)6.71 s** = 41.49 s。
  详见 `docs/EXPERIMENTS.md` B87/B88。
- **pr4(`pr4-sm8x-sparse-mla-2d-tile.patch`)端到端复现**:独立树(生产树 `git worktree` + 编译产物,
  生产树 md5 未变)实测 **TTFT 47.44 → 31.18 s(1.52×)**、
  `_sparse_mla_fwd_with_sink_kernel(_hb)` **24.80 → 8.60 s(2.89×)**,其余各项(MoE GEMM / NCCL / H2D)**一个都没动**。
  与 pr4 自测的 1.52× / 2.88× 吻合。详见 B89。
- `scripts/serve_v41.sh` 的 `CED=0/1` 开关(默认 1 = 树默认):`CED=0` 加 `--no-swa-bounded-replay`,
  因为上游复用 `CacheConfig.swa_bounded_replay` 且**没有独立 CED 旗标**、启动后不可改 ⇒ A/B 必须两臂各起一次。
- 插件 `[layer-timing]` 行新增**墙钟时间戳与层名**(`XIAOTU_LAYER_TIMING=1` 时才有该行,只改打印):
  配合 `XIAOTU_LAYER_TIMING_EVERY=1` 可**逐层**反解 `pre/eng/post` 并得到层间墙钟。
  解析器 `dev-docs/report/tuning/probes/attrib_layer_timing.py`、`attrib_trace.py`。
- `gp_capturing()`(`vllm_xiaotu_moe/mixed_experts.py`):CUDA graph 捕获期判定;
  捕获期**强制不用旁路流**,落到「当前流 + 不建跨流事件」的保守路径。
- `XIAOTU_DEATH_DEBUG=1`(`vllm_xiaotu_moe/__init__.py`,默认关闭):给
  SIGTERM/SIGINT/SIGHUP/SIGUSR1/SIGUSR2 装全线程栈 dump + 心跳,用于排查
  「进程被静默杀掉」类问题。
- `scripts/bench_random_12cells.sh`:random 数据集 12 格基准(3 模型 × {128,16384} × {C=1,C=2}),
  含 KV cap 自动推算与**前置断言**、日志/pidfile 双判据的早退检测、显式 `PYTHONPATH`、
  清场与观测分离。
- `dev-docs/RND_CAMPAIGN_DIAGNOSIS.md`:本次事故的完整诊断记录(含所有走过的弯路)。

### Changed

- **生产口径:回到 `MBT=4096`,保证 DeepSeek **1M** 与 GLM-5.3 **512K**(都已实测启动通过)**:
  * `serve_glm53_mainline.sh` 默认 `MAXLEN 262144 → 524288`、`MBT 8192 → 4096`
    (实测 `/v1/models=524288`、KV 池 **952,107 token**);
  * `serve_v41.sh` 默认 `MBT → 4096`(1M 的生产调用写在头部注释里;实测 `/v1/models=1048576`);
  * `serve_prod_8070.sh` 两个 MODE 的 `MAX_NBT → 4096`。
  * ⚠️ 同时修掉一个真实的坑:GLM 的 KV 自动封顶原为 `MAXLEN × SEQS × 1.10`,512K + `SEQS=2` 会算出
    **23.56 GiB** 并被引擎**一次性**申请 ⇒ CUDA OOM(只差 0.6 GiB)。**池不需要装下 SEQS 条满长序列**,
    改为 `MAXLEN × B/token`。详见 B90。
- README(中/英)性能节**精简**:表格后只保留「两个模型都未开投机解码」一句;
  口径/判据/`MBT` 取舍/CED A/B 等细节移入 `docs/EXPERIMENTS.md` 与 `docs/TUNING_GUIDE.md`。
- README(中/英)口径补充:**decode 取 median TPOT**。CED 开臂的 `mean` TPOT 被少数(p99≈185 ms)
  离群解码步拉高(69.76 vs median 51.84 ms),照抄 mean 会误判「CED 让 decode 慢 26%」。
- **删除 MiMo-V2.5 的「256K 不可行」说明**(README 中/英 + `docs/PREFILL_KNOWN_ISSUES.md` §5b)。
  原结论是 **TP=1** 的核算却写成 "61.9 GiB/rank",且**从未实测**;现在检查点已删、无法补测,
  用户也已转 MiMo-2.6 ⇒ 不再对 MiMo 的上下文上限做任何断言。
- `gp_side_stream_enabled()` 默认值由「开」改为**「关」**
  (`XIAOTU_GP_ASM_SIDE_STREAM=1` 仍可显式打开,仅供性能实验)。
  **代价:长 prompt TTFT +17%**(实测 89.8 s → 105.4–108.8 s),待用正确的流间同步换回。

### Fixed

- **GLM-5.3-Flash 长 prompt 首次请求触发 CUDA `illegal memory access`,导致引擎崩溃**
  (`EngineDeadError`,bench 侧表现为 `TTFT 0.00` + `ConnectionRefusedError`)。
  根因是 **FP8 GPU 预填充装配运行在未正确同步的旁路流上**。
  修复:FP8 装配的旁路流**默认关闭**,并新增捕获期保护。
  详见下方「事故复盘」。

---

## 事故复盘:GLM-5.3-Flash 长 prompt 非法访存

### 症状

`random` 数据集 12 格 campaign 中,GLM 的 `L=16384` 两格 `completed=0/8`、`TTFT 0.00`;
服务端日志里 `illegal memory access` 出现 **12 次**,`EngineDeadError`。

### 触发条件(实测收敛)

必须先跑过 **`L=128 C=1` 与 `L=128 C=2` 两种 decode batch**(对应两张 CUDA graph),
之后的**第一次长 prompt** 才越界。单变量矩阵:

| 序列 | 结果 |
|---|---|
| `128:1:8` + 长 | ✅ 干净 |
| `128:2:8` + 长 | ✅ 干净 |
| `128:1:16` + 长 | ✅ 干净 |
| 长 + 长 | ✅ 干净 |
| **`128:1:8` + `128:2:8` + 长** | ❌ **崩**(`illegal=12`) |
| 同上 + `CUDA_LAUNCH_BLOCKING=1` | ✅ **干净** |
| 同上,长 prompt 降到 4096 | ❌ 崩 |

「`CUDA_LAUNCH_BLOCKING=1` 让故障完全消失」= **竞态**的典型指纹,不是确定性越界。

### 根因

报错表面位置是 `byte_transpose.py` 的 Triton 转置 kernel,但那是**粘性错误的浮出点**
(`load_binary` 是模块加载,根本不碰显存)。逐层排除后锁定:

**`mixed_experts.py` 的 `apply()` 把整个 FP8 装配(`gpu_prefill_fp8.py` 的
`kmajor_from_engine_shards_fp8`,内含 H2D `cudaMemcpy2DAsync` + 转置 + GEMM)
跑在 `with torch.cuda.stream(_side)` 的旁路流上,而该路径对 CUDA graph 捕获期
没有任何保护** —— 而 MXFP4 路径(`gpu_prefill.py`)有 **3 处**
(`wait_no_capture` / `wait_capture_done` / `is_current_stream_capturing`)。
这是两条路径之间唯一的结构性不对称。

**单变量证据**:仅把 `XIAOTU_GP_ASM_SIDE_STREAM` 置 `0`(其余完全不变),
同一序列即干净,且 `GPU prefill ACTIVE=84`(装配确实跑了 84 次,不是"没走这条路")。

### 修复

```python
# mixed_experts.py
def gp_side_stream_enabled() -> bool:
    return os.environ.get("XIAOTU_GP_ASM_SIDE_STREAM", "0") == "1"   # 默认关

def gp_capturing() -> bool:
    try:    return bool(torch.cuda.is_current_stream_capturing())
    except Exception: return False

# apply() 内
_side_ok = gp_side_stream_enabled() and not gp_capturing()
```

### 验证

不设任何 env,依赖新默认值,跑那个此前 **5/5 必崩** 的序列:

```
step1 L=128   C=1 N=8 : ok=8/8  TTFT 1221 ms    Δillegal=0
step2 L=128   C=2 N=8 : ok=8/8  TTFT 2022 ms    Δillegal=0
step3 L=16384 C=1 N=1 : ok=1/1  TTFT 105602 ms  Δillegal=0
```

### 代价

| | 长 prompt TTFT |
|---|---|
| 旁路流开(旧默认) | ~89.8 s |
| 旁路流关(修复后) | ~105.4–108.8 s |

**+17%**。旁路流原本就是把这段 H2D 藏到 attention 后面用的;关掉退化为串行。
**后续应实现正确的流间同步(事件/依赖),把重叠换回来,而不是长期依赖关闭。**

### 排查中走过的弯路(留存以免重蹈)

1. **把异步粘性错误的浮出点当成根因**:先误判为 Triton 转置越界,直到 `XIAOTU_GPF_TT=0`
   仍崩才排除。
2. **自己实验脚本的 `kill -9` 制造了假故障**:脚本收尾 `kill -9 $(nvidia-smi ...)`
   只杀 worker,存活的 EngineCore 把它记成 `Worker proc died unexpectedly`,
   被我误读为「空闲期静默死亡」,并据此追查了 SIGTERM / memlock / OOM 数轮 —— **全是假象**。
   判定方法:考察「全程不做任何 kill」的对照组。
3. **prefix cache 混淆**:早期收窄实验里两条长请求用了**相同 seed**,第二条命中
   prefix cache(TTFT 89.7 s → 8.3 s),根本没跑装配,实验无效。
4. **开关设了没生效**:`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` 会被 `serve_*` 脚本写入的
   env-bridge 文件**覆盖**,`GPU_PREFILL=0` 也不会映射到它。**任何开关都必须先在日志里
   确认实际生效值。**

---

## 2026-09-21 · prefill 优化与 256K long prefill 交付

### Fixed

- **DeepSeek-V4.1-Flash（LTS）长 prompt 崩溃**：装配期 `aten::new_empty` 分配失败 ⇒ `EngineDeadError`。
  **根因定位到行**：`csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:1200-1204`
  在**每次 forward 内部**新建 `q_out = (q_in.size(0), q_head_padded, q_in.size(2))`（bf16），
  **大小 ∝ 本次 forward 的 token 数 `T`**。长 prompt 下该临时张量在 KV 池挤占显存后放不下。
  **修法（无需改代码）**：`MBT 16384 → 4096`，chunk 小 4 倍 ⇒ 该分配小 4 倍。
  **验证**：256K 下单条 16384 prompt `Successful=1`、`aten::new_empty=0`、`illegal=0`；
  prefill **115.8 tok/s**、decode **18.7 tok/s**。

- **GLM-5.3-Flash 在 256K 下 OOM**：满卡（39.40/39.49 GiB）后仅差 **120 MiB** 失败。
  修法同样是降 MBT（`16384 → 8192`，激活工作区 ∝ MBT）。**验证**：256K 下
  `Successful=1`、`[fp8-asm]=252`（= 42 层 × 3 chunk × 2 rank，自洽）、`DISABLED=0`、`illegal=0`；
  prefill **233.6 tok/s**、decode **21.4 tok/s**。

- **SM8x 稀疏 MLA prefill 慢 2.88×**：`_sparse_mla_fwd_with_sink_kernel` 在单条 16384 prefill 上
  占 **22.3 s / 58.8%** 的 CUDA 时间。二分定位：**全部收益来自把 `q` 载入为二维 `(1, BLOCK_D)` tile**
  （`BLOCK_H=1` 即 2.78×，KV 复用只再贡献 ~3%）。见 `patches/upstream/pr4-*.patch`。
  **验证**：该核 22.3 → **7.739 s**（占比 58.8% → 33.15%），端到端 TTFT 55,954 → **36,894 ms（1.52×）**。

### Changed

- **256K 及以上的 KV 上限必须按引擎反算的真实需求配置，不加乘性余量**：
  GLM 19,505 B/token、V4.1 30,639 B/token。256K 下 10% 余量 = 0.48 GiB，**实测直接 OOM**。
  （小 KV 场景如 32K/0.6 GiB 时该余量无害，因此这条只在长上下文暴露。）
- `gpu_prefill_fp8.py` 的 MoE GEMM tile 参数改为可覆盖：
  `XIAOTU_GPF_GEMM_{BM,BN,BK,BH,STAGES,WARPS}`，**默认值不变（64/64/64/64/2/4）**，
  使调优从「改源码+跑+回滚」变成「设环境变量+跑」。
- README / README_EN：性能表**移除 MiMo-V2.5 的数据**（后继 MiMo-2.6 即将发布；模型仍受支持），
  并新增**「长上下文（256K）long prefill」**一节记录 GLM 与 DeepSeek-V4.1-Flash 的成绩。

### Added

- `docs/PREFILL_KNOWN_ISSUES.md`：性能点总表、未解决项、**已证伪的 5 个假设（不要再试）**、
  测量方法上的 4 个坑、有效做法、上游关系澄清。
- `docs/EXPERIMENTS.md`：逐实验台账（B1–B28），含**我犯过并已更正的错误**与操作纪律。
- `patches/upstream/pr4-sm8x-sparse-mla-2d-tile.patch` + `pr4_body.md`。
- `docs/SM70_VOLTA_VERDICT.md`、`docs/SM70_VOLTA_FORK_PLAN.md`（V100/SM70 可行性调研）。

### 测量陷阱（本轮发现，影响判读）

- **`GPU prefill ACTIVE` 横幅在每请求 preflight 失败（slack 为负）时照样打印。**
  可靠判据只有两个：`XIAOTU_GPF_STAGE=1` 下 **`[fp8-asm]` 行数 > 0**，或 **`DISABLED` 未出现**。
  **注意**：`[fp8-asm]` 是 **GLM/FP8 路径**的标记，**V4.1（MXFP4）不打印它**，不能用它判定 V4.1。
- **profiler 的 `Self CUDA` 对 Python 层级 op 包含其子内核** ⇒ 父子行相加会重复计数
  （实测求和 48.34 s > 墙钟 36.86 s）。且各核**异步重叠**，故「核求和」不是墙钟。
- **清场必须杀 `VLLM::EngineCore` 子进程**：它不在任何 pidfile 里；验收判据是
  `nvidia-smi` **读到 0**，而不是 kill 命令返回成功。
- **`pgrep -f` / `pkill -f` 会匹配到自己的命令行**；`kill -9 -PGID` 会波及自己的进程组。

### 澄清

- `vllm/v1/attention/backends/mla/sparse_mla_kernels.py` 与 `flashmla_sparse_sm8x.py`
  **不在 vLLM main 上** —— 由本仓库 `patches/upstream/pr3-sm80-port.patch` 新增（3517 行）。
  本文件及 README 中与之相关的数字，测的都是**我们自己的 SM80 移植代码**。
  （曾据此误报上游 issue #57971，**已撤回并关闭**。）

---

## 2026-09-21 · 256K 长上下文显存排查（prefill 优化第二段）

### 结论：256K 下 `MBT=16384` 的**全部已试杠杆均失败**，`MBT=8192` 仍是可用配置

**本段没有提升吞吐**，但把「为什么 256K 下不能用更大的 MBT」查到了**实测根因**，
并**关闭了四条会诱导后人重复尝试的路径**。

### 根因（实测）

**约束是 KDA 线性注意力的 chunk 状态 `h` 逐层累积，不是 MoE 权重 staging。**

* **OOM 落在 `chunk_gated_delta_rule_fwd_h`**
  （`flash_linear_attention/ops/chunk_delta_h.py:352` 的 `h = k.new_empty(B, NT, H, V, K)`；
  `k` 是 bf16 ⇒ `h` 是 bf16）；
* **设备显存在一次请求内单调爬升 21.65 GiB**（外部轮询 `nvidia-smi`，2 Hz，84 点：
  18223 → 40395 MiB）⇒ ≈**0.62 GiB/层**，与 `h` 的 **0.5 GiB/层**
  （NT = T/64 = 256，H=64，V=K=128，bf16）× **34 个 KDA 层**吻合；
* **预算闭合**（单卡 39.49 GiB）：模型权重 11.10（日志）+ KV 4.76（日志）
  + staging 5.62 + inter 0.50 + 其余 ~16.8。
* **`h ∝ NT = T/BT`** ⇒ **MBT 减半即 `h` 减半（−8.5 GiB）** —— 这是 `MBT=8192` 能跑而 16384 不能的原因。

### 已关闭的路径（不要再试）

| 路径 | 实测结果 |
|---|---|
| 共享 `raw2` 与 `raw13` 的存储（省 1.12 GiB） | 分配层生效，但只让负载多跑 2 层 ⇒ 不够 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | 与不设时数字**逐字相同** ⇒ 不是碎片问题 |
| 调整 `GPU_UTIL` | util 上限不起作用（插件 staging 在 vLLM accounting 之外） |
| **`FLA_CHUNK_SIZE` 64 → 128** | **引擎初始化失败** ⇒ 该常量被 GDN/KDA/Kimi 共用，128 不兼容 |

### Fixed / Changed

* `gpu_prefill_fp8.py`：新增 `XIAOTU_GPF_RAW2_SHARE`（**默认 0 = 关闭**）——
  开启后 `raw2` 复用 `raw13` 存储，省 1.12 GiB 设备显存，且经
  `scripts/test_gpu_prefill_fp8_assembly.py` **逐字节验证**。
  **因在 256K 未兑现收益，默认关闭**（`=1` 可 opt-in）。
  ⚠️ 开启时 w13 的转置被提前到 w2 的 DMA **之前**，否则 w2 的 DMA 会覆盖 `raw13`。
* `gpu_prefill_fp8.py`：MoE GEMM tile 参数改为环境变量可覆盖
  （`XIAOTU_GPF_GEMM_{BM,BN,BK,BH,STAGES,WARPS}`，**默认值 64/64/64/64/2/4 不变**）。
* 新增 `patches/upstream/fla-chunk-size-env.patch`：把 `FLA_CHUNK_SIZE` 做成环境变量可覆盖
  （默认 64 不变）。**⚠️ 该 patch 经 B73 实测证明：设为 128 会导致引擎初始化失败，
  故它只是诊断工具，不是可用修法。**

### 验证与判据（本段新增）

* **`scripts/test_gpu_prefill_fp8_assembly.py`** —— 逐字节验证装配逻辑，**秒级，不需加载模型**；
  它在几秒内抓出了我第一版共享实现的守卫 bug（`w13t_bytes_equal=False`）。
* **`gate_up_kernel_fp8` 的独立测试台**（合成输入，不加载模型）：线上默认参数 60.6 ms，
  而线上 profile 的 2.57 s 是**42 层总和**（每层 61 ms）—— **吻合到 1%**。
  ⇒ 核本身 **8.1% 达峰，属正常水平**（先前"0.19% 达峰"是把一层的 FLOPs 除以 42 层的时间）。
* **外部显存轮询**（`nvidia-smi`，零改树）：本段唯一无风险且一次成功的测量手段。

### 已知未解

* `h` **为何逐层不释放**（结构上它应在层内释放）——机制未查；
* 约 **14 s** 的未归属时间（GPU median 98% 忙，而叶子核只加出 17.81 s）；
* pr4 的端到端数值与 `head_mask` 分支仍未验证。

### 下一步

**256K 下 `MBT=16384` 需要动激活/KDA 侧的结构**（让 `h` 更早释放 / 把 KDA prefill 再分块 /
换用不累积 `h` 的实现），**而不是继续调参**。三条都需改核且改前必须做数值 A/B。

### Added

* **256K 下的 GLM-5.3-Flash prefill 提升 14.0%（零代码改动）**：
  把长上下文配置的 `MBT` 由 **8192 改为 12288** ⇒
  **TTFT 70,203 → 61,562 ms（−12.3%）、prefill 233.6 → 266.3 tok/s（+14.0%）**，
  decode 21.4 tok/s 不变。
  判据：`Successful=1`、`[fp8-asm]=168`（= 42 层 × 2 chunks × 2 rank）、`DISABLED=0`、`illegal=0`、`OOM=0`。

  **机理**：`MBT=12288` 与 `8192` 同为 **2 个 chunk**，但首块更大 ⇒ 每-chunk 固定开销摊薄更多；
  而 `MBT=16384` 需要 1 chunk 却放不下（KDA 的 chunk 状态 `h ∝ NT = T/BT`，
  12288 的 34 层累积约 12.75 GiB 落在预算内，16384 的 17.0 GiB 超出）。

  **与 32K 数据的一致性**：32K 下 8192→232.9、12288→266.5（+14.4%），
  256K 实测 +14.0% ⇒ 该比值在两种上下文长度下一致。

  ⚠️ **这条配置一直存在于我们自己的 32K 测试数据里，却在 256K 上长期未被测试** ——
  因为排查始终聚焦于"MBT=16384 / 单 chunk"这个更大但达不到的目标。
  **教训：先穷举已有的中间选项，再追更大的目标。**

* **GLM 256K 下 `MBT` 三档的收益/代价对照**（详见 `docs/TUNING_GUIDE.md` §8）：
  | MBT | prefill | TTFT | 上下文上限（⚠️推断） |
  |---|---|---|---|
  | 4096 | 154.5 tok/s | ~106 s | ≈ 730K–785K |
  | **8192** | **233.6 tok/s** | 70.2 s | ≈ 496K–551K |
  | **12288** | **266.3 tok/s** | 61.6 s | ≈ 262K–496K |

  `13312 / 14336 / 16384` 实测均 OOM（分别死于第 41 / 41 / 35 层）。
  根因：KDA 的 chunk 状态 `h ∝ NT = T/BT` 且逐层累积 —— `MBT` 越小 `h` 越小、能留给 KV 的显存越多
  ⇒ 上下文越长；但 chunk 数越多 ⇒ 每 chunk 重流 141.8 GiB 权重 ⇒ 越慢。

* **DeepSeek-V4.1-Flash @256K：`MBT` 由 4096 改为 8192 ⇒ prefill 115.8 → 355.0 tok/s（3.07×）**（零代码改动）。
  TTFT **141,618 → 46,150 ms（−67%）**；decode 18.7 → 19.5 tok/s。
  判据：`Successful=1`、`DISABLED=0`、`aten::new_empty=0`、`illegal=0`、`OOM=0`。

  **同一机理、同一条教训**：V4.1 的 `MBT` 此前只测过 `16384 → 4096` 这个跳变，
  **中间档位从未二分**；而 4096 ⇒ 4 chunks、8192 ⇒ 2 chunks ⇒ 权重重流减半。
  增益（3.07×）**大于**纯 chunk 数之比（2×），说明首块更大还额外摊薄了每-chunk 固定开销。

  ⚠️ 这是本会话第四次「先穷举中间选项」的教训生效（前三次：GLM 的 `MBT=12288`、纯配置优先、外部轮询）。
