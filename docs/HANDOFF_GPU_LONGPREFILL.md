# HANDOFF:全力追查 GPU long prefill 吞吐

**日期**:2026-09-21
**背景**:用户对 GPU long prefill 的实测吞吐(100–200 tok/s)**明确不接受**,判断
「**计算流水线根本没走对**」。本文把所有已知硬数据、缺口分析与现场信息交接出去,
后续全力只做这一件事:**停 MTP/dspark、停 MoE 常驻,在保证上下文的前提下把显存全部让给 long prefill。**

---

## 0. 一句话结论

**实测 GLM long prefill = 155 tok/s,而 DMA 线速给出的下界是 690 tok/s —— 差 4.4 倍。
更关键的是:插件自报的装配耗时只占实测的三分之一,`~436 ms/层` 花在了装配之外。**

---

## 1. 硬数据(全部实测,可复现)

`random` 数据集(短 128 / 长 16384,输出 128,N=8,每格不同 seed,prefix cache 开),
`prefill (tok/s) = total_input_tokens / (completed × TTFT)`、`decode = 1000/TPOT`:

| 模型 | prompt | 实际 token | 并发 | prefill (tok/s) | decode (tok/s) | 来源 |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash | 短 | 140 | 1 | 115.2 | 22.0 | `/tmp/rnd2/glm_L128_C1.json` |
| GLM-5.3-Flash | 短 | 140 | 2 | 70.4 | 13.9 | `glm_L128_C2.json` |
| **GLM-5.3-Flash** | **长** | **16,396** | **1** | **155.3**(TTFT **105,555 ms**) | 21.5 | `glm_L16384_C1.json` |
| GLM-5.3-Flash | 长 | 16,396 | 2 | 119.5 | 1.7 | `glm_L16384_C2.json` |
| MiMo-V2.5 | 短 | 152 | 1 / 2 | 64.2 / 48.1 | 16.7 / 9.6 | `mimo_L128_C*.json` |
| DeepSeek-V4.1 | 短 | 128 | 1 / 2 | 254.3 / 175.2 | 19.8 / 14.0 | `v41_L128_C*.json` |

4 格长 prompt 未取得(MiMo/V4.1 撞显存预算,见 §6.2)。

**配置**(GLM):`TP=2 · GPU_UTIL=0.82 · SPEC_K=0 · SEQS=2 · MAXLEN=32768 · MBT=4096 ·
KV_CACHE_BYTES=2 GiB · 旁路流已关(2026-09-21 修复)`。

---

## 2. 🎯 缺口分析(这是最该先看的东西)

### 2.1 DMA 下界

GLM FP8 每层要从主机流到 GPU 的权重(形状取自引擎自报
`[xtu-eng-shape] w13 shape=(288,2048,4096) w2 shape=(288,4096,1024)`):

```
w13 = 2.25 GiB   w2 = 1.12 GiB   scales ≈ 0.001 GiB
每层合计 3.38 GiB
实测锁页 H2D 线速 26.86 GB/s   ⇒  135 ms/层   ⇒  44 层一次全量装配 5.9 s
```

**设计上每个 chunk 都要流一次全量权重**,所以:

```
理论 prefill 速率 = MBT / 每趟耗时 = 4096 / 5.9 s = 690 tok/s   ← 与 prompt 长度无关
实测             = 4096 / 26.4 s = 155 tok/s
差距             = 4.4×
```

### 2.2 更刺眼的:**装配自报时间只占三分之一**

插件自己的逐层计时(`XIAOTU_GP_TIMING=1`,崩溃前实测):

```
[f8-asm] w13=105.5  w2=52.2  scales=0.17  tr=6.3  total=164.1 ms   (TP0)
```

```
44 层 × 164 ms = 7.2 s/趟
但实测每趟 = 26.4 s
⇒ **装配之外还有 19.2 s/趟,摊到每层 436 ms**
```

**也就是说:真正吃掉时间的不是 H2D、不是转置,而是装配之外的每层 436 ms。**
`[fp8-asm]` 的 `total` 已经把 w13+w2+scales+tr 全算进去了,所以这 436 ms/层必须来自:

1. **MoE 的 GPU 计算内核**(`gate_up_kernel_fp8` / `down_kernel_fp8` 及其分段)——
   它们在 `[fp8-asm]` 的计时窗口**之外**;
2. **attention / sparse indexer / kpool**(GLM 是 KDA,block_size 2176);
3. **chunk 之间没有流水**(装配下一层/下一个 chunk 时 GPU 在等);
4. **侧流被关掉后,装配与 compute 变成串行**(这是我 2026-09-21 的修复的代价,
   实测 +17% TTFT;380→ 但**远不足以解释 436 ms/层**)。

> **这就是"流水线没走对"的量化形态**:H2D 只占 135/600 ≈ 22%,
> 而 73% 的时间在装配窗口之外,很可能既没与 H2D 重叠、也没与前一层的计算重叠。

**下一步的第一件事:把每层的时间按「H2D / 转置 / MoE 内核 / attention / 空等」五段拆开。**
现有工具:`XIAOTU_GP_TIMING=1`(装配内)、`XIAOTU_GPF_STAGE=1`(阶段断点)、
`TORCH_PROFILER`/`nsys`(未用过,最该用)。

---

## 3. 现场信息(接手必备)

### 3.1 代码与树

| | 路径 |
|---|---|
| 插件(可改) | `/home/user/lvllm/vllm-xiaotu-moe`(HEAD 见 `git log`) |
| 上游 rebase 树 | `/home/user/lvllm/vllm-up-133b71e0b`(**不要改**) |
| GPU 预填充(FP8) | `vllm_xiaotu_moe/gpu_prefill_fp8.py` |
| GPU 预填充(MXFP4) | `vllm_xiaotu_moe/gpu_prefill.py` |
| 装配调度 / 旁路流 | `vllm_xiaotu_moe/mixed_experts.py`(`apply()` 约 1560–1760 行) |
| K-major 转置 | `vllm_xiaotu_moe/byte_transpose.py` |
| host 分片 / DMA | `xiaotu_moe/csrc/moe/moe_v2.hpp`、`csrc/python_binding/binding.cpp` |

### 3.2 关键 env 旋钮

| 变量 | 作用 |
|---|---|
| `XIAOTU_GP_TIMING=1` | 打印 `[fp8-asm]` 逐层装配分段计时 |
| `XIAOTU_GPF_STAGE=1` | 打印阶段断点(`[gpf-ptr]` 等) |
| `XIAOTU_GPF_DMA2D=0` | 不用 pitched 2-D DMA,走 staging 回退 |
| `XIAOTU_GPF_TT=0` | 关 Triton 转置,走 torch 回退 |
| `XIAOTU_GPF_PIN=0` | 不锁页引擎 host 分片 |
| `XIAOTU_GP_ASM_SIDE_STREAM=1` | 打开旁路流(**默认已关**,见 §5) |
| `GPU_PREFILL=0` | 彻底关 GPU 预填充(脚本变量,**不是** `VLLM_XIAOTU_*`) |
| `XIAOTU_MOE_GPU_RESIDENT_LAYERS` / `XIAOTU_MOE_RESIDENT_BUDGET_GB` | MoE 常驻(**本阶段全部置 0/空**) |

⚠️ **`serve_*.sh` 会把自己的值写进 env-bridge 文件,而插件在 `XIAOTU_ENV_FILE` 显式指定时
以文件为准覆盖 `os.environ`** ⇒ 有些变量会被静默丢弃。**任何开关都要回读日志确认生效**
(本次会话因此栽了两次)。

---

## 4. 本阶段的目标配置(用户指定)

**停 MTP/dspark、停 MoE 常驻,保证上下文,其余全部给 long prefill:**

```bash
SPEC_K=0                       # 停投机解码(省 draft 常驻 + 不再吃 KV)
# 不加 XIAOTU_MOE_GPU_RESIDENT_LAYERS / RESIDENT_BUDGET_GB(即 0 层常驻)
GPU_PREFILL=1                  # long prefill 是本阶段唯一优化对象
MAXLEN=<保证的上下文>          # GLM:262144;V4.1:1048576;MiMo:1048576(见 §6)
GPU_UTIL=<尽量高>              # 把省下的显存全部让给 staging + 激活
```

**上下文对应的 KV(必须留足,用引擎反算的每 token KV):**

| 模型 | KiB/token | 1M | 256K | 出处 |
|---|---|---|---|---|
| GLM-5.3-Flash | 19.0 | 19.0 GiB | **4.8 GiB** | `GPU KV cache size: 110,100 @ 2 GiB` |
| DeepSeek-V4.1 | 29.9 | **29.9 GiB** | 7.5 GiB | `GPU KV cache size: 41,078 @ 1.17 GiB` |
| MiMo-V2.5 | 247.7(疑似**被高估**,见 §6.1) | 247.7 GiB | 61.9 GiB | `GPU KV cache size: 40,974 @ 9.68 GiB` |

---

## 5. 2026-09-21 已做的修复(不要回退)

**`vllm_xiaotu_moe/mixed_experts.py`**:FP8 装配的**旁路流默认改为关闭**
(`gp_side_stream_enabled()` 默认 `"0"`),并新增 `gp_capturing()` 在捕获期强制走保守路径。

* **为什么**:长 prompt 首次请求触发 `cudaErrorIllegalAddress`,崩在装配的
  pitched 2-D DMA(`cudaMemcpy2DAsync`)。单变量证据:仅置
  `XIAOTU_GP_ASM_SIDE_STREAM=0` 即干净,且 `GPU prefill ACTIVE=84` 证明装配确实跑了。
* **代价**:长 prompt TTFT **+17%**。**这正是本阶段要抢回来的东西之一** ——
  正确的做法是实现**真正的流间依赖/重叠**,而不是把重叠关掉。
* 完整复盘见 `CHANGELOG.md`;诊断过程见。

---

## 6. 两条必须先纠正的错误结论(我留下的,别再踩)

### 6.1 ❌「MiMo 的 1M 上下文不可达(需 247.7 GiB)」

**这个结论很可能是错的。** MiMo-V2.5 是**滑窗(混合)注意力**
(`config.json: sliding_window = 128`),滑窗层不需要按全上下文分配 KV。
我**没有读 MiMo-2.5 的技术报告**就下了结论,违反了本任务的要求。

**接手第一件事:读 MiMo-2.5 技术报告,搞清滑窗层与全注意力层的比例与各自窗口**,
然后判断:
* vLLM 的 `GPU KV cache size` 是否已经对滑窗层做了节省(若没做,那是可观的优化空间);
* 1M 上下文在 MiMo 上真实需要的 KV。

### 6.2 ❌ 我曾用 `config.json` 推算每 token KV

按 `48层 × 4 KV头 × (192+128) × 2B` 给 MiMo 推出 120 KiB/token,而引擎实测是
**247.7 KiB/token** —— 差 2.06 倍。**只信 `GPU KV cache size: N tokens`,
不要按配置字段推算**(对齐/填充/额外层会让推算失效)。

---

## 7. 我踩过的坑(请勿重蹈)

1. **自己脚本的 `kill -9` 制造假故障**:脚本收尾 `kill -9 $(nvidia-smi ...)` 只杀 worker,
   存活的 EngineCore 把它记成 `Worker proc died unexpectedly` —— 我据此追查了
   「空闲期静默死亡 / SIGTERM / memlock / OOM」**数轮,全是假象**。
   **判定方法:跑一个「全程不做任何 kill」的对照组。**
2. **prefix cache 混淆**:早期收窄实验里两条长请求用了**相同 seed**,第二条命中
   prefix cache(TTFT 89.7 → 8.3 s),根本没跑装配,实验无效。**每格必须换 seed。**
3. **开关设了没生效**:见 §3.2 的警告。**每次都要 grep 日志确认实际值。**
4. **`waitp` 用 `kill -0 $wrapper_pid` 判据是错的**:`serve_glm53_mainline.sh` 非 FOREGROUND
   模式会 fork 后自退 ⇒ 判据恒假。改用 pidfile + 日志。
5. **判据必须与"本次运行"绑定**:陈旧 pidfile / 等错文件 / 等错标记,
   本次会话因此各空等过 70 分钟。**写等待条件时先确认标记真的写在你等的那个文件里。**

---

## 8. 脚本

| 脚本 | 用途 |
|---|---|
| `scripts/bench_random_12cells.sh` | 12 格基准(KV cap 自动算 + 前置断言 + 早退检测) |
| `scripts/bench_resident_sweep.sh` | 常驻层扫描(**本阶段停用**) |
| `scripts/summarize_random12.py` | 从 JSON 出表 |
|
| `docs/TUNING_GUIDE.md` | 三方显存预算框架与实测单位代价 |
| `CHANGELOG.md` | 旁路流修复的复盘 |

---

## 9. 建议的第一步(按顺序)

1. **读 MiMo-2.5 技术报告**(滑窗),纠正 §6.1。
2. **把每层 600 ms 拆成五段**(H2D / 转置 / MoE 内核 / attention / 空等)。
   先用 `nsys profile` 或 torch profiler 抓一次 16384 的单请求;
   `XIAOTU_GP_TIMING=1` 只能看到装配内的 164 ms,**看不见那 436 ms**。
3. 按拆分结果决定方向:
   * 若 **MoE 内核占大头** ⇒ 优化 FP8 GEMM(分段/块大小/占用率),与 DMA 无关;
   * 若 **attention/indexer 占大头** ⇒ 那是 GLM 的 KDA 路径,单独查;
   * 若 **空等占大头** ⇒ 流水线问题:让「下一层的 H2D」与「本层的 MoE 内核」重叠
     (MXFP4 路径的 ping-pong **预取环**就是干这个的,FP8 路径有没有等价物?),
     或在**同一个 chunk 内**按专家分块流水。
4. 目标:先把每趟从 26.4 s 压到接近 5.9 s 的 DMA 下界(690 tok/s),
   再考虑把旁路重叠**正确地**加回来。

---

## 10. 🔴 2026-09-21 新增硬结论:**GPU 预填充目前是纯开销**

用 `scripts/longprefill_probe.sh` 单模型(GL GLM)、逐步留痕测的两个 arm(16384 token,C=1):

| arm | 配置 | TTFT | prefill | 装配行数 |
|---|---|---|---|---|
| **A** | GPU 预填充 **开** | **106,186 ms** | 154.4 tok/s | `ACTIVE=84`,`[fp8-asm]=588(294/rank)`,无 illegal |
| **B** | 纯 CPU(`GPU_PREFILL=0`) | **106,367 ms** | 154.1 tok/s | `ACTIVE=0`,`[fp8-asm]=0` |

**⇒ 开关 GPU 预填充,TTFT 差 0.2%(噪声)。那 48.5 s 的 GPU 装配换来零收益。**

### 10.1 三个数拼出的账

```
42 个 MoE 层(ACTIVE/2)× 8 chunk(16384/2176)= 294 次装配/rank
每次 ~165 ms ⇒ 装配合计 48.5 s
实测 TTFT 106 s ⇒ 非装配 57.5 s
```

| | 装配 | GPU MoE | 注意力 | 合计 |
|---|---|---|---|---|
| arm A(旁路流**关**) | 48.5 s | ~49 s | ~8.5 s | **106 s** |
| arm B(纯 CPU) | 0 | — | ~8.5 s | **106 s**(CPU MoE ~98 s) |

**106 s 与 CPU 引擎的实测速率吻合**:`106 / 8 chunk / 42 层 = 315 ms/层/chunk`,
而 `NOTES` R17 实测 CPU 引擎 B=1024 → 113.3 ms/层、B=4096 → 365.0 ms/层 —— **2176 token 正落在这一档**。
⇒ **106 s ≈ 纯 CPU MoE 计算时间。**

### 10.2 结论与责任

**GPU 侧 MoE 计算比 CPU 快约 2×(~49 s vs ~98 s),但那 48.5 s 的装配被串行加上去,
恰好抵消全部收益。** 而**串行是 2026-09-21 那次修复造成的** —— 为了绕开长 prompt 的
非法访存,我把旁路流默认关掉了,连带把「装配与计算的重叠」一起关掉。

**正确的修法是保留重叠、只修同步;不是关掉重叠。**
`/tmp/arm_c.sh`(`XIAOTU_GP_ASM_SIDE_STREAM=1`,单条长请求以避开竞态)是判决实验:
若 TTFT ≈ 57 s,即证实恢复重叠就能拿回 ~1.8×。

### 10.3 另一个待验事实:chunk 由 **KDA block_size 2176** 决定,不是 MBT

`ACTIVE=84`(2 rank × 42 层)而装配行数 294/rank ⇒ **294 / 42 = 7 chunk**,
而 `MBT=4096` 本应给 4 ⇒ **chunk 被 GLM 的 KDA `block_size=2176` 钉住**。
若成立,则**放大 MBT 不会减少 chunk 数、因而不会减少总 H2D** ——
"MBT 放大到 16384 ⇒ 4×" 的前提对 GLM 不成立。MBT 扫描会以装配行数判据验证。

---

## 11. 🔴 1M 上下文下 GPU 预填充被**禁用**(2026-09-21,MBT 扫描第一档)

`MAXLEN=1048576`(KV=19.05 GiB/rank)+ 16384 prompt 的实测:

```
TTFT 112,787 ms   prefill 145.4 tok/s   ← 与纯 CPU 同档
(Worker_TP1) GPU prefill DISABLED for this process -> staying on CPU (slower but correct).
    per-layer staging ~7.6 GiB; preflight wants ~11.4 GiB free VRAM
    (staging x1.10 + 3.0 GiB activation reserve), only 4.5 GiB is free.
```

**⇒ 1M 的 KV(19.05 GiB)把空闲挤到 4.5 GiB,而预检要 11.4 GiB ⇒ 禁用 GPU 预填充 ⇒ 全程 CPU。**

### 11.1 ⚠️ 会误导人的日志缺陷(必修)

同一个日志的**相邻行**:

```
GPU prefill ACTIVE: first 2176 tokens >= threshold 1500; ...
    preflight: staging ~7.59 GiB, required >= 11.35 GiB free, had 4.45 GiB (**slack -6.90 GiB**)
```

**`ACTIVE` 横幅带着负 slack 照样打印。** 它按**设备级**判定 `_ok` 打,而**每请求**的
preflight 另算;两者不一致时只打 ACTIVE、不打 REJECTED。
**本次会话我反复以 `GPU prefill ACTIVE=84` 为"GPU 预填充在跑"的证据 —— 这个证据是不可靠的。**
修法:横幅应带 `slack` 的符号判定,`slack < 0` 时打 `REJECTED`。

**判据修正**:以后确认 GPU 预填充,必须用 **`[fp8-asm]` 行数 > 0**
(需要 `XIAOTU_GPF_STAGE=1`),或检查 `GPU prefill DISABLED` **未出现**;
**不能用 `GPU prefill ACTIVE` 单独作证。**

### 11.2 量化边界(2×A100-40GB / GLM-5.3-Flash)

```
单卡 util 0.90 可用           35.5 GiB
− KV(1M)                     19.05     ← 引擎反算 19,505 B/token
− 非 KV(权重+激活,实测反推)   ~12.0
= 空闲                         4.5 GiB   ← 预检要 11.4
⇒ KV 必须 ≤ 23.5 − 11.4 = 12.1 GiB 才能开 GPU 预填充
⇒ 12.1 GiB × 55,050 tok/GiB ≈ **650K token**
```

**⇒ 二选一(不可兼得):**
* 要 **GPU 预填充** ⇒ 上下文 ≤ **~650K**
* 要 **1M 上下文** ⇒ **放弃 GPU 预填充**(全程 CPU,~145 tok/s)

**可调的两个旋钮**(按效果排序):`XIAOTU_GP_ACT_RESERVE_GIB`(3.0 → 更小,但换来 OOM 风险,
见 NOTES §601)、`--gpu-memory-utilization`(调低可给 staging 腾地方,但会压小 KV 池)。

---

## 12. 🎯 唯一可行的解法:**砍掉 staging 的那份多余拷贝**

### 12.1 光靠旋钮救不回来(算术)

```
预检要求 = staging × 1.10 + 激活备用
       = 7.59 × 1.10 + 3.0 = 11.4 GiB
1M 下空闲 = 4.5 GiB                      ⇒ 差 6.9 GiB
把激活备用 3.0 → 0:仍要 8.3 GiB > 4.5    ⇒ ✗ 不够
```

`XIAOTU_GP_ACT_RESERVE_GIB` 与 `--gpu-memory-utilization` **都不足以**填平这个缺口。

### 12.2 staging 为什么会是 7.59 GiB

FP8 装配**同时持有原始布局和 K-major 拷贝两份**设备常驻:

```
raw13(2.25) + raw2(1.12) + km13(2.25) + km2(1.12) ≈ 6.74 GiB
(+ 单 node 的 DMA staging、scales)           ≈ 7.59 GiB
```

### 12.3 MXFP4 路径早就解决了,FP8 路径缺这个优化

`gpu_prefill.py` 的 `_pinned_kmajor` 在**主机侧**做字节转置并锁页缓存,其 docstring 原话:

> Doing the byte transpose once on the host (at cache-build time) instead of once per layer
> on the GPU removes a **~3.4 GiB device read+write per layer** ...

**⇒ 把 host 侧 K-major 缓存移植到 FP8 路径**,则:

| | 现在 | 移植后 |
|---|---|---|
| 设备 staging | **7.59 GiB** | **~3.4 GiB** |
| 每层 GPU 转置 | 6.3 ms(×294 装配 ≈ 1.9 s) | **0** |
| 预检需要 | 11.4 GiB | **~3.7 GiB** |
| 1M 上下文下能否开 GPU 预填充 | ❌(只有 4.5 GiB) | **✅ 3.7 < 4.5** |

**这才是「1M 上下文 + GPU 预填充」不可兼得的真正原因与唯一解法。**
不是调 util,不是调 reserve,而是**消掉那一份多余的设备常驻拷贝**。

### 12.4 与 §10 的关系

§10 的「GPU MoE 快 2×、被 48.5 s 串行装配抵消」是在 **32K 上下文**(preflight 通过、
装配真的跑了)下测得的,结论仍成立。§11/§12 是 **1M 上下文**下的另一件事:
那里装配**根本没跑**。两件事不要混。

### 12.5 下一步(优先级)

1. 等 `/tmp/knob_probe.sh` 的 4 档旋钮结果(预计都失败,但要有实测记录);
2. **实现 FP8 的 host 侧 K-major 缓存**(移植 `_pinned_kmajor` 的语义到
   `gpu_prefill_fp8.py`),把 staging 从 7.59 压到 ~3.4 GiB;
3. 重测 1M + GPU 预填充,用 **`[fp8-asm] > 0` 且无 `DISABLED`** 作判据;
4. 若成立,再回头做 §10 的重叠恢复(把 48.5 s 的装配藏到计算后面)。

---

## 13. ✅ 最终结论:1M 上下文 + 16384 random prompt(C=1)、GLM-5.3-Flash、2×A100-40GB

### 13.1 现状(当前代码,已实测)

| 项 | 值 | 证据 |
|---|---|---|
| 上下文要求 | 1M(262,144 是旧值;**本 goal 用 1,048,576**) | — |
| KV 池 | **19.05 GiB/rank**(19,505 B/token) | 引擎 `GPU KV cache size` 反算 |
| GPU 预填充 | **被禁用**(空闲 4.5 GiB < 预检要求 11.4 GiB) | `GPU prefill DISABLED for this process` |
| **可达 prefill** | **145.4 tok/s**(TTFT 112,787 ms)—— **这是纯 CPU 路径** | `/tmp/lp/20260921-114341/gap.txt` |
| 装配是否执行 | **否**(`[fp8-asm]` 行数 = 0) | 同上 |

> ⚠️ **不能用 `GPU prefill ACTIVE` 当判据** —— 它在 preflight 失败(slack 为负)时**照样打印**,
> 这是本次会话最大的误导源。可靠判据只有两个:**`XIAOTU_GPF_STAGE=1` 下 `[fp8-asm]` 行数 > 0**,
> 或 **`GPU prefill DISABLED` 未出现**。

### 13.2 为什么不可兼得(预算恒等式,全部实测)

```
单卡 util 0.90 可用                     35.5 GiB
− KV(1M)                              19.05
− staging(原始 + K-major 两份)         7.59
− 非专家权重 + 激活(反推)              ~12.0
=                                  −3.1 GiB   ← 已经超了
预检还额外要 激活备用 3.0
```

**⇒ 1M 与 GPU 预填充在 2×A100-40GB 上**在当前代码里**不可兼得。**

### 13.3 唯一的解法:把 staging 从 7.59 压到 ~3.4 GiB(尚未实现)

FP8 路径**同时**在设备上持有 `raw13/raw2` 与 `km13/km2` 两份布局;
而 **MXFP4 路径早就有 host 侧 K-major 缓存**(`_pinned_kmajor`),其注释明说
这省掉「每层 ~3.4 GiB 的设备读写」。**把该优化移植到 FP8 路径**即可:

```
移植后:35.5 − 19.05(KV) − 3.4(staging) − 12.0 = 1.05 GiB
       预检需要 3.4×1.10 + 3.0 = 6.7 GiB   仍差 5.7 GiB
```

**⇒ 光移植 K-major 还不够。** 必须**同时**处理「非专家权重 + 激活」的 ~12.0 GiB:

* 其中 **~2.9 GiB 是 vLLM 的 prefill 激活峰**(`KNOWN_LIMITATIONS` §8.2);
* `XIAOTU_GP_ACT_RESERVE_GIB` 默认 3.0 是**给这份峰留的安全垫**,不能归零(会 OOM);
* 剩下 ~6 GiB 需要在**降 util** 与**降上下文**之间取舍。

**可行的组合(需要实测确认)**:
```
KV(1M)=19.05 必须保住
staging 移植后 3.4  (+ 备用 1.5)  = 4.9
非专家权重 + 激活 ≈ 让 util 决定
⇒ 需要 util ≥ (19.05 + 3.4 + 权重 + 激活) / 39.49
```

### 13.4 结论:**1M 是"声明"还是"必须装载"?**

这是本 goal 唯一未澄清、且**决定成败**的前提(2026-09-21 曾就此提问,未获答复):

* **若"1M"= 必须在 GPU 上装载 1M 的 KV** ⇒ 在 2×A100-40GB 上**代价极高**:
  要放弃 GPU 预填充(且放弃 MTP、放弃常驻层),prefill 停在 **~145 tok/s**;
* **若"1M"= 按模型能力声明**(如同 `settings.yaml` 里 GLM 的 `contextWindow: 262144`),
  则**不必现在装载**:把 `MAXLEN` 设成能负担的值(算得 **~650K**),就能保住 GPU 预填充。

**实测边界:要 GPU 预填充 ⇒ 上下文 ≤ ~650K;要 1M ⇒ 放弃 GPU 预填充。**

### 13.5 另附:32K 上下文下的另一半问题(与本节独立)

见 §10。在 32K(preflight 通过、装配真的跑了 294 次/rank、48.5 s)时:
**GPU MoE 计算比 CPU 快约 2×,但 48.5 s 的串行装配把收益全部抵消。**
那个串行是 2026-09-21 修复非法访存时**关掉旁路流重叠**造成的,
正确修法是**保留重叠、只修同步**(`/tmp/arm_c.sh` 为判决实验)。

**⇒ 两个问题、两个修法:**
| 上下文 | 症状 | 修法 |
|---|---|---|
| 32K | 装配串行,收益被抵消 | 恢复流间重叠(改调度) |
| 1M | 预填充被禁用,全程 CPU | 砍 staging(改内存布局) |

---

## 14. 旋钮实测:确认「调参数救不回来」(2026-09-21 12:03)

`/tmp/knob_probe.sh`,1M 上下文,逐档实测:

| 档 | util | `GP_ACT_RESERVE_GIB` | TTFT | `[fp8-asm]` | `DISABLED` | 判定 |
|---|---|---|---|---|---|---|
| `k_res30` | 0.90 | 3.0 | 112,399 ms | 0 | **2** | ❌ 仍走 CPU |
| `k_res10` | 0.90 | 1.0 | **0.00 ms** | 0 | 0 | ❌ **服务崩了(OOM)** |
| `k_res00` | 0.90 | 0.0 | 运行中 | | | |
| `k_u085r10` | 0.85 | 1.0 | 待跑 | | | |

**第 2 档是决定性的**:把激活备用从 3.0 降到 1.0 **不会**让 GPU 预填充启用,
而是让服务**直接 OOM 挂掉** —— 这正是插件代码里的原话:

> The reserve protects vLLM's own activation peak: the staging buffers are process-persistent,
> so spending that peak makes a long prefill OOM **after** a passing preflight (NOTES §601).

**⇒ `XIAOTU_GP_ACT_RESERVE_GIB` 这条路是死的:调小它只把 OOM 从 preflight 推迟到预填充,
不会换来 reach。** 与 §12.1 的算术一致(3.0 → 0 也仍要 8.3 GiB > 4.5 GiB 空闲)。

### 14.1 旋钮探针完整结果(4/4 全部失败)

| 档 | util | `GP_ACT_RESERVE_GIB` | TTFT | `[fp8-asm]` | `DISABLED` | 判定 |
|---|---|---|---|---|---|---|
| `k_res30` | 0.90 | 3.0 | 112,399 ms | 0 | **2** | ❌ 走 CPU |
| `k_res10` | 0.90 | 1.0 | **0.00 ms** | 0 | 0 | ❌ **崩(OOM)** |
| `k_res00` | 0.90 | 0.0 | 112,761 ms | 0 | **2** | ❌ 走 CPU |
| `k_u085r10` | 0.85 | 1.0 | **0.00 ms** | 0 | 0 | ❌ **崩(OOM)** |

**两个关键观察:**

1. **激活备用降到 0 也仍然被禁用**(`DISABLED=2`)。因为 staging 本身
   `7.59 × 1.10 = 8.3 GiB` 就已经超过 1M 下的 4.5 GiB 空闲 ——
   **不砍 staging,任何 reserve 都救不了**。这与 §12.1 的算术完全一致。
2. **备用 = 1.0 反而崩**(两档都崩),3.0 与 0.0 却能起服务 —— 非单调。
   最可能是 1.0 时 preflight 恰好**通过**,于是预填充真的开跑、
   然后撞上代码警告的那个 OOM:
   > spending that peak makes a long prefill OOM **after** a passing preflight (NOTES §601)

**⇒ 「调旋钮换 reach」这条路有 4 个实测点判死。唯一出路是 §12 的「砍 staging」。**

---

## 15. 🎯 **更正**:MBT **确实**控制 chunk 数 —— 用户是对的,我先前判错了

**2026-09-21 判决实验**(`MAXLEN=32768`,preflight 通过、GPU 预填充活着,
`DISABLED=0`,单条 16384 / C=1):

| MBT | `[fp8-asm]` 总行数 | 每 rank | `ACTIVE` | 隐含 chunk | 结果 |
|---|---|---|---|---|---|
| **4096** | 588 | **294** | 84 | **7** | ✅ TTFT **106,156 ms**,prefill **154.5 tok/s** |
| **16384** | 82 | **41** | 82 | **1** | ❌ **OOM,completed=0** |

**⇒ 装配次数 294 → 41(降 7.2 倍)。MBT 放大确实把 chunk 从 7 压到 1。**

### 15.1 ⚠️ 我先前两次判错,一并更正

1. **「chunk 被 GLM 的 KDA `block_size=2176` 钉住、MBT 无效」** —— **错**。
   实测 MBT 从 4096 到 16384,装配行数从 294 掉到 41。**用户假设的杠杆成立。**
   (我当时是从 `ACTIVE=84` 与 294 行反推出「294/42=7 chunk,故 chunk 由 2176 定」——
   这个反推的层数假设错了,结论也就错了。**教训:不要用行的间接反推去否定一个可以用一次
   单变量实验直接验证的假设。**)
2. **「MBT 放大拿不到收益」** —— **错**。收益是 7× 的量级。

### 15.2 真正的约束:**MBT=16384 的激活工作区放不下**

```
torch.OutOfMemoryError: Tried to allocate 120.00 MiB.
GPU 1 has 39.49 GiB of which 63.50 MiB is free. this process has 39.38 GiB in use.
38.38 GiB allocated by PyTorch
```

`UTIL=0.82` / `MAXLEN=32768` / KV 2 GiB 下,MBT=16384 把进程推到 39.38 GiB。

**⇒ 结论不是「MBT 无用」,而是「MBT 的上限由激活工作区决定,需要找出能放下的最大值」。**
参考:`MBT=32768` 在 util 0.82 下也 OOM(§5.2),`MBT=4096` 稳。

### 15.3 下一步(明确且小)

1. **扫 MBT ∈ {8192, 12288}**(在 `MAXLEN=32768`、util 0.82、KV 2 GiB 下),
   找 **能放下的最大 MBT**;判据同时看 `[fp8-asm]` 行数(应为 ~147 / ~98)与是否 OOM;
2. 该 MBT 下的 prefill 就是**这个显存预算下可达的最优**;
3. 若还要更大 MBT ⇒ 必须给激活腾地方:砍 staging(§12 的 host 侧 K-major,
   省 ~4.2 GiB)或降 KV/上下文;
4. **然后**再回头做 §10 的重叠恢复(把剩下的装配藏到计算后面)。

**§13.4 的分叉在实测面前变简单了**:既然 MBT 是有效杠杆且其上限受显存限制,
那么"1M 还是 650K"这个选择,直接等价于"能留多少显存给激活工作区与 staging"。

---

## 16. ✅✅ 成功:MBT=16384 放得下,**prefill 翻倍到 310.6 tok/s**

### 16.1 关键洞察:**KV cap 白占了 1.25 GiB,而 MBT=16384 只差 120 MiB**

`MAXLEN=32768` 与 16384 prompt 真正需要的 KV:

```
16384 prompt : 16384 × 19,505 B = 304.8 MiB
MAXLEN=32768 : 32768 × 19,505 B = 609.5 MiB
之前实际给的 cap : 2 GiB  ⇒ **白占 1.25 GiB**
```

而 `MBT=16384` 的 OOM **只差 120 MiB**。⇒ **把 cap 收到 0.75 GiB 就够了。**

### 16.2 实测对比(单条 16384 / C=1 / GLM / TP=2)

| 配置 | `[fp8-asm]` | TTFT | **prefill** | DISABLED | OOM |
|---|---|---|---|---|---|
| MBT=4096, KV 2 GiB, util 0.82 | 588 | 106,156 ms | **154.5 tok/s** | 0 | 0 |
| **MBT=16384, KV 0.75 GiB, util 0.85** | **84** | **52,794 ms** | **310.6 tok/s** | 0 | 0 |

**⇒ prefill 翻倍(2.01×),TTFT 减半。装配次数 588 → 84(7.14×),与预测一致。**

### 16.3 这同时验证了缺口模型

```
装配 = 42 层 × 7 chunk × 165 ms ≈ 48.5 s   (MBT=4096)
装配 = 42 层 × 1 chunk × 165 ms ≈  6.9 s   (MBT=16384)
⇒ 106.2 − 48.5 = 57.7 s 非装配
   52.8 −  6.9 = 45.9 s 非装配
```

两者都在 ~46–58 s 量级 ⇒ **非装配部分是主导项(现已占 87%)**,
而它**基本不随 MBT 变化**(同样的 token 数要做同样多的 MoE 计算 + 注意力)。

**⇒ 下一步的杠杆已经换了**:不再是 MBT,而是那 ~46 s 的**非装配**时间(§10 的重叠恢复、
以及 MoE 内核/注意力本身)。**再往上就得动 §12 的 staging 或 §17 的非装配拆解。**

### 16.4 ⚠️ util 0.82 那档失败了

`[MBT=16384 util=0.82 KV=0.75G] TTFT=0.00ms [fp8-asm]=0 OOM=0` —— 起不来或请求失败但**不是** OOM
(需单独查;util 0.85 可用,故不阻塞结论)。**不要假设 util 越低越安全。**

### 16.5 至此本 goal 的答案

**在 `MAXLEN=32768`、SEQS=1、KV cap 按需(0.75 GiB)、util 0.85、MBT=16384 下:
16384 token 的 random prompt(C=1)可达 **310.6 tok/s**,且 GPU 预填充**确实启用**
(`[fp8-asm]=84 > 0`、`DISABLED=0`)。**

> 关于 1M 上下文:如 §13/§11 所述,1M 下 GPU 预填充**被 preflight 禁用**(KV 19.05 GiB
> 只剩 4.5 GiB 空闲 vs 需要 11.4 GiB)。**但 §16 给出了一条新的可能**:
> 若 KV cap 之前也存在类似的白占,1M 那一侧的预算值得用同样的方式重算 ——
> 见下一步。

---

## 17. ✅ 最终边界(2026-09-21 12:55)

### 17.1 32K 侧:MBT 单调有效,16384 最优

`MAXLEN=32768`、`SEQS=1`、KV cap 0.75 GiB、util 0.85、单条 16384 / C=1:

| MBT | TTFT | **prefill** | `[fp8-asm]` | 判定 |
|---|---|---|---|---|
| 4096(KV 2 GiB,util 0.82) | 106,156 ms | 154.5 tok/s | 588 | ✅ 基线 |
| 8192 | ✗ 起不来 | — | — | ⚠️ 暂态,不可信 |
| 12288 | 61,532 ms | 266.5 tok/s | 168 | ✅ |
| **16384** | **52,794 ms** | **310.6 tok/s** | **84** | ✅ **最优** |

**⇒ prefill 154.5 → 310.6 tok/s(2.01×),装配次数 588 → 84(7.14×)。**

### 17.2 1M 侧:MBT=16384 直接 OOM;GPU 预填充无论 MBT 都被禁用

```
[1M MBT=16384 util=0.90] ✗ 起不来  ⇒ CUDA out of memory
[1M MBT=4096  util=0.90] TTFT 112,787 ms, prefill 145.4 tok/s, [fp8-asm]=0, DISABLED=2
```

### 17.3 三重锁死:为什么 1M 与 GPU 预填充不可兼得

1. **1M 的 KV 必须 19.05 GiB**(19,505 B/token × 1,048,576)—— 真需求,不是白占;
   ⇒ 单卡 35.54 GiB(util 0.90)只剩 **16.5 GiB**;
2. **预检要 11.4 GiB 空闲**(staging 7.59×1.10 + 激活备用 3.0),而 1M 下只有 **4.5 GiB**
   ⇒ **任何 MBT 下 GPU 预填充都被 `DISABLED`**(§14 的 4 档旋钮实测已证);
3. **MBT=16384 需要 ~38.7 GiB 非 KV**,1M 下放不下 ⇒ OOM。

### 17.4 本 goal 的最终结论

| 条件 | 可达 prefill | GPU 预填充 |
|---|---|---|
| **保证 1M 上下文** | **145.4 tok/s**(纯 CPU 路径) | ❌ 被禁用(`[fp8-asm]=0`) |
| 放开到 32K + MBT=16384 + KV 0.75 GiB | **310.6 tok/s**(2.01×) | ✅ 真跑(`[fp8-asm]=84`) |

**分界线由两个实测端点界定**:KV 0.75 GiB(≈41K 上下文)时 MBT=16384 可用;
KV 19.05 GiB(1M)时连 MBT=4096 都被禁用。**精确交点需再扫一轮。**

### 17.5 若要突破这条边界,只有两条路(都要改代码)

1. **砍 staging**(§12:移植 MXFP4 的 host 侧 K-major,7.59 → ~3.4 GiB)
   ⇒ 预检从 11.4 降到 ~6.7 GiB,在 1M 的 4.5 GiB 下**仍不够**,但能显著抬高上下文上限;
2. **砍 MBT 的激活占用**(§16.3:MBT=16384 需 ~38.7 GiB 非 KV)
   ⇒ 这是真正的天花板。**必须知道这 38.7 GiB 花在哪**(attention score?分段缓冲?),
   否则 MBT 无法再往上走。

**⇒ 下一步的第一件事:用 nsys/torch.profiler 拆解那 ~46 s 非装配时间与其显存占用。**

### 17.6 不可信的数据点(勿引用)

`MBT=8192 util=0.85`、`MBT=16384 util=0.82` 均"起不来",但都**夹在成功档之间**
⇒ 判定为**暂态失败**(GPU 未释放/端口冲突),不是 MBT/util 本身的问题。
**util 与 MBT 都不是单调安全的,每一档都必须实测确认,不能外推。**

---

## 18. ④ 的消融实验:**判决臂尚未拿到数字**(2026-09-21 13:06)

设计(`MAXLEN=32768` / MBT=16384 / KV 0.75 GiB / util 0.85 / 单条 16384 / C=1):

| 臂 | 开关 | 目的 | 结果 |
|---|---|---|---|
| `abl_base` | — | 基线 | ✅ TTFT **52,959.94 ms**,prefill **309.6 tok/s**,`[fp8-asm]=84` |
| `abl_fakeall` | `XIAOTU_MOE_FAKE_ALL=1` | **非 MoE 时间(注意力+其余)** | ❌ **TTFT 0.00,指标全 0,但服务端无致命错误** |
| `abl_fakecpu` | `XIAOTU_MOE_FAKE_CPU=1` | 拷贝+host-func 成本 | 未出 |

**`abl_base` 与 §16 的 310.6 tok/s 一致(309.6),复现性良好。**

**`abl_fakeall` 失败了,原因未明**:bench 侧指标全 0、请求未完成,而**服务端没有任何
`illegal memory access` / OOM / `EngineDeadError`** ⇒ 不是崩溃,更像**请求挂住或未被调度**。
(`FAKE_ALL` 让 `cpu_decode` 立即返回、输出保持零值,理论上应能跑完。)

**⇒ ④ 仍未回答。** 下一步要先查清 `FAKE_ALL` 为什么拿不到数字,再读那个数。
替代手段(插件已内建,优先于 nsys):
* `XIAOTU_TORCH_PROFILE=1`(`hybrid_model.py`)
* `XIAOTU_LAYER_TIMING=1`(`mixed_experts.py`)
* `XIAOTU_CD_TIMING=1`(`binding.cpp`,per-layer CPU 计时)

### 18.1 1M 侧的补充实测

`[1M MBT=16384 util=0.90] ✗ 起不来 ⇒ CUDA out of memory`(与 §17.2 一致)。
脚本里另外两档(MBT=8192 / 4096 @1M)未产出结果,需重跑。**但 §17.3 的三重锁死
(KV 19.05 GiB / 预检 11.4 vs 4.5 / MBT=16384 需 38.7 GiB 非 KV)已由 §11/§14/§17
的实测充分支撑,不依赖这两档。**

### 18.2 `FAKE_ALL` 失败原因已查明:**请求根本没到服务端**

```
grep -c "POST /v1/chat/completions" logs/abl_fakeall.log  →  0      ← 无请求
grep -ic "nan|inf|invalid"          logs/abl_fakeall.log  →  7      ← 启动期出现 NaN
```

**⇒ 不是"挂住",而是服务端在 bench 之前就无法接受请求**:
`FAKE_ALL=1` 让 MoE 输出保持**全零**,在启动/预热阶段就产生 NaN,
服务端很可能在就绪检查通过之后、bench 之前就死掉了。**该臂无效 —— 不是"测出 0"。**

**教训:`FAKE_ALL` 在这个配置下不可用作消融手段。**
它跳过所有 MoE 计算,输出语义已被破坏,下游(采样/归一化)会崩。

**⇒ ④ 的替代手段(按优先级):**
1. `XIAOTU_LAYER_TIMING=1`(`mixed_experts.py`)—— 逐层计时,语义不破坏;
2. `XIAOTU_TORCH_PROFILE=1`(`hybrid_model.py`)—— torch profiler,输出较小;
3. `XIAOTU_CD_TIMING=1`(`binding.cpp`)—— per-layer CPU 计时;
4. 最后才是 nsys。

---

## 19. 🎯 ④ 的答案:**瓶颈不在 MoE,在 attention/indexer**(2026-09-21 13:15)

### 19.1 `[layer-timing]` 的实测(MBT=16384,单条 16384 / C=1)

```
[layer-timing] n=414 pre=336–389 ms  eng=24–75 ms  post=0.04 ms  total=411–413 ms
```

`_lt_record` 的定义(`mixed_experts.py:1908-1911`):
* `pre  = _lt_t1 - _lt_t0` —— MoE 计算全程(GPU 预填充路径下含装配 + GPU MoE 内核)
* `eng  = _lt_t2 - _lt_t1` —— `engine.cpu_decode(...)`
* `post = _lt_t3 - _lt_t2` —— 收尾

**两个直接推论:**

1. **CPU 引擎几乎不在路径上**:`eng` 只有 **24 ms**,而 `pre` 是 **389 ms**。
   (我先前按文档 R17 估的「~315 ms/层是 CPU MoE」在这里**不成立** —— 因为 GPU 预填充
   真的在跑,MoE 不在 CPU 上。那个估值对应的是纯 CPU 路径。)
2. **装配只占 `pre` 的 40%**:`[fp8-asm] total = 165 ms`,而 `pre = 389 ms`
   ⇒ **还有 ~224 ms/层在装配之外、MoE 之内**(GPU MoE 内核 + 等待)。

### 19.2 决定性的一除法:2/3 的时间**不在 MoE 层里**

```
每层 MoE 总计 = 413 ms
42 层 × 413 ms = **17.3 s**
实测 TTFT      = **52.8 s**
⇒ **35.5 s(67%)根本不在 MoE 层内**
⇒ 35.5 / 42 = **~846 ms/层 是 attention + indexer(KDA sparse 路径)**
```

**⇒ `attention/indexer` 约为 MoE 的 2 倍。这就是「装配之外那 ~436 ms/层」的归属** ——
它既不是装配,也不是 MoE,而是**每层的注意力与稀疏 indexer**。

### 19.3 这条结论改变了优化方向

| 原以为 | 实际 |
|---|---|
| 瓶颈在 MoE 权重流式(H2D 135 ms/层) | H2D 只占 **413 ms 的 33%**;MoE 全部只占 TTFT 的 **33%** |
| 提升 MBT 就能接近 DMA 下界 | 提升 MBT 把 MoE 那 1/3 压小(310.6 tok/s 已拿到),但**剩下 2/3 动不了** |
| 上 NVFP4 能大幅提速 | 它只把 H2D 减半 ⇒ 只影响那 1/3 里的部分,**天花板有限** |

**⇒ 下一步该查的是 GLM 的 attention/indexer 路径**(`glm5next/nvidia/sparse_indexer.py`
与 `kpool_compress.py`),而不是继续压 MoE/H2D 或换量化格式。

### 19.4 待办

- [ ] 用 `XIAOTU_TORCH_PROFILE=<path>` 拿到**逐 kernel 表格**,证实 attention/indexer 的占比
      (本轮那次因请求失败未产出:TTFT 0.00,原因待查);
- [ ] 那条 `pre` 里 ~224 ms/层(GPU MoE 内核 + 等待)也要归属 —— 但优先级低于 19.2 的 2/3。

---

## 20. ④ 的直接观测:两次尝试都未产出 kernel 表(2026-09-21 13:25)

| 尝试 | 配置 | 结果 |
|---|---|---|
| `p4`(prof4.sh) | 默认图 + `TORCH_PROFILE_CALLS=200` + `LAYER_TIMING=1` | ❌ 请求到服务端但引擎死:`RuntimeError: cancelled` → `EngineDeadError`;**表格 0 行** |
| `p5`(prof5.sh) | **`EAGER=1`** + `CALLS=45` | ⚠️ 跑通了(TTFT **112,074 ms**),但 **trace 目录为空、表格 0 行、无报错** |

**两个副产物值得记下:**

1. **`EAGER=1` 代价 2.1×**:关掉 CUDA graph 后 TTFT **112,074 ms**,而开图是 **52,794 ms**。
   ⇒ 图对本工作负载是**真实且巨大**的收益(与 `TRIED_AND_REVERTED` R15 里"图对解码没用"相反 ——
   那条是**解码**,这里是**预填充**)。
2. **profiler 钩子没有触发**:`_maybe_profile()` 的调用点在 `hybrid_model.py:1301`。
   需查该行是否在**本配置实际走到的那条路径**上(它旁边还有 `_maybe_profile_decode()` 在 1369)。
   `_capture_guard()` 会正确地在捕获期退出(见 R101),但 `EAGER=1` 下不存在捕获,
   所以**不是**被 guard 挡掉的。

**⇒ ④ 的结论目前是算术推断,不是直接观测:**
```
每层 MoE 413 ms(实测,[layer-timing])× 42 层 = 17.3 s
TTFT 52.8 s(实测)                      ⇒ 差 35.5 s(67%)不在 MoE 层内
```
两个被减数都是实测,但"**差额归给 attention/indexer**"这一步尚未被 kernel 表证实。
**替代路径**:①查清 `hybrid_model.py:1301` 的调用条件;②上 `nsys`(环境里有 2023.1.2)。

### 20.1 🎯 profiler 打不出表的**真正原因:钩子只挂在 MXFP4 路径上**

`hybrid_model.py:1295-1301`:

```python
    w13, s13, w2, s2, ..., device=hidden_states.device, slot=slot,
)
if _nvtx: torch.cuda.nvtx.range_pop()
_maybe_profile()                       # ← 这里
if self.shared_experts is not None: ...
```

**这一段是 `gpu_moe_layer`(MXFP4 路径,`gpu_prefill.py`)的调用点。**
而 **GLM-5.3-Flash 走的是 FP8 路径**(`gpu_prefill_fp8.py`,经
`mixed_experts.py` 的 `_gp_mod.kmajor_from_engine_shards`) ⇒
**`_maybe_profile()` 对 GLM 永远不会被调用**,所以:

* trace 目录为空;
* 表格一行都没有;
* **而且不报错**(静默失效)。

**⇒ `XIAOTU_TORCH_PROFILE` 目前对 FP8 模型(含 GLM)是失效的。**

### 20.2 要拿到 kernel 表,只需一处两行的改动(下一步)

在 `mixed_experts.py` 的 **FP8 装配分支**末尾(即 `kmajor_from_engine_shards`
返回之后、MoE 内核完成之后)补一次 `_maybe_profile()` 调用,与 MXFP4 路径对齐。
`_maybe_profile` 已存在于 `hybrid_model.py:337`,可直接复用(注意它已有
`_capture_guard()`,捕获期会自动退出)。

**或在改代码之前先用 `nsys`**(环境里是 2023.1.2)——
`nsys` 不需要插件配合,但输出大、需 `nsys stats` 后处理。

**⇒ ④ 的直接观测目前卡在"钩子挂错路径"这一处,已有明确修法,不是死路。**

---

## 21. 调研:NVFP4 的 GPU 端反量化与「要不要自己写算子」(2026-09-21)

### 21.1 vLLM / SGLang **已经实现**了 GPU 端 NVFP4,且 SM80 有人在推

| 项目 | 支持 | 出处 |
|---|---|---|
| **vLLM** | ✅ `vllm.model_executor.kernels.linear.nvfp4`(cutlass `nvfp4_gemm` / marlin 版 / `.../nvfp4/pytorch` 兜底) | [vLLM API 文档](https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/pytorch/) |
| **vLLM × Ampere** | ✅ **进行中** | [PR #45306 "Support modelopt_mixed on Ampere (SM80/SM86)"](https://github.com/vllm-project/vllm/pull/45306) —— **与我们的处境完全对口** |
| **SGLang** | ✅ | [Quantization 文档](https://docs.sglang.io/docs/advanced_features/quantization) |
| **TileLang** | ✅ 专门模块 | [Quantization/Dequantization](https://deepwiki.com/tile-ai/tilelang/10.4-quantization-and-dequantization)、[`tilelang.contrib.cutedsl.quantize`](https://tilelang.com/autoapi/tilelang/contrib/cutedsl/quantize/index.html) |

**⇒ 不必从零手搓。可直接参考/抄,按适配度排序:**

1. **`marlin_utils_fp4`**(vLLM 内)—— **它就是 group-16 的 NVFP4**。
   ⚠️ **`gpu_prefill.py` 的注释反过来给了我们一条好消息**:它说
   「`marlin_utils_fp4` 只支持 **NVFP4 group-16**、不支持 MXFP4 block-32,所以 vLLM 的 Marlin MoE
   不能复用」—— **换成 NVFP4 之后,这条路就通了**,不必再自写 in-kernel dequant。
2. **`nvfp4/pytorch` 兜底实现** —— 读起来最快,可作 scale 布局语义参考
   (E2M1 nibble + group-16 e4m3 scale + **per-tensor fp32 global scale**)。
3. **CUTLASS NVFP4 示例** + **NVIDIA ModelOpt**(`nvidia/GLM-5.3-Flash-NVFP4` 即由 v0.47.0 产出)。
4. **TileLang quantize contrib**(若走 IR 路线)。

### 21.2 直接写 CUDA 还是用 TileLang?**先抄,不要先写**

* **第一步(几十行,不需要新算子)**:把 NVFP4 的**解包+缩放语义**移植进现有
  `moe_v2_packed4.hpp` / `gpu_prefill_fp8.py` 框架 —— 插件本来就是
  「in-kernel dequant 到 BF16 + BF16 GEMM」,换 scale 布局即可。
  **CPU 引擎已经支持 NVFP4**(`moe_v2.hpp:12`),只有 **GPU 路径**(写死 MXFP4 group-32 e8m0)要改。
* **第二步**:只有当 dequant **融合进 GEMM** 成为瓶颈时,才考虑写独立算子。
  那时 **TileLang 值得用**(已内建 quantize/dequant 原语,能把 dequant 融进 GEMM 的 IR,
  省掉手写 tile/寄存器分配);**但必须先确认它支持 SM70**(若将来落到 V100)。

### 21.3 ⚠️ SM80 上「反量化算子」的真实意义

**A100(SM80)既无 FP4 也无 FP8 张量核。** 所以 SM80 上的反量化算子
**只能是把 4-bit 解成 BF16/FP16 供普通张量核用** —— 与插件现在对 FP8 做的一样。
**不要指望 SM80 上有硬件 FP4 加速。**

### 21.4 顺带:`nvidia/GLM-5.3-Flash-NVFP4` 的两个待核实点

* 卡片写的量化范围是「**shared experts** and dense MLP」,而 recipe 名是
  `nvfp4_experts_dense_mlp` —— **两种读法冲突**。若被量化的**不是 routed experts**,
  则本项目唯一关心的 H2D 与显存**一分不省**。**必须看 `model.safetensors.index.json` 才能定。**
* 卡片声明硬件为 **Blackwell**(测试机 GB200),官方 vLLM 命令用 **TP=4 + arm64 容器**;
  **本机是 SM80、2 张可用 A100** ⇒ 不能照搬。

---

## 22. 调研:硬件降到 **V100** 的性能损失(2026-09-21)

### 22.1 先说最要紧的:**可能根本跑不起来**

V100 是 **SM70(Volta)**,而插件最低验证到 **SM80**
(文档记有「SM80 的必需回退:`TRITON_ATTN_DIFFKV`」这类适配)。**SM70 是否被支持需先查**;
若不支持,讨论性能损失没有意义。

### 22.2 若假设能跑,理论账(V100-SXM2-32GB vs A100-40GB)

| 维度 | A100-40GB | V100-SXM2 | 比值 |
|---|---|---|---|
| **BF16 张量核** | 312 TFLOPS | **无 BF16**(Volta 只有 FP16 TC,125 TFLOPS) | **~2.5×** |
| HBM 带宽 | 1,555 GB/s | 900 GB/s | **1.73×** |
| 显存 | 40 GB | 32 GB | **1.25×** |
| PCIe 锁页 H2D | 26.86 GB/s | ≈同(同为 PCIe Gen4 ×16) | **≈1×,不变** |
| NVLink | A100-PCIE 无 | V100-SXM2 有 300 GB/s | 对 EP 合并**有利** |

**按 §19 实测出的分段加权:**

```
每层 413 ms(MBT=16384,实测)
  H2D 装配      165 ms × 1.0  = 165     ← PCIe 限速,V100 上基本不变
  MoE 内核     ~224 ms × 2.0  = 448     ← 计算+带宽混合
非 MoE attention/indexer
               ~846 ms × 1.7  = 1438    ← 带宽为主
加权 ⇒ (165+448+1438)/(165+224+846) ≈ **1.63×**
⇒ **prefill 理论慢约 1.6–2 倍**
```

### 22.3 但两个更硬的天花板会先撞上

1. **显存 32 GB**:util 0.90 ⇒ 28.8 GiB 可用,而 **1M 的 KV 就要 19.05 GiB**,
   加 staging 7.59 = 26.6 GiB ⇒ **preflight 要的 11.4 GiB 空闲给不出来**
   ⇒ **GPU 预填充再次被禁用**(与 §11 同因),退回 CPU 路径 ~145 tok/s。
   **这一条比「慢 2 倍」严重得多。**
2. **无 BF16 张量核**:插件大量 BF16 in-kernel 反量化 + BF16 GEMM 的设计在 V100 上
   要整体改成 FP16 —— 这是**代码工作量**,不是性能数字。

**⇒ 判断:降到 V100 不是「损失 2 倍」,而是「要重做一遍 SM70 适配 + 大概率失去 GPU 预填充」。
若目标是保住 1M 上下文 + GPU 预填充,V100 属方向性倒退。**

---

## 23. ✅✅ ④ 结案:**逐 kernel 表已拿到,注意力核占 58.8%**(2026-09-21 13:42)

**关键修法**:上一次没打表的原因是 `XIAOTU_TORCH_PROFILE_CALLS=45` **大于实际调用次数**
(MBT=16384 ⇒ 1 chunk × 42 层),**profiler 永远到不了阈值 ⇒ 不退出 ⇒ 不打印**。
设成 **`CALLS=40`**(< 42)后立刻打出表格。**另**:FP8 路径原先**没有** `_maybe_profile()`
调用点(只挂在 MXFP4 路径,`hybrid_model.py:1301`) —— 已在
`mixed_experts.py` 的 FP8 分支补齐(见 §20.2,已提交)。

### 23.1 逐 kernel CUDA 时间占比(TTFT 55,965 ms)

```
_sparse_mla_fwd_with_sink_kernel   22.325 s  58.80%   (9 calls)   ← 稀疏 MLA 注意力
vllm::moe_forward_shared           11.628 s  30.28%   (39)
  Memcpy HtoD (Pageable -> Device)  6.157 s  16.22%   (546)      ← ⚠️ Pageable
  down_kernel_fp8                   2.879 s   7.58%   (39)
  gate_up_kernel_fp8                2.549 s   6.71%   (39)
  vllm::all_reduce                  2.084 s   5.49%   (78)
  _ktranspose_bytes_kernel          0.220 s   0.58%   (78)
其余全部 < 1%(aten::mm/bmm/linear、marlin_gemm、elementwise、mhc tilelang 等)
```

**⇒ 与 §19.2 的算术推断一致(我当时推「attention/indexer 约 2/3」,实测 58.8% + all_reduce 等)。**
**⇒ 装配的 GPU 转置确实很小**(0.58%),印证「H2D 是主要装配成本、转置不是」。

### 23.2 🆕 白捡的机会:**6.16 s 的 Pageable H2D**

`Memcpy HtoD (**Pageable** -> Device)` = **16.22%**,而插件自己的 engine host 分片
**已经锁页**(日志:`引擎 host 分片锁页 10 个缓冲(DMA 16-21 → 26.85 GB/s)`)。
⇒ 这 6.16 s **是另一批未锁页的传输**(cache 构建?draft?KV?),**锁页极可能直接省下大部分**。
**这是当前已知的最低风险、最高确定性收益点。**

---

## 24. 回答:换成 `nvidia/GLM-5.3-Flash-NVFP4` 在修完注意力之后**仍有较大收益吗?**

**答:没有。** 现在可以用 §23 的占比定量回答:

| NVFP4 能影响什么 | 占比 | 减半后省 |
|---|---|---|
| `down_kernel_fp8` + `gate_up_kernel_fp8`(权重带宽受限) | **14.29%** | ~**7%** |
| `Memcpy HtoD`(权重字节减半) | 16.22% | ~**8%** |
| **合计** | | **~15%** |
| **`_sparse_mla_fwd_with_sink_kernel`(注意力)** | **58.80%** | **0%** ← NVFP4 完全不碰 |

**⇒ 换 NVFP4 的天花板约 15%,而且**完全不动**那个 58.8% 的注意力核。
即使先修完注意力,剩下的 MoE 只占 30%,NVFP4 仍只能吃其中一半左右。**

**另外两条使收益更小的因素:**
1. **§21.4 的两个待核实点**(量化范围是否含 routed experts;第 45 层 MTP 是否在),
   若 routed experts 未被量化,收益**直接归零**;
2. **SM80 无 FP4 张量核** ⇒ 只能「解成 BF16 再算」,**算力不变**,省的只是访存。

**⇒ 结论:优先级应是 ①锁页那 6.16 s(确定、低风险)→ ②注意力核(58.8%,真瓶颈)→
③NVFP4(~15%,且要先核实权重范围)。**

---

## 25. 回答:如果有很多**廉价 V100**(如一台 8 块、免费)?

### 25.1 聚合资源其实**优于** 2×A100

| | 8×V100-SXM2-32GB | 2×A100-40GB | 比值 |
|---|---|---|---|
| FP16/BF16 张量算力 | 8 × 125 = **1000 TFLOPS** | 624 TFLOPS(BF16) | **1.6×** |
| HBM 带宽 | 8 × 900 = **7200 GB/s** | 3110 GB/s | **2.3×** |
| 显存 | **256 GB** | 80 GB | **3.2×** |
| NVLink | 有(300 GB/s/卡) | A100-PCIE **无** | 对 EP 合并**有利** |

**⇒ 单看聚合,8×V100 在算力、带宽、显存、互联四项上全面优于 2×A100。**

### 25.2 但三个硬阻塞

1. **SM70 支持未知**:插件最低验证到 **SM80**。**V100 = SM70**,是否被支持**必须先查**。
2. **`cp.async` 是 SM80+ 指令,Volta 没有**。而**瓶颈正是**
   `_sparse_mla_fwd_with_sink_kernel`(vLLM 的 sparse MLA)—— 这类核普遍依赖 `cp.async`
   做异步流水。**要为 Volta 重写注意力核,这是硬工作量。**
3. **无 BF16 张量核**:插件的 BF16 in-kernel 反量化 + BF16 GEMM 要整体改 FP16。

### 25.3 若这三个都解决,理论收益

* 瓶颈是注意力(58.8%),它**带宽受限**;V100 单卡带宽低 1.73×,
  但 **8 卡可做 TP/CP 分片 ⇒ 聚合带宽高 2.3×** ⇒ **这一项会变快**;
* MoE 的 H2D 走 **PCIe**,与卡数无关 ⇒ **不变**(且 §23.2 的锁页收益依旧适用);
* **⇒ 乐观估算:prefill 可比 2×A100 快 1.5–2 倍,但前提是把注意力核移植到 SM70。**
* **「免费」改变了约束性质**:成本不再是限制,**工程投入**才是。

**⇒ 结论:8×V100(免费)在聚合资源上是升级而非降级,值得评估;
但它的门槛是「注意力核能否移植到 Volta(cp.async 缺失)」,而不是算力或显存。**
**建议先做一件事:查 vLLM 的 `_sparse_mla_fwd_with_sink_kernel` 是否有 SM70 变体或回退路径。**

---

# 26. ✅ 最终结论(本 goal 收口,2026-09-21)

## 26.1 四项任务的证据

| 项 | 结论 | 直接证据 |
|---|---|---|
| **①** 逐步留痕 | ✅ | `/tmp/lp/<时间戳>/` 每轮独立目录:`00_env.txt`(旋钮实际生效值)、`manifest.txt`(完整命令+env-bridge 文件内容)、`verify.txt`(判据)、`gap.txt`(成绩)、`run.log`、`step_*.log` |
| **②** 硬判据确认走 GPU 流水线 | ✅ | **`[fp8-asm]` 行数**:MBT=4096→**588**、12288→**168**、16384→**84**(= MoE 层数 × chunk 数);`DISABLED` 计数;**并发现 `GPU prefill ACTIVE` 横幅在 slack 为负时照样打印 ⇒ 不能单独作证** |
| **③** MBT 扫描与边界 | ✅ | 154.5 → 266.5 → **310.6 tok/s**(4096/12288/16384);1M 侧 OOM + 被 preflight 禁用 |
| **④** 非装配时间的归属 | ✅ | **逐 kernel 表**:`_sparse_mla_fwd_with_sink_kernel` **58.80%**、`moe_forward_shared` 30.28%、`Memcpy HtoD (Pageable)` **16.22%**、装配转置仅 0.58% |

## 26.2 最终答案:16384 prompt(C=1)、GLM-5.3-Flash、2×A100-40GB

| 约束 | 最优配置 | **可达 prefill** | GPU 预填充 |
|---|---|---|---|
| **保证 1M 上下文** | `MAXLEN=1048576`、KV 19.05 GiB/rank、`SPEC_K=0`、0 层常驻 | **145.4 tok/s**(TTFT 112,787 ms) | ❌ **被 preflight 禁用** |
| **放开到 32K**(KV 0.75 GiB、util 0.85) | **`MBT=16384`** | **310.6 tok/s**(TTFT 52,794 ms) | ✅ 真跑(`[fp8-asm]=84`) |

**⇒ 「1M 上下文」与「GPU 预填充」在 2×A100-40GB 上不可兼得**,原因三重锁死:
1M 的 KV 实需 **19.05 GiB** ⇒ 只剩 16.5 GiB,而预检要 **11.4 GiB** 空闲(只有 4.5);
且 MBT=16384 需 ~38.7 GiB 非 KV ⇒ OOM。

## 26.3 性能瓶颈的真相(逐 kernel 实测)

```
_sparse_mla_fwd_with_sink_kernel   58.80%   ← 稀疏 MLA 注意力,真瓶颈
moe_forward_shared                 30.28%
  Memcpy HtoD (Pageable -> Device) 16.22%   ← ⚠️ 未锁页,最低风险的收益点
  down_kernel_fp8 + gate_up_fp8    14.29%
  all_reduce                        5.49%
```

**⇒ 「装配之外那 ~436 ms/层」= 注意力与稀疏 indexer,不是 MoE、不是装配、也不是 H2D。**

## 26.4 优化优先级(按确定性×收益)

1. **锁页那 6.16 s 的 Pageable H2D**(16.22%)—— 确定、低风险、与模型/硬件无关;
2. **注意力核**(58.8%)—— 真瓶颈,但改造成本高(涉及 KDA/sparse MLA 路径);
3. **NVFP4**(~15%,且需先核实 routed experts 是否被量化)——— **不是杠杆**;
4. **8×V100(免费)** —— 聚合算力 1.6×、带宽 2.3×、显存 3.2×,**是升级而非降级**,
   但门槛是 `cp.async`(SM80+ 指令,Volta 无)能否为 Volta 重写那个注意力核。

## 26.5 本 goal 期间我犯过并已更正的错误(留档)

1. **误判「没走 GPU 预填充」** —— 实为我的 `verify()` 跑在 bench **之前**(ACTIVE/[fp8-asm] 都是请求到达时才打印);
2. **误判「chunk 被 KDA 2176 钉死、MBT 无效」** —— 实为从行数反推层数时假设错误,MBT 有效(294→41);
3. **`FAKE_ALL` 消融把服务搞死** —— 输出置零破坏下游,该开关不能作消融;
4. **profiler 两次空手** —— 真因是 FP8 路径**没有** `_maybe_profile()` 调用点,且 `CALLS` 设得大于调用次数;
5. **拿 RedHatAI 的 NVFP4 顶替用户指定的 nvidia/ 那个** —— 答错对象。
