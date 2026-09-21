# GLM-5.3-Flash prefill 优化：已知问题与排查结论

> 目的：把 2026-09-21 这一轮 prefill 优化里**修好的、证伪的、以及仍未解决的**问题一次说清，
> 使后来者不必重走。**每条都标注强度：✅实测 / ⚠️推断 / ❌已证伪。**
>
> 完整逐轮记录见 `docs/EXPERIMENTS.md`（台账，B1–B22）；补丁见
> `patches/upstream/pr4-*.patch` + `pr4_body.md`。

---

## 1. 性能点总表（profile 口径，单条 16384 prompt / C=1 / TP=2 / A100×2）

| # | 项 | 原占比 | 现状 | 强度 |
|---|---|---|---|---|
| 1 | `_sparse_mla_fwd_with_sink_kernel`（稀疏 MLA 注意力） | **22.3 s / 58.8%** | **7.739 s / 33.15%** | ✅ 已修（端到端 TTFT **1.52×**） |
| 2 | `Memcpy HtoD`（专家权重 DMA） | 6.07 s / 16.5% | 不变 | ❌ **已证伪为可优化项**（见 §3.1） |
| 3 | FP8 MoE GEMM（`gate_up` + `down`） | 5.45 s / 14.8% | 待 A14 扫描 | ✅ 实测低效（见 §2.1），⚠️ 未归因 |
| 4 | `ncclDevKernel_AllReduce` | 2.107 s / 5.7% | 待 A13 | ⚠️ 净成本未定 |
| 5 | 未归属缺口 | 上界 ≤13.7 s | — | ⚠️ **存在性未知**（见 §4.1） |
| 6 | `_ktranspose_bytes_kernel` | 0.22 s / 0.6% | — | ✅ 已测，**很小，不值得动** |

---

## 2. 未解决项（按价值排序）

### 2.1 ⭐ FP8 MoE GEMM 只有 **0.61 TFLOPS（A100 BF16 峰值的 0.19%）**

**实测**：单条 16384 prefill 的专家 GEMM ≈ 3.30 TFLOP（推断），实测 5.45 s
⇒ **0.61 TFLOPS**；按 312 TFLOPS 峰值理论 10.6 ms，**慢 516 倍**。
**即便 FLOPs 推算打对折，达成率仍是个位数百分比。**

**已知的调优面（原先完全硬编码，现已可覆盖）**：
```python
# vllm_xiaotu_moe/gpu_prefill_fp8.py
XIAOTU_GPF_GEMM_{BM,BN,BK,BH,STAGES,WARPS}   默认 64/64/64/64/2/4
grid: gate_up_kernel_fp8[(E=288, cdiv(2I, bn))] / down_kernel_fp8[(E=288, cdiv(H, bh))]
```

**四个未验证的候选原因**：① 64 宽 tile 算术强度低；② **grid 以 `E=288` 为第一维**，
每 program 需遍历该专家的 token（局部性差）；③ `num_stages=2` 重叠不足；
④ in-kernel FP8→BF16 反量化的开销。

**⇒ 停止判据（A14 已按此设计）**：若 `STAGES=4/WARPS=8` 与 `BM=BN=128` 两档对 GEMM 时间
**都不敏感**，则瓶颈不在 tile，而在 grid 布局或反量化 —— **应在该项上收手，不再投入。**

### 2.2 ⚠️ `all_reduce` 的净成本未定

profile 记 78 次调用、2.107 s CUDA。**78 对 42 层这个比值我至今没有解释。**
已确认两个调用点在**互斥路径**上（GPU 预填充 `hybrid_model.py:1288` / CPU 长 qlen 回退 `:1399`），
**不是"每层归约两次"**（这是我最初的假设，已否）。
`XIAOTU_SKIP_AR=1`（源码自带的计时诊断开关）可量其净成本。

### 2.3 pr4 补丁的三处未结项

| 项 | 状态 |
|---|---|
| **端到端数值未验证** | 核级 A/B 已做（`max|diff| = 2.819e-05`，规约顺序，**非 bit 级**）；**端到端是否导致输出变化未测**。6 次尝试均因脚本 bug 失败（见 `pr4_body.md`）。 |
| **`head_mask` 分支从未执行** | wrapper 会给 `num_heads` 之后的 head 补零，新核改用 `head_mask` 写；所有测试的 H 都是 `BLOCK_H` 的整数倍，GLM 的 64 又能被 8 整除 ⇒ **该路径是未测代码**。 |
| **布局转换的解释未证实** | 二分量出「2.8× 全部来自二维 `q` 载入」（✅ 实测），但**为什么**仍是推断（一维广播导致 layout 转换/失去向量化）。`TRITON_KERNEL_DUMP` 两次都没抓到对照；未试的路径是 `device_caches` / `triton.compile` 取 `asm`。**在此之前不得当事实陈述。** |

---

## 3. 已证伪的假设（**不要再试**）

### 3.1 ❌ 「那 6.07 s 的 `Memcpy HtoD (Pageable)` 是未锁页传输，锁页即可省」

`[pin] host 分片锁页: 10/10 个缓冲成功` ⇒ 锁页**确实生效**；profile 里**没有任何 Pinned 条目**
⇒ `Pageable` 是 profiler 对插件 C++ 里 `cudaMemcpy2DAsync` 的**误分类**。

算术定案：42 层 × 3.38 GiB = 141.8 GiB；@26.85 GB/s（实测锁页线速）= **5.67 s** vs 实测 **6.072 s** ✓；
@16 GB/s（pageable）会是 9.51 s ✗。**⇒ 已经在锁页线速上，没有 pageable 可修。**

反证已做：`XIAOTU_GPF_PIN=0` ⇒ TTFT 从 **36,894 → 55,968 ms**。
**⇒ 锁页本身值 1.52×，而且早就在生效。**
⚠️ 但它带来 **+19.1 s** 而按速率差只该 +3.2 s —— **差额原因未查明**。

**⇒ H2D 的唯一杠杆是「减少字节数」（如 NVFP4 让专家权重小 1.78×），不是"修锁页"。**

### 3.2 ❌ 「`XIAOTU_GPF_ACT_RESERVE_GIB` 调小可以换 reach」（4 档实测全败）

`reserve` 3.0 / 0.0 ⇒ 仍被禁用（staging 7.59×1.10 = 8.35 GiB 本身就超过空闲）；
`reserve` 1.0 ⇒ **服务直接 OOM 崩**。**调它只是把 OOM 从 preflight 推迟到预填充。**

### 3.3 ❌ 「MBT 放大拿不到收益 / chunk 被 KDA 2176 钉死」

实测 MBT 4096→16384 让装配行数 **588 → 84（7.14×）**，prefill **154.5 → 310.6 tok/s（2.01×）**。
**MBT 是有效杠杆**；我先前从行数反推层数得出的「chunk 由 block_size 决定」是错的。

### 3.4 ❌ 「1M 上下文可以与 GPU 预填充共存」

三重锁死：1M 的 KV 实需 **19.05 GiB**；预检要 **11.4 GiB** 空闲而只有 **4.5**；
MBT=16384 需 ~38.7 GiB 非 KV ⇒ OOM。**任何 MBT / reserve 都救不回来。**
（**256K 只需 4.8 GiB KV，远在预算内** —— 这是新目标选 256K 的原因。）

### 3.5 ❌ 「`XIAOTU_MOE_FAKE_ALL=1` 可用作消融」

请求**根本没到服务端**（0 个 POST，启动期 7 处 NaN）——输出置零破坏了下游数学。

---

## 4. 测量方法上的坑（会让数字骗你）

### 4.1 ⚠️ profiler 的父子行**重复计数**

`Self CUDA` 对 Python 层级 op 而言**包含其子内核**：
`moe_forward_shared` 11.45 s ≈ `down_fp8` 2.88 + `gate_up_fp8` 2.57 + `Memcpy HtoD` 6.07 = 11.52；
`all_reduce` 2.11 = `ncclDevKernel` 2.11；`unified_mla_attention` 7.74 = 稀疏核 7.74。
**⇒ 直接求和会得到 48.34 s > 墙钟 36.86 s（负数差额）。必须排除容器行。**

**而且**：各核**异步重叠**，所以「核求和」**不是墙钟**，两者不能直接相减。
`[layer-timing]` 与 profile **也可能重叠** ⇒ 我据此推出的「15.7 s 缺口」**已撤回**（B17→B18）。

### 4.2 ⚠️ `GPU prefill ACTIVE` 横幅**会说谎**

它在**每请求 preflight 失败（slack 为负）时照样打印**。
**可靠判据只有两个**：`XIAOTU_GPF_STAGE=1` 下 **`[fp8-asm]` 行数 > 0**，或 **`DISABLED` 未出现**。

### 4.3 ⚠️ 清场必须杀 `VLLM::EngineCore` 子进程

`serve_glm53_mainline.sh` 的 pidfile 里是**启动器**；真正的 EngineCore **不在任何 pidfile 里**。
只杀脚本 + pidfile + `nvidia-smi` 列出的 pid **会漏掉它**（实测残留 2×39,248 MiB 逾 6 分钟）。
**验收判据：`nvidia-smi` 读到 0，而不是 kill 命令返回成功。**

### 4.4 ⚠️ 进程纪律（本会话踩了 4 次）

**绝不用 `pgrep -f` / `pkill -f` 匹配任何可能出现在自己命令行里的字符串**（进程名、tag、模型名、路径都会）。
用括号技巧 `pgrep -f '[V]LLM::EngineCore'` 打断自匹配；
**`kill -9 -PGID` 尤其危险**（会波及自己的进程组）—— 除非确认 PGID 不是自己的，否则不要用。

---

## 5. 有效的做法（值得保留）

1. **独立测试台做数值 A/B**（`/tmp/kern_ab.py`）：构造随机输入直接调两个核，
   **不需要加载模型**（省掉 6 分钟/次），且能精确控制形状与边界情形；
2. **把可调参数做成环境变量**，让扫描从「改源码 → 跑 → 回滚」变成「设环境变量 → 跑」
   —— 前者正是本会话多数事故的来源（含一次把生产树留在修改状态）；
3. **改树前先记 md5、改完必比对**；三段式操作要么改树的副本，要么把回滚做成 `trap`；
4. **先找现成开关/先例，再考虑动代码**：`XIAOTU_SKIP_AR`（量 all-reduce）、
   `XIAOTU_LAYER_TIMING`、`XIAOTU_TORCH_PROFILE`、`_PREFILL_INDEXED_HEAD_BLOCK = 8`
   （head 分块在本仓库已有生产先例，被 DSv4/V4.1 使用）都是查出来的，不是我写出来的。

---

## 6. 上游关系澄清

`vllm/v1/attention/backends/mla/sparse_mla_kernels.py` 与 `flashmla_sparse_sm8x.py`
**不在 vLLM main 上** —— 它们由本仓库的 `patches/upstream/pr3-sm80-port.patch` 新增
（`new file mode 100644`，3517 行）。
**⇒ 本文件里所有关于这两个核的数字，测的都是我们自己的 SM80 移植代码，不是上游代码。**
（我据此误报过上游 [vllm#57971](https://github.com/vllm-project/vllm/issues/57971)，**已撤回并关闭**。）
