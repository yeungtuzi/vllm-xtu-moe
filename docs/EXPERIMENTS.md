# 实验台账（EXPERIMENTS）

> **为什么有这个文件**：2026-09-21 的会话里我开了 10+ 个后台实验，靠 `grep` 各路 `.out`
> 文件来回忆状态，结果**丢了两次收尾**（`attn_long` 的失败臂、`after_sweep_c` 整整两小时）。
> 根因是**流程缺失**：没有一份"已排队 / 已出结果 / 未收"的清单。
>
> **纪律**：
> 1. **开一个实验就在这里加一行**（写清命令、目的、判据），状态置 `RUNNING`；
> 2. **拿到结果立刻更新这一行**（写数字，不只是"完成"）；
> 3. **每轮结束前扫一遍本表**，把 `RUNNING` 的核对一遍 —— 这就是防丢失的机制本身；
> 4. 结论不确定就写 `⚠️` 并注明为什么，不要留空。
>
> 状态取值：`RUNNING` / `✅` / `❌` / `⚠️`（有结果但不可信或部分）

---

## A. 当前在跑

| # | 实验 | 目的 | 判据 | 状态 |
|---|---|---|---|---|
| A1 | `/tmp/attn_long.sh` (L=32768, 65536) | 长上下文下注意力占比曲线 | `[xiaotu-profile]` 的核时间 | ⚠️ **32768 臂 `TTFT=0.00` 失败**；65536 臂未出。**已无价值**（同配置连败），建议杀掉 |

---

## B. 已完成（本会话全部硬数据）

### B1 12 格 random 基准（`bench_random_12cells.sh`）

| 模型 | L | C | prefill | decode | 状态 |
|---|---|---|---|---|---|
| GLM-5.3-Flash | 128 | 1 | 115.2 | 22.0 | ✅ |
| GLM-5.3-Flash | 128 | 2 | 70.4 | 13.9 | ✅ |
| GLM-5.3-Flash | 16384 | 1 | 155.3 | 21.5 | ✅ |
| GLM-5.3-Flash | 16384 | 2 | 119.5 | 1.7 | ✅ |
| MiMo-V2.5 | 128 | 1 / 2 | 64.2 / 48.1 | 16.7 / 9.6 | ✅ |
| MiMo-V2.5 | 16384 | 1 / 2 | — | — | ❌ KV cap 过宽 OOM（后修为 1.10× 余量） |
| DeepSeek-V4.1 | 128 | 1 / 2 | 254.3 / 175.2 | 19.8 / 14.0 | ✅ |
| DeepSeek-V4.1 | 16384 | 1 / 2 | — | — | ❌ `aten::new_empty` 分配失败（未修） |

### B2 MBT 曲线（`MAXLEN=32768`, KV 0.75 GiB, util 0.85, 单条 16384, C=1）

| MBT | TTFT | **prefill** | `[fp8-asm]` | 状态 |
|---|---|---|---|---|
| 4096（KV 2 GiB, util 0.82） | 106,156 ms | 154.5 | 588 | ✅ |
| **8192** | **70,392 ms** | **232.9** | 252 | ✅（早先那次 `✗起不来` 是**暂态**，已证） |
| 12288 | 61,532 ms | 266.5 | 168 | ✅ |
| **16384** | **52,794 ms** | **310.6** | 84 | ✅ **最优** |
| 32768 | — | — | — | ❌ util 0.82 下 OOM |

**⇒ 关键洞察**：MBT=16384 只差 **120 MiB** 就能放下，而 KV cap 白占了 1.25 GiB
（16384 prompt 只需 0.30 GiB）⇒ **收紧 cap 到 0.75 GiB 即成立**。

### B3 1M 上下文（三重锁死，已实测）

| 配置 | 结果 | 状态 |
|---|---|---|
| 1M + MBT=16384, util 0.90 | ✗ 起不来，CUDA OOM | ✅ |
| 1M + MBT=8192, res 3.0 | ✗ 起不来，CUDA OOM | ✅ |
| 1M + MBT=4096, util 0.90 | TTFT 112,787 ms，prefill **145.4**，`[fp8-asm]=0`，**DISABLED=2** | ✅ |
| 1M + res 1.0（两档 util） | ✗ **服务崩**（不是变快） | ✅ |

**三重锁死**：1M 的 KV 实需 **19.05 GiB**（引擎反算 19,505 B/token）⇒ 只剩 16.5 GiB；
预检要 **11.4 GiB** 而只有 **4.5**；MBT=16384 需 ~38.7 GiB 非 KV ⇒ OOM。

### B4 消融（回答"那 ~46 s 非装配时间是什么"）

| 手段 | 结果 | 状态 |
|---|---|---|
| `XIAOTU_MOE_FAKE_ALL=1` | ❌ 无效 —— 请求根本没到服务端（0 个 POST，启动期 7 处 NaN）。**该开关不能用作消融** | ✅ 已查明 |
| `XIAOTU_LAYER_TIMING=1` | ✅ `pre=336–389 ms` / `eng=24–75 ms` / `post=0.04 ms`，每层 413 ms | ✅ |
| **逐 kernel 表**（`XIAOTU_TORCH_PROFILE`，`CALLS=40`） | ✅ 见下 | ✅ |

**逐 kernel CUDA 占比（L=16384, TTFT 55,965 ms）：**

```
_sparse_mla_fwd_with_sink_kernel   22.325 s  58.80%   ← 真瓶颈
vllm::moe_forward_shared           11.628 s  30.28%
  Memcpy HtoD (Pageable -> Device)  6.157 s  16.22%   ← ⚠️ 未锁页,最低风险的收益点
  down_kernel_fp8 + gate_up_fp8     5.428 s  14.29%
  vllm::all_reduce                  2.084 s   5.49%
  _ktranspose_bytes_kernel          0.220 s   0.58%   ← 装配转置确实很小
```

**⇒ 42 层 × 413 ms = 17.3 s ≪ TTFT 52.8 s ⇒ 67% 不在 MoE 层内 ⇒ 归给注意力/indexer。**

### B5 注意力核标度律（`MBT=L`，各 1 chunk）

| L | 注意力核 | TTFT | 占比 |
|---|---|---|---|
| 4,096 | 3.129 s | 28,673 ms | 10.9% |
| 8,192 | 9.538 s | 38,501 ms | 24.8% |
| 16,384 | 22.323 s | 55,954 ms | 39.9% |

**⇒ 拟合指数 `O(L^1.42)`。** 对照：稀疏 `Σmin(pos,topk)` 预测 5.0×（4× L），
稠密 L² 预测 16×，**实测 7.13× ⇒ 远更接近稀疏 ⇒ 稀疏确实在限制工作量，问题在单位效率**
（0.71 ns/key-attend）。

**⚠️ 不完整**：我先前把「占比随 L 上升」讲成纯趋势，**但那三行里藏着一个约 25 s 的固定项**
（`TTFT − 注意力`：25.5 / 29.0 / 33.6 s，L 增 4× 只增 1.32×）。
**占比上升有一部分只是分母里那个固定项没跟着涨。** 该固定项尚未解释。

### B6 旁路流重叠恢复（arm C，旁路流重开）

| 配置 | TTFT | 装配 `total` | illegal |
|---|---|---|---|
| 旁路流**关**（现默认，我的修复） | **106,156 ms** | 163.5 ms/层 | 0 |
| 旁路流**开**（重叠恢复） | **90,161 ms** | 145.8 ms/层（`tr` 28.5→**9.6 ms**） | 0 |

**⇒ 收益 1.18×（省 15%），不是我在 §10 预测的 1.8×。** 我的修复是「用 15% 换正确性」，
正确做法仍是**保留重叠 + 修竞态**。单条长请求下旁路流安全（崩溃需"两格短请求之后"）。

### B7 旋钮探针（1M 下逐个试腾显存）

| 档 | util | `GP_ACT_RESERVE_GIB` | TTFT | 判定 |
|---|---|---|---|---|
| k_res30 | 0.90 | 3.0 | 112,399 ms | ❌ 仍走 CPU |
| k_res10 | 0.90 | 1.0 | **0.00 ms** | ❌ **崩** |
| k_res00 | 0.90 | 0.0 | 112,761 ms | ❌ 仍走 CPU |
| k_u085r10 | 0.85 | 1.0 | **0.00 ms** | ❌ **崩** |

**⇒ 4/4 失败**。reserve=0 也被禁用（staging 8.35 GiB > 4.5 GiB 空闲）；reserve=1.0 崩，
印证代码注释「spending that peak makes a long prefill OOM **after** a passing preflight」。

### B8 NVFP4 调研（`nvidia/GLM-5.3-Flash-NVFP4`）

routed experts **确实是 NVFP4 W4A4**（group 16 + E4M3 block scale + FP32 per-tensor global scale）；
**model card 措辞错误**（写 shared experts，实为 routed）；shared experts 仍 BF16；
**第 45 层 MTP 在，但整层 BF16**；磁盘 204.44 GB，routed experts 171.23 GB NVFP4。
**⇒ 流式引擎的专家集合 304.4 GB(FP8) → 171.2 GB，1.78×。**
**但优先级低**：它只影响 H2D 与 MoE GEMM（合计约 30%），**完全不碰那 58.8% 的注意力**。

### B9 V100 / SM70 调研（已按指示搁置）

**跑不了**：`FlashMLASparseSM8XBackend.supports_compute_capability` → `capability.major == 8`；
bf16 在 cc<8.0 硬报错。**更正我 §25.2**：`cp.async` **不是**真阻塞点（路由与 bf16 是），
且 `TRITON_ATTN_DIFFKV` 是 **MiMo-V2.x** 的回退，不适用于 GLM 的稀疏 MLA。
产出 `docs/SM70_VOLTA_VERDICT.md`、`docs/SM70_VOLTA_FORK_PLAN.md`。

---

## C. 上报上游

| # | 内容 | 状态 |
|---|---|---|
| C1 | [vllm-project/vllm#57971](https://github.com/vllm-project/vllm/issues/57971) —— SM8x sparse-MLA prefill 回退核：**KV 按 head 重复读（64× 冗余）** + **~17× 带宽低效** | ✅ OPEN，作者 yeungtuzi |

**证据**：grid `(num_tokens, active_heads)`，而 `kv` 的寻址**不含 `head_idx`**
⇒ 64 个 head 各自重读同一份 latent。算法需 32.2 GB，实际读 2062 GB。
即使按这 64× 冗余算，HBM 下界 1,326 ms vs 实测 22,323 ms ⇒ 有效带宽 **92 GB/s**（峰值 6%）。

---

## D. 待做（按优先级）

| # | 任务 | 依据 | 状态 |
|---|---|---|---|
| **D1** | **pr4-A1'**：head 分块，`BLOCK_H=4`，**保留逐元素数学** ⇒ **bit 级 A/B**，隔离「KV 复用」收益 | `patches/upstream/pr4_body.md` | 待开始 |
| **D2** | **pr4-A2**：`BLOCK_H=16` + `tl.dot` ⇒ KV 流量 /16 + 张量核 | 同上 | 待 D1 |
| **D3** | **锁页那 6.16 s 的 Pageable H2D**（16.22%），与模型/硬件无关、风险最低 | B4 | 待做 |
| **D4** | 查清那个 **~25 s 固定项**（B5 的 ⚠️） | B5 | 待做 |
| **D5** | 修 DeepSeek-V4.1 长 prompt 的 `aten::new_empty` 分配失败 | B1 | 待做 |
| **D6** | MiMo/V4.1 长格补测（KV cap 已修为 1.10× 余量） | B1 | 待做 |

**A1' 的设计约束（已核实，别再走回头路）**：
- 逐元素 + head 分块会产生 `(H, K, D)` 三维中间张量 ⇒ `BLOCK_H=16` 时 512 KB，**寄存器放不下**；
  只有 `BLOCK_H=2~4` 可行（64–128 KB）。
- `tl.dot` 要求 `M ≥ 16` ⇒ **A1'（bit 级）与 A2（张量核）不能同时满足**，必须分两步。
- `BLOCK_K=16` 是 PV 那个 `tl.dot` 的规约维，**恰好是 Triton 下限，没有余量**。
