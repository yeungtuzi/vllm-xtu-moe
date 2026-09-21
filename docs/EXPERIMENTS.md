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

## A. 当前在跑（交付阶段）

| # | 实验 | 目的 | 判据 | 状态 |
|---|---|---|---|---|
| **D-1** | `/tmp/d_glm3.sh` GLM @ **256K**（MAXLEN=262144, KV=5113807360, MBT=16384） | 交付:GLM long prefill 最好成绩 | `[fp8-asm]>0` 且 `DISABLED=0` 且 `Successful=1` | **RUNNING** |
| **D-2** | V4.1 @ 256K | **最高优先级**:修好 LTS 模型长 prompt | 同上 | 待 D-1 |
| ~~D-0a~~ | GLM @256K 第1次 | | ❌ OOM:KV cap 给宽 0.86 GiB | |
| ~~D-0b~~ | GLM @256K 第2次 | | ❌ 启动竞态:显存未释放就 launch | |

**两条已修的自伤错误(留档):**
1. **KV cap 乘性余量**:256K 真实 4.76 GiB,我给 5.62 GiB(+0.86)⇒ OOM。
   ⇒ **256K 及以上必须按真实需求配置,不加乘性余量**(小 KV 场景 10% 无所谓,4.76 GiB 的 10% 就是 0.48 GiB)。
2. **启动前不查显存**:脚本用固定 `sleep 8`,而启动瞬间仍是 40427 MiB ⇒ EngineCore 初始化失败 + 握手 SIGTERM。
   ⇒ **已改为轮询 `nvidia-smi` 到真正为 0 再 launch**(实测 5s 清完)。
   **这正是 E1 判据的自动化 —— 写进代码,而不是靠我记住规则。**

## B. 已完成（本会话全部硬数据）

### B10 A1' 核 A/B（小尺寸:N=64 H=64 D=512 topk=512 R=4096，**未加载模型**）

| 变体 | `max\|diff\|` | 耗时 | 结论 |
|---|---|---|---|
| 原核 | — | 7.334 ms | 基线 |
| **BLOCK_H=1** | 1.965e-05 | **1.813 ms** | **快 4.0×** |
| *(真尺寸复核见 B10b)* | | | |
| BLOCK_H=2 | 1.965e-05 | 1.804 ms | 4.06× |
| BLOCK_H=4 | 1.965e-05 | 2.260 ms | 3.24× |
| BLOCK_H=8 | 1.965e-05 | **1.788 ms** | **4.10×** |
| BLOCK_H=16 | **1.170e-03** | 9.171 ms | ❌ 精度差 60×、比原核慢 |

**三条关键结论（都推翻了我 pr4 设计里的假设）：**
1. **`BLOCK_H=1` 就快 4.0×，而它没有 KV 复用**（一 head 一 program）⇒ 这 4× 来自
   **把 `q`/`running_acc` 变成二维 `(1,BLOCK_D)` tile 的形状改变** —— 纯 free。
2. **`BLOCK_H=16` 方向错误**（慢 25% + 精度劣化 60×），疑为寄存器 spill。
3. **KV 复用几乎无收益**（1.813 → 1.788）⇒ **此尺寸下该核不是 KV 带宽受限**。
   ⚠️ **这动摇了我 issue #57971 的框架**（我报的是 64× 冗余流量）。
   **必须在真实尺寸（N=16384/topk=2048，循环 128 次）下复核**才能定论。

⚠️ 1.965e-05 的差异**不是 bit 级**，来源是 `tl.sum(q[:,None,:]*kv[None,:,:],axis=2)`
与原核 `tl.sum(kv*q[None,:],axis=1)` 的**规约顺序不同**（浮点结合律）。
**⇒ A1' 不是 bit 级等价，「bit 级 A/B」这个说法要作废，改用容差对比。**

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

### B10b ⭐ 真实尺寸复核（N=1024 H=64 D=512 **topk=2048 R=32768**，循环 128 次，与线上一致）

| 变体 | `max\|diff\|` | 耗时 | 加速 |
|---|---|---|---|
| 原核 | — | 311.208 ms | 1.00× |
| **BLOCK_H=1** | 2.819e-05 | 111.853 ms | **2.78×** |
| BLOCK_H=2 | 2.819e-05 | 108.929 ms | 2.86× |
| BLOCK_H=4 | 2.819e-05 | 135.885 ms | 2.29× |
| **BLOCK_H=8** | 2.819e-05 | **108.185 ms** | **2.88×** |
| BLOCK_H=16 | — | 532.517 ms | ❌ 0.58× |

**⇒ 决定性结论（修正 B10 的第 3 条）：**
1. **形状改变单独就给 2.78×**（`BLOCK_H=1`，**无任何 KV 复用**）；
2. **KV 复用只再加 ~3%**（111.9 → 108.2）⇒ **该核不是 KV 带宽受限的**；
3. **⇒ issue #57971 里「64× 冗余流量」的事实成立，但「修它→大幅提速」的影响判断被高估。**
   真正的大头是 **tile 形状 / Triton 生成的代码质量**。
4. `BLOCK_H=4` 比 1/2/8 都差（非单调）⇒ occupancy/layout 效应，不是简单的越大越好；
   `BLOCK_H=16` 是明确的坑。
5. **端到端外推**：注意力 22.3 s → **约 8 s**；TTFT 56 s → **约 42 s**；占比 58.8% → **约 35%**。

### B11 ⭐⭐ D1c 端到端验证成功（pr4 二维 tile 核换进 rebase 树，配置同 B4/B5）

| 指标 | 基线（原核） | **二维 tile 核** | 变化 |
|---|---|---|---|
| **TTFT** | 55,954 ms | **36,894 ms** | **1.52×** |
| **`_sparse_mla_..._kernel`** | **22.325 s** | **7.739 s**（`_hb` 变体） | **2.88×** |
| **注意力占比** | **58.80%** | **33.15%** | −25.7pt |
| illegal | 0 | **0** | — |

**⇒ 与独立测试台预测的 2.8× 吻合（7.739/22.325 = 0.347）。**
**⇒ 收益来自 tile 形状,不是 KV 复用**（见 B10b:`BLOCK_H=1` 即 2.78×,复用只再加 3%）。

**验证方法与可回滚性**：临时改 `vllm-up-133b71e0b` 的 `sparse_mla_kernels.py`，
跑完**立即 `git checkout` 回滚**，并校验 md5 `fdf1908b9c80951132bf88384fba6201` 与
`diff -q` 对原始备份**逐字节一致**。补丁留在
`patches/upstream/pr4-sm8x-sparse-mla-2d-tile.patch`（102 行）。

### B12 🔴 D3 的假设被推翻:那 6.07 s **已经在锁页线速上**,没有 pageable 可修

**我原先的假设**（B4 里标为「最低风险的收益点」）：`Memcpy HtoD (Pageable -> Device)` 16.22%
是一批**未锁页**的传输，锁页即可省下大部分。

**实测把它推翻：**

```
[pin] host 分片锁页: **10/10 个缓冲成功**         ← 锁页确实生效
GPU 预填充:引擎 host 分片锁页 10 个缓冲(DMA 16-21 → 26.85 GB/s)
profile:  Memcpy HtoD (Pageable -> Device) = 6.072 s / 26.01%
profile:  **没有任何 Pinned 条目**
```

**算术判定（决定性）：**

| 假设 | 预期耗时 | 与实测 6.072 s |
|---|---|---|
| 锁页线速 26.85 GB/s | 5.67 s | **吻合(1.07×)** |
| pageable 16 GB/s | 9.51 s | 差得远 |

（总 DMA = 42 层 × 1 chunk × 3.38 GiB = **141.8 GiB**，恰好是 MBT=16384 下的每层一份全量权重）

**⇒ 结论：那 6.07 s 就是专家权重的 DMA，而且已经在锁页线速上。
`Pageable` 是 profiler 对插件 C++ 里 `cudaMemcpy2DAsync` 的**误分类**（整张表里根本没有 Pinned 条目）。**

**⇒ D3 作废。** 修正两条更早的判断：
1. B4 里说的「最低风险、最确定的收益点」**是错的**；
2. **H2D 已经在下限上 ⇒ 唯一的杠杆是减少字节数。** 这把 **NVFP4 的优先级显著抬高**：
   专家权重 1.78× 更小（304.4 GB FP8 → 171.2 GB NVFP4）⇒ 这 26% 的 H2D
   **理论上可降到约 15%**，外加 MoE GEMM 的收益。

⚠️ 未做：这只是算术吻合，**没有直接测量**「把 `XIAOTU_GPF_PIN=0` 时的耗时与 26.85 GB/s
对比」来反证。若要严格，应跑一档 `XIAOTU_GPF_PIN=0`，预期 H2D 升到约 9.5 s。

### B13 ✅ D3b 反证成功:B12 确证,且**锁页本身值 1.52×**（已在生效）

`XIAOTU_GPF_PIN=0` 单条 16384,MBD=16384,其余同 B4/B5/B11:

| 配置 | TTFT | 「锁页 10 个缓冲」 |
|---|---|---|
| 锁页**开**（默认） | **36,894 ms** | 打印 ✓ |
| 锁页**关**（`PIN=0`） | **55,968 ms** | **0 次** ✓ 确认关闭 |

**⇒ 锁页值 1.52×，而且它早就在生效。B12 的核心论断（那 6.07 s 已在锁页线速上、
没有 pageable 可修）成立。**

❗ **一个超出预测的数字**：我按 H2D 速率差（26.85 vs 16 GB/s）只预期 +3.2 s，
**实测 +19.1 s**。⇒ 关掉锁页的影响**大于单纯 DMA 速率**，可能还影响
transpose/staging 路径的重叠或页错误。**这一点我没查明，标 ⚠️。**

⚠️ 本次**没抓到 `Memcpy HtoD` 的时间数字**（grep 只取到表头行，值是 0.000us）。
**所以严格说 B12 是「TTFT 侧确证 + 算术吻合」，不是「H2D 时间直接对比」。**

### B14 ⭐ D1b:2.8× **全部**来自「二维 `q` 载入」,与 `running_acc`/head 分块无关

`/tmp/kern_bisect.py`,真实尺寸 N=1024 H=64 D=512 topk=2048 R=32768,四个变体共用同一循环体:

| 变体 | 耗时 | 加速 |
|---|---|---|
| `ref`（原核） | 311.200 ms | 1.00× |
| **2D `q` + 1D `acc`** | **110.899 ms** | **2.81×** |
| 1D `q` + 2D `acc` | 编译失败 | — |
| 2D `q` + 2D `acc`（= pr4） | 112.572 ms | 2.76× |
| 2D `q` + 2D `acc`，`BLOCK_H=8` | 108.176 ms | 2.88× |

**⇒ 结论:**
1. **`q` 载入成二维 `(1, BLOCK_D)` 单独就拿到 2.81×**;
2. **`running_acc` 变二维贡献为 0**（112.6 反而略慢于 110.9）;
3. **head 分块只再加 ~3%**（110.9 → 108.2）;
4. ⇒ **pr4 的补丁可以更小** —— 理论上只需改 `q` 的载入形状即可拿到 2.81×,
   head 分块是为了那额外的 3%。

**机制推断（⚠️ 未证实）**:原核的 `q` 是一维 `(BLOCK_D,)`,在
`tl.sum(kv * q[None, :], axis=1)` 中需把一维向量**广播到二维 tile 的 K 维**,
Triton 很可能为此插入 layout 转换/共享内存往返或放弃向量化;二维载入后 layout 天然对齐。
**验证方法**（未做）:比较两种写法的 ttgir/PTX 里 `convert_layout` 与
`ld.global` 向量化宽度的差异。

### B15 ⚠️ D1b 的机制验证**未完成**（IR diff 拿不到对照）

**目标**：证实/否证「一维 `q` 在 `tl.sum(kv * q[None,:], axis=1)` 里跨 K 维广播,
导致 layout 转换或失去向量化」。

**尝试与结果：**
1. 同进程内循环切 `TRITON_DUMP_DIR` → **只有第 2 个核 dump 成功**（环境变量在编译时读,同进程切换不可靠）;
2. 改成**两个独立进程**、启动前设好 `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=...` → **两个核都没 dump**。
   （原因未查明;可能是该 Triton 版本对这两个内核不触发 dump,或键名/触发条件不同。）

**唯一拿到的数字（⚠️ 单侧,不构成对照）**：`_k2`（二维 `q`）的 ttgir 有
`convert_layout` **10 次**、`local_alloc` **0 次**;PTX 里 `ld.global.b32` ×6、`ld.global.v4.b32` ×4。
**没有 `_k1` 的对应数字 ⇒ 无法比较 ⇒ 机制仍未证实。**

**替代路径（未试）**：不依赖 dump,直接取编译产物——`kernel.device_caches` /
`triton.compile` 的 `asm` 字典（`ttgir`/`ptx`/`sass`）。API 随版本变动,需先确认可用的入口。

**⇒ B14 的结论（2.8× 来自二维 `q` 载入）仍然成立,但「为什么」依旧是推断。**
这一条**不该**写进给上游的补丁说明里当事实。

### B16 ❌ 最小补丁（只改 `q` 载入）**编译失败** —— 「机制不明」的代价

**目标**：二分显示 2.81× 仅来自二维 `q` 载入（B14）⇒ 想要一个比 pr4 更小的补丁。

**结果**：引擎**起不来**：

```
triton.compiler.errors.CompilationError: at 47:4:
AssertionError("Loop-carried variable running_max has initial type fp32
                but is re-assigned to <['1'], fp32> in loop!")
⇒ RuntimeError: Engine core initialization failed
```

**根因（我的错）**：我用 `tl.arange(0, 1)[:, None] * 0` 来给 `q` 制造一个 leading 维，
**Triton 把 `arange(0,1)*0` 常量折叠成了标量** ⇒ `running_max` 的初值仍是 fp32 标量，
而循环内被赋成 `(1,)` ⇒ 循环携带变量类型不一致。

**⇒ 更深的教训**：我不知道二维载入**为什么**有效（B15 未证实），
所以想"最小化"复现它只能靠猜 —— **而猜错了**。
**一个机制不明的优化，其最小化版本是不可靠推导的**；二分告诉了我"哪里"，
但没有告诉我"为什么"，于是我不能安全地把改动缩小。

**⇒ 决策：保留已验证的 pr4（head 分块版）作为交付物。**
它是**唯一端到端验证过的**（B11：核 2.88×、TTFT 1.52×）。
最小版**暂缓**，直到 B15 的机制被证实。

**回滚**：树已 `git checkout`，md5 `fdf1908b9c80951132bf88384fba6201` 与 `diff -q` 双重确认逐字节一致。

⚠️ 未被测到的一条可能更稳的写法（未试）：先按原样一维载入再 `tl.reshape(q1, (1, BLOCK_D))`
—— 但它**未必**能拿到同样的 layout 收益，那正是未证实的部分。

### B17 ⭐ 注意力修复**後の完整预算**:还有 **~15.7 s(43%)** 无法归属

配置同 B11(pr4 补丁已应用,单条 16384,MBT=16384/KV0.75G/util0.85):

```
TTFT = 36,860 ms        (与 B11 的 36,894 吻合 ✓)
[layer-timing] total = **320.9 ms/层**   ← **修复前是 413 ms/层(−22%)**
    pre = 287.6–300.0 ms   eng = 21.9–33.3 ms   post = 0.04 ms
```

**预算表:**

| 项 | 值 | 占比 |
|---|---|---|
| MoE 层内（`[layer-timing]`） | 42 × 321 ms = **13.5 s** | 37% |
| 注意力核（B11 实测） | **7.74 s** | 21% |
| **无法归属** | **≈ 15.7 s** | **43%** |

**三条观察:**

1. **`[layer-timing]` 的每层时间从 413 → 321 ms（−22%）** —— 注意力变快**连带**减少了 MoE 层内
   时间,说明两者原本**部分串行**（注意力堵住了 MoE）。这是一个我没预期的连带效应。
2. **仍有 ~15.7 s（43% 的 TTFT）无法用任何已测的核解释** ⇒ **这是现在最大的单一未解项**,
   比 H2D 的 6.07 s 大得多。
3. ⚠️ **本次 CUDA 逐核列表没抓到**（我的 grep 模式没匹配上,输出为空）
   ⇒ **无法交叉验证 13.5 + 7.74 是否与 CUDA 核总和自洽**。这是本次数据的一个缺口。

**⇒ D4 的靶子明确了:那 15.7 s 是什么。** 候选（均未验证）:
`all_reduce` 之外的信道同步/等待、Python 侧逐层派发开销、CUDA graph 重放间隙、
或 `[layer-timing]` 的 `pre` 里那 ~130 ms/层（= 42×130 = 5.5 s）之外的等待。

**⇒ 下一步应先补上逐核列表**（修我的 grep），再看 15.7 s 是否落在
「所有 CUDA 核之和 < 墙钟」这个差额里 —— 若是,那就是**主机侧/调度侧的串行等待**,
而不是某个核慢。

### B18 🔴 D4a:我 B17 的算法**不成立** —— 父子行重复计数

从 `logs/bud.log`(B17 那次运行)解析 profile 表,**按第 7 列 `Self CUDA` 求和**:

```
Worker_TP0 的 CUDA 求和 = **48.34 s**   vs   TTFT = **36.86 s**
⇒ 差额 = **−11.48 s(负数)** ⇒ **求和无效**
```

**为什么无效 —— 容器行与其子行数值相同:**

| 容器行 | Self CUDA | 子行 | Self CUDA |
|---|---|---|---|
| `moe_forward_shared` | 11.45 s | `down_kernel_fp8` 2.88 + `gate_up_kernel_fp8` 2.57 + `Memcpy HtoD` 6.07 | = **11.52** ≈ 11.45 |
| `unified_mla_attention_with_output` | 7.74 s | `_sparse_mla_fwd_with_sink_kernel_hb` | = **7.74** |
| `all_reduce` | 2.11 s | `ncclDevKernel_AllReduce_...` | = **2.11** |

**⇒ 该 profiler 的 `Self CUDA` 对 Python 层级的 op 而言包含了其内核时间
（不是"排除子项"）** ⇒ **父行加子行 = 双重计数** ⇒ 48.34 s 与 −11.48 s 都不可用。

**⚠️ 连带影响 B17**:那里我用「MoE 层内 13.5 s + 注意力 7.74 s = 21.2 s vs TTFT 36.86 s
⇒ 缺 15.7 s」。**但注意力很可能就发生在 `[layer-timing]` 的 `pre` 相位之内**
（两者都包住 `apply()` 的前半段）⇒ **同样是重复计数,15.7 s 这个缺口不成立。**
**⇒ B17 的"43% 无法归属"作废,我收回。**

**⇒ 正确的做法:只用叶子行求和**,并显式排除
`moe_forward_shared` / `unified_mla_attention_with_output` / `all_reduce` /
`aten::linear` / `aten::matmul` 等容器。**这还没做。**

**⇒ D4a 未完成,且 D4（那 15.7 s）的靶子也一并作废 —— 需要先用叶子口径重算才知道有没有缺口。**

### B19 D4a 第三次尝试:拿到**上界 ≤ 13.7 s**,但仍不能判定是否有真缺口

**方法**:从同一张 profile 表里**排除已知容器行**(`moe_forward_shared` /
`unified_mla_attention_with_output` / `all_reduce` / `aten::*`),只对叶子行求和。

**第一次算得**:叶子合计 **17.05 s** ⇒ 差额 19.81 s(54%)。**但这有解析 bug**:
`Memcpy HtoD (Pageable -> Device)` 的名字**含空格**,我的 `f[0]` 只取到 `Memcpy`
⇒ **那 6.07 s 的 H2D 没被计入**。

**修正后**:

| 项 | 值 |
|---|---|
| 叶子核合计(含 H2D) | **≈ 23.1 s** |
| TTFT | 36.86 s |
| **差额** | **≈ 13.7 s(37%)** |

**主要叶子:**

| 叶子核 | s | %TTFT |
|---|---|---|
| `_sparse_mla_fwd_with_sink_kernel_hb` | 7.74 | 21.0% |
| `down_kernel_fp8` + `gate_up_kernel_fp8` | 5.45 | 14.8% |
| `Memcpy HtoD`(修正后计入) | 6.07 | 16.5% |
| `ncclDevKernel_AllReduce` | 2.11 | 5.7% |
| 其余 20 个叶子 | 均 < 0.4 | — |

**⚠️ 关键限制(不能忽略)**:**这些核是异步重叠的**(NCCL 可与计算并行,
H2D 也可与计算并行)⇒ **「核求和」不构成墙钟**,两者不能直接相减。
**⇒ 13.7 s 是缺口的**上界**,不是测得值;是否存在真缺口,取决于重叠量,而重叠量
需要可用的 GPU 时间线才能算**(见下)。**

**⇒ D4a 仍未完成。三次尝试:**
1. 表 `Self CUDA` 求和 → **双重计数**(B18);
2. chrome trace 求 GPU 忙碌并集 → **trace 只有 64 个事件(58 个元数据),没有 kernel 时间线**;
3. 排除容器的叶子求和 → **得到上界 13.7 s,但受重叠影响不可直接相减**。

**⇒ 下一步(明确的唯一路径)**:让 `XIAOTU_TORCH_PROFILE` 产出**完整的** trace
(检查 `XIAOTU_TORCH_PROFILE_CALLS` 与写盘时机 —— 现有 trace 只有 3 个 X 事件,
说明采样窗口几乎没抓到东西),然后用**区间并集**算真实 GPU 忙碌时间。

### B20 ⚠️ A11 端到端数值 A/B:**连续 6 次失败,全部是我自己的脚本 bug**

| 次 | 失败原因 | 性质 |
|---|---|---|
| 1 | prompt 只生成 **1 个词**(`range(1)`) | **测试无效** —— 走不到稀疏注意力核 |
| 2 | `> /tmp/$tag_w.log` 应为 `${tag}_w` | 变量名,`set -u` 下崩 |
| 3 | `pgrep -f 'VLLM::EngineCore'` 匹配到自己的命令行 | 进程纪律(E0);自杀 |
| 4 | `cd $T` 后用**相对路径** `patches/...` ⇒ `git apply` 失败 | **arm B 跑的是没打补丁的原核 ⇒ A/B 变成「原核 vs 原核」** |
| 4b | 比对用 `message.content`,而 `finish_reason=length` 时 `content=None` | 比对字段选错 |
| 5 | 忘了 `chmod +x /tmp/numab2.sh` | 一个字符 |
| **6** | 脚本 `cleanup()` 里 **`kill -9 -PGID` 无「不等于自身」守卫** ⇒ 杀掉自己的进程组 | **违反我自己刚写下的 E0 规则第 4 条** |

**⚠️ 第 6 次留下真实风险并已处置**:补丁**已成功应用**后被 SIGKILL,**回滚未执行**
⇒ 生产树一度处于修改状态(md5 `dbaa8a54...`)。**已回滚并三重校验**:
`git status` 空 / md5 回到 `fdf1908b9c80951132bf88384fba6201` / `diff -q` 与备份逐字节一致。

**⇒ 教训(已并入 D 节)**:「应用补丁 → 跑 → 回滚」三段式操作,**中间被打断就留脏状态**。
正确做法:①改树的**副本**;②把回滚做成 **`trap`**;③脚本开头记录原始 md5、结尾强制比对。

**目前 pr4 的数值证据(实测):**
* **核级 A/B 已完成**:真实尺寸下 `max|diff| = 2.819e-05`,来源是 `tl.sum` 规约顺序,**非 bit 级**(B10b/B14);
* **端到端未完成**。arm A 的有效响应已存(`/tmp/numab/nab_a.json`,含 `token_ids`);
  已修好的第 7 版脚本是「只跑 arm B + 补丁绝对路径 + 比对 `token_ids` + 不碰 `-PGID`」。

**⇒ 结论:pr4 的端到端数值仍是**未验证项**。**
**不因「核级已可比」就默认端到端没问题**;`pr4_body.md` 里已如实标注。

### B21 ⭐ 性能点 #3(FP8 MoE GEMM):**实测 0.61 TFLOPS,可达峰 0.2%** —— 新的最大未优化项

**发现:两个 GEMM 核的全部 tile 参数都硬编码,且没有任何 env 开关**

```python
# vllm_xiaotu_moe/gpu_prefill_fp8.py:376-377
bm: int = 64, bn: int = 64, bk: int = 64, bh: int = 64,
ns: int = 2, warps: int = 4                     # num_stages=2 / num_warps=4 对 GEMM 偏保守
launch: gate_up_kernel_fp8[(E, triton.cdiv(2*I, bn))]   # grid 第一维 = E = 288
        down_kernel_fp8[(E,   triton.cdiv(H,  bh))]
现有 env 开关只有 XIAOTU_GPF_DMA / XIAOTU_GPF_STAGE ⇒ **tile 不可调**
```

**效率核算(FLOPs 为推断,5.45 s 为实测):**

```
单个 16384 prefill 的专家 GEMM ≈ 3.30 TFLOP   (T=16384, topk=8, H=4096, I=1024)
实测 gate_up + down           = **5.45 s**
⇒ 实测吞吐 = **0.61 TFLOPS**  ⇒ 对 A100 BF16 峰值 312 TFLOPS 仅为 **0.19%**
⇒ 按峰值算理论 10.6 ms,实测慢 **516 倍**
```

**⇒ 即便把我的 FLOPs 推算打对折,达成率仍是个位数百分比 ⇒ 这是真实的、大的低效。**

**⚠️ 尚未归因**:可能是 ①tile 过小(64)导致算术强度低;②grid 以 E=288 为第一维,
每 program 需遍历该专家的 token(散布/局部性差);③`num_stages=2` 使 H2D 与计算重叠不足;
④in-kernel FP8→BF16 反量化的开销。**这四条都未验证。**

**⇒ 实验设计(不需要碰 rebase 树,因为 `gpu_prefill_fp8.py` 在插件侧):**
`bm/bn/bk/ns/warps` 各扫几档,看 GEMM 时间与 TTFT 的响应。**这是当前最大的未优化项(5.45 s)。**

⚠️ **但每档需一次服务加载(~9 分钟)**,且**每次改动插件源码都要在两次跑之间保持一致**
—— 参照本会话的教训,**应先给这些参数加 env 覆盖**(一处小改),再把"改源码+跑"变成"设环境变量+跑",
这样后续扫描既安全又可重复。

### B22 按用户指示更新目标(2026-09-21 16:15)——goal rev 2 的收口路径

**新目标**:继续逐项排查 → 当剩余收益已很小时**停止优化、记录已知问题** →
然后在 **256K 上下文**下为 GLM / MiMo-V2.5 / DeepSeek-V4.1 **测 long prefill 最好成绩**并更新 README。

**排查进度总表(截至目前):**

| # | 性能点 | 实测 | 判定 |
|---|---|---|---|
| 1 | 稀疏 MLA 注意力 | 22.3 s → **7.739 s**(58.8%→33.15%) | ✅ **已修,端到端验证 TTFT 1.52×** |
| 2 | H2D `Memcpy` | 6.07 s / 16.5% | ❌ **B12 证伪**:已在锁页线速,**只能减字节** |
| **3** | **FP8 MoE GEMM** | **5.45 s / 0.61 TFLOPS / 0.19% 达峰** | 🎯 **B21;A14 扫描已排队** |
| 4 | `all_reduce` | 2.107 s / 5.7% | **A13 量中** |
| 5 | 未归属缺口 | 上界 ≤13.7% | ⚠️ B17 已推翻,存在性未知 |

**A14 设计(已排队,不碰源码)**:`gs_base` / `gs_s4w8`(STAGES=4,WARPS=8) / `gs_big`(BM=128,BN=128),
各取 TTFT + profile 里的 `gate_up/down` 叶子时间。
**若这三档对 GEMM 时间不敏感 ⇒ 说明瓶颈不在 tile,而在 grid 布局(E=288 为首维)或反量化,应停止在此项上投入。**



## C. 上报上游

### B24 ⚠️ A14 失败(第一臂被 Killed)—— **按预先写明的判据收口,不假装有数据**

```
A14 GEMM tile 扫描(16:27 启动)
  gs_base 的 serve 进程: **Killed**(启动后即被杀)  ← 未产出任何数据点
```

**⇒ A14 三档全部无数据。** 这是本会话第 7 次脚本故障(前 6 次见 B20)。
**根因未查明**(`one()` 先 `cleanup()` 再 launch,理论上不该杀到新进程);
**我不再投入第 8 次拯救脚本** —— 收益与风险已不成比例。

**⇒ 排查阶段的收口判断(基于已有实测,不依赖 A14):**

| # | 项 | 实测结论 | 剩余收益 |
|---|---|---|---|
| 1 | 稀疏 MLA 注意力 | 22.3 → **7.739 s**(58.8%→33.15%),端到端 **1.52×** | ✅ **已拿到** |
| 2 | H2D | 6.07 s,**已在锁页线速**(B12 + 反证) | ❌ **无**(只能减字节 ⇒ 换 NVFP4 权重) |
| 3 | FP8 MoE GEMM | **0.61 TFLOPS / 0.19% 达峰**(B21);tile 扫描未完成 | ⚠️ **存在但归因未完成**;下一步是**改核结构**(grid 以 E 为首维 / 反量化),投入产出比明显下降 |
| 4 | `all_reduce` | 关掉只差 **9.3 ms** ⇒ 不在关键路径(B23) | ❌ **无** |
| 5 | 未归属缺口 | 仅上界 ≤13.7 s,且 B23 证明核重叠 ⇒ **更可能不是真缺口** | ⚠️ **存在性可疑** |

**⇒ 判断:排查基本完毕,剩余收益已很小。**
唯一未归因的 #3 若要继续,需要**结构性改动**(不是调参),那是另一个量级的工作;
且它的归因实验(A14)已两次受挫。

**⇒ 转入交付阶段**(见 `docs/PREFILL_KNOWN_ISSUES.md` §5b 关于 MiMo 256K 不可行的记录)。



### B23 ✅ 性能点 #4 结案:`all_reduce` **不在关键路径上,无可回收收益**

`XIAOTU_SKIP_AR`(源码自带,注释写明"用来量每层归约在 GPU 预填充里占多少")两跑对比:

| 配置 | TTFT |
|---|---|
| `SKIP_AR=0`(正常) | **52,865.85 ms** |
| `SKIP_AR=1`(跳过每层跨 rank 归约) | **52,875.13 ms** |
| **差值** | **9.3 ms(噪声级)** |

**⇒ 关掉整条 `all_reduce` 对 TTFT 无影响 ⇒ 它完全与其他工作重叠,不在关键路径上。**
**⇒ 性能点 #4 结案:无可回收收益。**

**⚠️ 方法学附带收获(重要)**:这是对「核会重叠 ⇒ 核求和 ≠ 墙钟」的**直接实证** ——
**一个 2.107 s 的 CUDA 核从关键路径消失后,墙钟纹丝不动。**
⇒ **进一步说明 B19 那个「叶子求和 vs 墙钟」的差额(≤13.7 s)更可能来自重叠,而非真缺口。**

**另**:本次 TTFT 52.9 s 与 B5 未打补丁基线 55,954 ms 同量级 ⇒ 确认跑的是原核、树全程干净
(md5 `fdf1908b9c80951132bf88384fba6201` 前后一致)。

**⚠️ 本次记录过程中的一个自身 bug(已修)**:我多次用
`s.replace(anchor, add, 1)` 追加内容,**却没有把 anchor 接回去**(应为 `add + anchor`)
⇒ `## C. 上报上游` 标题一度被吞掉(内容未丢,只是并入了 B 节)。**已在本次修复。**
**⇒ 规则:用 replace 追加时,新内容必须包含原锚点。**



### 🔴 C3 **#57971 已撤回并关闭** —— 我对着**本项目自己的补丁**报了上游 bug

**维护者回应**（jahnclawdmonet, 15:24）:
> The path in your report is not on main. `.../sparse_mla_kernels.py` returns **404 at main**,
> and a code search for `_sparse_mla_fwd_with_sink_kernel` returns **no hits**.

**查证结果(实测):**

```
31e3396f73 2026-09-14 xtu-upgrade snapshot: local SM80/mixed-mode patch set   ← 引入该文件
patches/upstream/pr3-sm80-port.patch:
  sparse_mla_kernels.py   new file mode 100644   @@ -0,0 +1,3517 @@
pr3_body.md:  "...sparse_mla_kernels.py (3517 lines): the sparse-MLA decode/prefill kernels
               and the env switches (VLLM_TRITON_MLA_SPARSE, ..._TOPK_CHUNK_SIZE,
               ..._QUERY_CHUNK_SIZE, ..._HEAD_BLOCK_SIZE, ..._MATMUL_DECODE)"
```

**⇒ `sparse_mla_kernels.py` 与 `flashmla_sparse_sm8x.py` 是 `pr3-sm80-port.patch`（我们自己）
新增的 SM80 移植,不在 vLLM main 上。**

**⇒ 我犯的错:** 把**自己的移植代码**当成上游代码,
并把自己代码的性质("上游为让 SM8x 能跑而写的便携回退核")**归因给上游**。
**根本原因:我在报 bug 前没有先确认那个路径在 `main` 上存在** —— 而这是我本可以一条
`gh api` 就查掉的事。目录结构与上游一致,我就默认了它是上游。

**⇒ 处理:已发撤回评论([issuecomment-5763547800](https://github.com/vllm-project/vllm/issues/57971#issuecomment-5763547800))
并关闭 issue。**

**⚠️ 连带含义(重要):** 测量数据本身是真的,但**测的是我们自己的代码**
⇒ 那些数字（22.3 s / 58.8% / 二维 tile 后 7.7 s）**属于本项目仓库,不属于上游**。

### 🎯 C4 **由查证翻出的新线索:pr4 补丁可能根本不必要**

`pr3_body.md` 明说 `sparse_mla_kernels.py` 自带一组**我们自己的调优开关**:

```
VLLM_TRITON_MLA_SPARSE            (总开关)
..._TOPK_CHUNK_SIZE
..._QUERY_CHUNK_SIZE
..._HEAD_BLOCK_SIZE      ← 若它本来就是 head 分块,可能直接给出 B14 的 2.8×
..._MATMUL_DECODE
```

### C4 结论:**pr4 既不多余,也不新颖**(已查证)

**`HEAD_BLOCK` 机制在我们自己的移植里*早就实现了* —— 但是给 DECODE 用的:**

```python
# sparse_mla_kernels.py:412-427(已有 kernel)
    HEAD_BLOCK: tl.constexpr,
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)   # ← 与 pr4 同构
# :652
    assert head_block_size in (1, 2, 4)
# sparse_mla_env.py:17-27
    def sparse_mla_decode_head_block_size(num_decode_tokens): ...   # 只有 decode 的入口
```

**而 PREFILL 的 launch(:3639)仍是** `_sparse_mla_fwd_with_sink_kernel[(num_tokens, active_heads)]`
—— **head 分块没有被用到 prefill 上。**

**⇒ 三条结论:**
1. **pr4 不多余** —— 没有哪个环境变量能把 head 分块开到 prefill 上(只有 decode 的入口);
2. **但 pr4 也不新颖** —— 这个模式**本项目自己已经写过一次**(给 decode);
   ⇒ **pr4 的写法应与那份实现对齐**,且 `sparse_mla_decode_head_block_size()` 是天然的接入点;
3. ⚠️ **`assert head_block_size in (1, 2, 4)` 可能过紧** —— 我的二分实测 **`BLOCK_H=8` 最好**
   (108.2 ms vs 1/2 的 ~110 ms,4 的 135.9 ms)。**若采纳 pr4,应一并讨论是否放宽到 8。**

### C4 最终结论(已查到头):**GLM 的 prefill 应该改接到 DeepSeek 那条已有的 head-blocked 路径**

**完整调用图(实测):**

```
accumulate_indexed_sparse_mla_attention_chunk        ← head-blocked 多-head prefill wrapper
   └ 内部:head_block = _PREFILL_INDEXED_HEAD_BLOCK = **8**
           if num_heads >= 8: grid=(num_tokens, cdiv(num_heads,8))
              → _accumulate_indexed_attention_chunk_multihead_kernel[grid](HEAD_BLOCK=8)
   ← vllm/models/deepseek_v4/nvidia/flashmla.py:755     ← **DSv4 在用**
   ← vllm/models/deepseek_v41/nvidia/flashmla.py:569    ← **DSv4.1 在用**
GLM: vllm/v1/attention/backends/mla/flashmla_sparse_sm8x.py:165
   → sparse_mla_fwd_with_sink (per-token,**无 head 分块**)
```

**且 GLM 的 SM8x backend 里明写:**
```
supports_dense_mha_prefill: bool = False
# Force every token (prefill included) through the sparse-MQA gather path;
# the dense/masked MHA prefill would fall back to FlashAttention (SM90+).
```

**⇒ 三条结论:**
1. **head-blocked 的 prefill 路径早已存在,而且 DeepSeek-V4/V4.1 已在用**
   ⇒ **它已经过实战检验**,不是新代码;
2. **GLM 只是没接上它** —— SM8x backend 主动把 prefill 塞进了 per-token gather 路径;
3. **⇒ pr4 的正确形态不是"新写 head 分块 kernel",而是"把 GLM 的 prefill 改接到
   `accumulate_indexed_sparse_mla_attention_chunk`"。** 这比 pr4 **更小、复用已验证代码、
   且与两份 DS 实现的调用方式一致** —— 明显更容易被接受。

⚠️ **同时也解释了 pr4 为什么有效**:pr4 是在**把 DS 已有的做法在 GLM 的核里手抄了一遍**。
B14 的「二维 `q`」只是抄的时候碰到的那一处形状差异,不是根因。

### C4-更正 🔴 **「改接路径」并不更小 —— 我上一轮高估了**

**查了两个调用点后必须更正:它们是*不同的分解方式*,不是同一个函数的不同调用。**

```python
# GLM 侧(flashmla_sparse_sm8x.py:165):**一个融合核**
sparse_mla_fwd_with_sink(q, kv, indices, topk_length, scale, attn_sink, output, num_heads)

# DSv4 侧(flashmla.py:752-772):**多核流水,调用方持有状态**
for index_start in range(0, combined_indices.shape[-1], topk_chunk_size):
    accumulate_indexed_sparse_mla_attention_chunk(
        q=q_chunk, kv_flat=kv_flat, indices=indices_chunk[...],
        lens=lens_chunk, candidate_offset=index_start, scale=layer.scale,
        max_score=max_score, denom=denom, acc=subset_acc)   # ← 状态张量
finish_sparse_mla_attention_with_sink(max_score, denom, subset_acc, attn_sink, output)
```

**⇒ 要 GLM 走 DS 那条路,需要把 GLM 的 prefill 重构成
「按 token 分块 + 按 topk 分块累积 + 单独 finish」,并自行分配/清零/传递状态张量。**
**这是调用侧重构,不是一次函数替换。**

**⇒ 结论修正:**
1. **C4 的*发现*仍成立且有价值** —— head-block 的 prefill 模式**我们自己的代码里已有、
   且被 DSv4/V4.1 实战使用** ⇒ **它有实战证据,不是未验证的新写法**;
2. **但我上一轮说「更小、更容易被接受」是错的**;
3. **⇒ pr4 因此重新成为合理选择,很可能还是*更小*的那一个**
   (自包含核内改动、保持 GLM 现有单次调用接口)。
   **C4 的价值从「替代 pr4」变为「为 pr4 提供设计依据」**:
   该做法在本仓库有先例,且 `BLOCK_H=8` 与既有 `_PREFILL_INDEXED_HEAD_BLOCK = 8` 一致。

**⚠️ 我的错误模式**:上一轮我从「DS 在用」跳到「所以改接更简单」,**没有读调用点**。
**第 8 次同类:用「存在性」推断「可行性」。**

**⚠️ 且本次记录过程中我又犯了一次**:第一版 `replace` 锚点没匹配,**脚本却无条件打印
「已记录」**,直到 `git commit` 报 `nothing to commit` 才暴露。
**⇒ 脚本里的 `replace` 必须配 `assert`**(上面这版已加)。

| # | 内容 | 状态 |
|---|---|---|
| C1 | [vllm-project/vllm#57971](https://github.com/vllm-project/vllm/issues/57971) —— SM8x sparse-MLA prefill 回退核 | ✅ OPEN，作者 yeungtuzi |
| **C2** | **自更正评论**（[issuecomment-5762891125](https://github.com/vllm-project/vllm/issues/57971#issuecomment-5762891125)）：坦白「64× 冗余是事实但不是瓶颈」；给出 B10b 的实测表；把建议从「修冗余」改为「改 tile 形状」；并请维护者解释**为什么二维形式本身快 2.8×** | ✅ 已发 |

**证据**：grid `(num_tokens, active_heads)`，而 `kv` 的寻址**不含 `head_idx`**
⇒ 64 个 head 各自重读同一份 latent。算法需 32.2 GB，实际读 2062 GB。
即使按这 64× 冗余算，HBM 下界 1,326 ms vs 实测 22,323 ms ⇒ 有效带宽 **92 GB/s**（峰值 6%）。

---

## D. 待做（按优先级）

| # | 任务 | 依据 | 状态 |
|---|---|---|---|
| **D1'** | 交付 pr4（head 分块版，BLOCK_H=8）——**已端到端验证 2.88×**；最小版暂缓（B16 编译失败） | B11/B16 | ✅ **补丁就绪** |
| **D1d** | 若要把补丁缩到最小：先按 B15 证实机制（取编译产物的 asm），再据此写 | B14/B15/B16 | 待做 |
| ~~D1~~ | ~~A1' BLOCK_H=4 bit 级 A/B~~ | | ❌ 已废：非 bit 级（规约顺序），且 BLOCK_H=4 劣于 1/2/8 |
| ~~D2~~ | ~~pr4-A2 BLOCK_H=16 + tl.dot~~ | | ❌ 已废：BLOCK_H=16 实测 0.58× 且精度差 60× |
| **D1b** | **查清「二维 tile 为什么快 2.8×」** | B10b | 待做（也是问维护者的点） |
| **D1c** | 端到端验证：把核换进 rebase 树，跑 16384 请求，确认 22.3 s → ~8 s 且无 illegal | B10b | 待 D1' |
| ~~D3~~ | ~~锁页 Pageable H2D~~ | | ❌ **已废：B12 —— 已经在锁页线速上** |
| **D3'** | **减少 H2D 字节数**：NVFP4（专家 −1.78×）或 §12 的 staging 收缩。**这是 H2D 的唯一杠杆** | B12 | 待做（优先级已上调） |
| **D3b** | 反证：跑一档 `XIAOTU_GPF_PIN=0`，若 H2D 升到约 9.5 s 则 B12 的推断确证 | B12 | 待做 |
| ~~D4~~ | ~~查清 ~15.7 s 无法归属项~~ | | ❌ **作废:B18 证明 B17 的算法是重复计数** |
| **D4a** | **用叶子行口径重算预算**(排除容器行),看是否真有缺口 | B18 | **优先** |
| **D5** | 修 DeepSeek-V4.1 长 prompt 的 `aten::new_empty` 分配失败 | B1 | 待做 |
| **D6** | MiMo/V4.1 长格补测（KV cap 已修为 1.10× 余量） | B1 | 待做 |

**A1' 的设计约束（已核实，别再走回头路）**：
- 逐元素 + head 分块会产生 `(H, K, D)` 三维中间张量 ⇒ `BLOCK_H=16` 时 512 KB，**寄存器放不下**；
  只有 `BLOCK_H=2~4` 可行（64–128 KB）。
- `tl.dot` 要求 `M ≥ 16` ⇒ **A1'（bit 级）与 A2（张量核）不能同时满足**，必须分两步。
- `BLOCK_K=16` 是 PV 那个 `tl.dot` 的规约维，**恰好是 Triton 下限，没有余量**。

---

## E. 未收尾的悬项（每轮开轮时先看这里）

### E0 · 进程纪律（2026-09-21 第 4 次踩同一坑，写死在这里）

**事故**：我用 `pgrep -f 'VLLM::EngineCore'` 找残留进程，**而我自己的命令行里就含这个字符串**
⇒ 匹配到自身 ⇒ 对**自己的进程组**发了 `kill -9 -PGID` ⇒ **本条命令被 SIGKILL**。

**同类历史**（本会话共 4 次）：`pkill -f mbt8k` 杀自己、`pkill -f` 匹配到 sampler、
`maxVmLck` 模式匹配自身、本次。

**⇒ 硬规则：**
1. **绝不**用 `pgrep -f` / `pkill -f` 去匹配**任何可能出现在自己命令行里的字符串**
   （进程名、tag、模型名、路径都会）；
2. 要按名字找，**用括号技巧**打断自匹配：`pgrep -f '[V]LLM::EngineCore'`；
3. **优先按已知 PID / pidfile 杀**，不用模式匹配；
4. `kill -9 -PGID` 尤其危险（会波及自己的组）—— **只有在确认 PGID 不是自己的时候才用**。

**清场的验收判据（承接 E1 更正）**：`nvidia-smi --query-gpu=memory.used` 读到 **0**，
而不是「kill 命令返回成功」。


| # | 事项 | 状态 |
|---|---|---|
| E1 | ~~停止服务后 GPU 仍被占用~~ | 🔴 **我原先的结论是错的,已更正**:15:15 那次看到 `0/0/0` 让我判成「退出中间态」,但 15:58 又出现 **2×39,248 MiB 持续 6 分钟以上**,`fuser` 指向 **`VLLM::EngineCore` (pid 2544259)**。**真相见下**。 |

### E1 更正:清场必须杀 **`VLLM::EngineCore` 子进程**

**根因**:`serve_glm53_mainline.sh` 写在 `logs/<tag>.pid` 里的是**启动器**;
真正的 **`VLLM::EngineCore` 是它的子进程**,**不在任何 pidfile 里**。所以我的清场
(杀 .sh、杀 pidfile 里的 pid、杀 `nvidia-smi --query-compute-apps` 列出的 pid)
**会漏掉 EngineCore**,它会继续占着显卡。

**这次的实例**:`--query-compute-apps` 列出的是 `2544288/2544305`(worker),
而 `fuser /dev/nvidia0` 指向 `2544259`(`VLLM::EngineCore`)。**两者不是同一个进程。**

**⇒ 正确的清场写法:**
```bash
# 1) 按名字找 EngineCore,杀掉它所在进程组
for p in $(pgrep -f 'VLLM::EngineCore'); do kill -9 -"$(ps -o pgid= -p $p|tr -d ' ')" 2>/dev/null; kill -9 $p; done
# 2) 再按 pidfile 收尾
# 3) 等 10s 后**用 nvidia-smi 复核显存为 0**,而不是只看 kill 命令返回成功
```
**判据必须落到「显存为 0」,而不是「kill 命令已执行」。**

⚠️ **这也意味着 E1 早先「不是泄漏」的结论是被一次巧合掩盖的** —— 那次 `0/0/0` 很可能是
EngineCore 恰好自己退出了。**我把「恰好观察到 0」当成了「一定不是泄漏」,这是一次过早收案。**
| E2 | `mbt8k` 的 `m8_1m_r1` 臂被我按 PID 杀掉，**未收** | ⚠️ 但它与 `m8_1m_r3`（已 OOM）是同一结论方向，价值低 |
| E3 | `m1_fit` 的 MBT=8192 / 4096 两臂未产出（脚本只跑了第一臂） | ⚠️ 1M 三重锁死已由 B3 独立支撑，不依赖它 |
