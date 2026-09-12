# 后续规划(低优先级)—— 第一条(CPU 解码引擎对齐 lk_moe)的剩余项

> 状态:**第 96 轮由用户结案**。第一条的引擎目标在其可达范围内已完成:
> DEDUP=12 `1.22 → 0.64-0.65 ms`、DEDUP=23 `1.32 → 0.84 ms`,与 lk 同口径 **0.89× / 0.80×**,
> 数值门禁 7 OK。本文件收集"当时没做完/没做成"的项,供以后按需回顾,**当前不投入**。
> 证据链:NOTES §113-§142、TRIED_AND_REVERTED R1-R85。

## A. 已定位、但收益有限或受门禁阻挡(最值得以后回看)

### A1. `dotp16`(bf16 点积)路径 —— 已修结构、被精度卡住
- 第 94 轮给该分支加了 NR 列分块(MR=4/NR=4,12 条独立累加链),性能从 0.72-0.76 → **0.64-0.66 ms**
  (与 fp32 默认的 0.67 相当),**证实 R65 的"bf16 更慢"真因是它一直是未做列分块的旧 GEMV**。
- 但它**不能启用**:数值门禁 `me=2` 用例 `max_rel 8.42e-3` > 项目门限 `2e-3`
  (其余用例都 OK:me=1 8.0e-4、me=3 8.7e-4、混合 4.9e-4)。
- **重新打开的前提**:① 把门限按"该路径单独分布"重新校准(注意 lk_moe 生产就是用 bf16 点积);
  或 ② 定位 Zen4 `vdpbf16ps` 成对求和的具体误差来源,用"每 K 块拆两次 + 更频繁归约"换精度。
- 收益上限不大(与 fp32 持平),所以列在低优先级。

### A2. DEDUP=23 的 ms 条款(0.84 vs 目标 0.70)
- DEDUP=23 时 `na=20`、字节 252 MB,是**访存主导**区;我们 300 GB/s vs lk 376 GB/s(0.80×)。
- 已排除:分片数、线程数、分块参数、领票方式、伪共享、bf16、L3 策略、master 绑核、冷启动。

### A3. DEDUP=12 的每线程条款(1.94 vs 2.2)
- 低并发(T≤60)时我们 **2.21-2.29 GB/s·线程 = lk 满并发时的 2.21**(§138);
  只在 T=120 掉 12-16%。同机 D=12 的"两个条款互斥"(T=120 满足 ms、T=60 满足每线程)。

## B. 未解之谜(值得以后单独查)

1. **"折 scale"为何不等价**(R73):把按组 scale 乘进权重(`w'=w·sv`)在数学上恒等(scalar × dot),
   实测却把 me=2/3/4..7 全部打坏(max_rel 1.6-2.7,2× 量级)。`row_scale` 的定义已核对
   (`row_scale((j/gn)*kb_stride, g)`,每 (列块, k 组) 一个标量)。**这个矛盾至今没解释**,
   而它背后可能藏着解码/缩放配对顺序的真实语义。
2. **残差到底属于什么**:访存字节与指令数两个单因素模型都失败(§141:bf16 减 10-15% op 只换 3-4%);
   "120 线程执行效率"是当前唯一站得住的表述,但没有可操作的机制。

## C. 已明确撤销/不要重试(避免重复劳动)
- K-major 列 lane 内核(R77):同构 op 账证明列 lane 与 k lane 的指令数**完全相同**(≈0.23 条/MAC);
  省的是"解码复用更多行"(需要 me=8,由路由决定),不是"摊到列上"。
- "绕过 L3 / 非临时载入"(R81):等 na 对照证明工作集越大越慢,不是 L3 驻留的错。
- 批量领票/静态均分(R70)、cacheline padding(§123)、reserve 替 resize(R58)、
  DPBF16 作默认(R85)、spin 超时(R62)、NSHARD 手调(R67)、SHARDSPLIT 细分(R59/R82)、
  master 绑核(R78)、冷启动(R79) —— 全部实测无收益或更差。

## D. 若要把第一条彻底做满,唯一还剩的方向
按 lk 的真实形态重建内核:**权重"预解码 + 预乘 scale"到工作缓冲,并让一份解码结果同时服务
多列 × 多行**。在 `[N][K/2]` 布局下做不到(列间相隔 K/2 字节),需要 K-major 权重
(插件侧 `gpu_prefill._kmajor_bytes` 已有同一份张量)+ 列 lane 内核。§136 的 op 账说明:
**只有当 `me`(每专家行数)足够大时它才划算**,而基准口径 me=3-4 ⇒ 预期收益 ≤1.2×。

---

## P0(阻塞项·用户 2026-09-12 指定)引擎"请求几次就 abort"必须修掉

**为什么是 P0**:它使 item 2 的任何"达标"声明都缺一个必要条件 —— 引擎必须在一段**持续负载**
内存活。当前没有任何一次 TP=2 会话撑过一次完整压测。

**故障链**(已定位,见 NOTES §172-§175):
分片池有任务永不完成(`remaining_` 停 1/2)→ `numa_pool.hpp:713` 看门狗(默认 300s,
`XIAOTU_MOE_SHARD_WD` 可调)打印 dump 后 **`abort()`** → worker SIGABRT(无 traceback)
→ `VllmWorker-N died unexpectedly (exit code: None)` → shm 读端 cancel
→ `RuntimeError: cancelled` → HTTP 500。

**已否证的假设(不要重复)**:①Triton OOM/常驻层(R91 撤回);②常驻层是预填充杠杆(R91);
③我的热路径埋点导致卡死(R93 更正);④`XIAOTU_MOE_EP_SHM` 的 /dev/shm 双 barrier(R121 否证);
⑤陈旧 `node_base_` 读(R94 否证)。**另:`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 与该故障无关
(R120 更正:`cancelled` 是次生症状)**。

**下一步(已验证手段)**:`XIAOTU_MOE_POOL_TRACE=1 XIAOTU_MOE_SHARD_WD=60` +
已埋好的两个**纯诊断**计数器(不改控制流):
- `abandoned_`:三处 `break`(`:943`/`:981`/`:988`)丢弃已领票据时自增;
- `skipped_dec_`:两处世代守卫为假导致递减被跳过时自增。
看门狗 dump 里谁非零即定位真正路径,**先拿读数再改控制流**(这是第 122 轮用一次白烧换来的规矩)。

**完成判据(四件都要)**:①看门狗不再出现;②复现序列 24 个请求全过;
③`scripts/check_engine_aligned.sh` 数值门禁通过(R55);④**一次 `C=8/N=32` 持续压测跑完并给出持续吞吐**。

**另有一条独立的小问题(仅性能,非正确性)**:`hybrid_model.py:736-738` 的注释断言
"取 `max(预分配, 阈值)`",实现却只取 `XIAOTU_MOE_EP_SHM_TOKENS`(默认 1024)
⇒ GPU 阈值 >1025 时 CPU 通路会**静默**走 NCCL 回退(正确但慢、无日志)。
建议 `_toks = max(EP_SHM_TOKENS, gp_min-1)`(gp_min>0 时)并加一次 warning;
**纯 CPU 模式(gp_min=0)不应按 8192 预分配**(stride 134 MB × 43 层 × 2 rank ≈ 11.5 GB /dev/shm),
回退即正确(引擎 `binding.cpp:440 if (bytes <= ep->capacity)` + `hybrid_model.py:983` 双门禁)。

### P0 更新(第 120 轮末,方向已换)
**已否证的假设增至 6 个**:①Triton OOM/常驻(R91);②常驻是预填充杠杆(R91);
③我的热路径埋点(R93);④EP shm 双 barrier(R121);⑤陈旧 `node_base_` 读(R94);
⑥"递减被世代守卫跳过"(**§179**:`skipped_dec` 读数为 1/0,与缺口不匹配)。
**同时确认两个"基线噪声"**:`miss=1` 与 `abandoned = #workers(=120)` —— 都**不是**丢票证据,
不得再据此改代码(NOTES §174b/§179b)。

**当前唯一真信号**:`exec` 的缺口。`diag2_a2` 的 `total=128 / exec=126 / rem=2`
⇒ 2 个被计入的任务被领走后**既没执行、也没走到任何 break**
⇒ 持票 worker **卡在任务体 `sf_(...)` 内,或阻塞在 re-anchor 的 `work_mtx_` 上**。
⇒ **不要再查票据/世代/递减协议**,改查任务体里会长时间阻塞的东西。

**下一轮具体三步(按成本排序)**:
1. **最便宜判别实验**:`XIAOTU_MOE_THREADS=120 → 32`,跑同一复现序列。
   若是 GIL/锁竞争 ⇒ 触发概率显著变化;若是纯越界/计算错 ⇒ 不变。
2. **直接证据**:在任务体入口/出口各加一个计数器(`entered`/`left`),看门狗 dump 里打印
   两者之差 = "进了没出"的 worker 数,把 §179(d) 的推断变成事实。
3. 若指向 GIL:检查 CPU MoE shard 回调是否在持 GIL 的情况下做长耗时操作,
   以及主线程(torch/采样)是否有长时间持 GIL 的段落。

### P0 最终判定(第 120 轮末,**两种失败模式**)
| 模式 | 签名 | 出现 | 直接证据 | 修法 |
|---|---|---|---|---|
| **A** | `exec==total`、`rem` 少 1、**`skipped_dec`=1** | 2/3 | `skipped_dec` 与缺口**精确相等** | `remaining_[2]` 按世代奇偶分桶,递减**领票即必减**(去掉守卫) |
| **B** | `exec=total-2`、`rem=2`、`skipped_dec=0` | 1/3 | `exec` 缺口 | 查任务体阻塞(GIL/锁),`entered`/`left` 计数定位 |

**机制(模式 A)**:`:983` re-anchor 把 worker 本地 `gen` 改成"活代"⇒ 迟到 worker 的递减
记到活代上(活代被多减、早到 0、调用方提前返回),自己那代被少减 ⇒ 净效果是某次调用永不归零。
**两模式都要修**,只修一个仍会挂死。`abandoned≡#workers` 与 `miss=1` 为**基线噪声,不得当证据**。

---

## P0-A 修复规格(可直接机械执行;**已证明**,不是猜测)

**证据**:三份 dump 中 `skipped_dec` 与缺口**精确相等**(a1/a3:`exec=256=total, rem=1, skipped_dec=1`),
即 `:956`/`:991` 的世代守卫确实跳过了递减(模式 A,占 2/3)。**所有观测到的失败都是
`WATCHDOG(sharded)`** ⇒ **只需改分片路径**,不要碰 flat 路径(它的 10 处 `remaining_` 引用保持原样)。

### 改动(分片路径专用,**新增**一个按世代奇偶分桶的计数器,不动 flat 的 `remaining_`)
1. **声明**(紧邻 `shard_exec_`,约 :1165 一带):
   ```cpp
   std::atomic<size_t> remaining_sh_[2];   // 按世代奇偶分桶的分片计数(见下)
   static int sh_slot(uint64_t gen) { return (int)((gen >> 1) & 1ULL); }
   ```
   注意 `gen` 恒为**偶数**(就绪态),所以用 `(gen>>1)&1`,**不能**用 `gen&1`(恒 0)。
2. **发布**(:696):`remaining_.store(total);` → `remaining_sh_[sh_slot(gen)].store(total);`
3. **等待**(:713 谓词):`remaining_.load(acquire) == 0` → `remaining_sh_[sh_slot(gen)].load(acquire) == 0`
4. **看门狗打印**(:717):`remaining_.load()` → `remaining_sh_[sh_slot(gen)].load()`
5. **两处递减**(:957-965 与 :991-992 一带):**去掉世代守卫,无条件递减**
   ```cpp
   if (!torch_dummy) {}                    // (占位说明:此处不再有守卫)
   if (remaining_sh_[sh_slot(gen)].fetch_sub(1, acq_rel) == 1) { notify_all(); }
   ```
   （原 `if (current_gen_ == gen && ...)` 的守卫整段删除;`skipped_dec_` 的计数点改为
   **不变式检查**:修复后它应恒为 0,非 0 即引入新 bug —— 保留计数,便于回归。）

### 为什么奇偶分桶能修好(安全性论证)
- 原守卫要解决的问题是:**迟到 worker 的递减会污染新一代的倒计时**。
- 分桶后,一个属于第 `g` 代的递减**只能**落到 `remaining_sh_[sh_slot(g)]`;
  第 `g+2` 代用**另一个**桶 ⇒ **结构上不可能**污染下一代 ⇒ 守卫不再需要。
- **桶的复用**:第 `g+4` 代才会复用 `sh_slot(g)`。而复用发生在 `:696` 的 `store(total)`,
  它只在该调用已经等到 `g+2` 归零之后才执行;`g` 代的递减在 `g+2` 归零前必已全部完成
  (`g+2` 的发布以 `g` 归零为前提,见 `:709-712` 的等待)。⇒ 复用是安全的。
- **不会漏减**:每张票由 `fetch_add` 唯一领取,领票者必走"执行 + 递减"。
- **不会多减**:票据唯一 ⇒ 递减唯一。
- 若某 worker 在 re-anchor 后把 `gen` 改成活代 `ng`,它递减的是**活代**的桶,而它执行的
  任务也用活代的 `loc`/任务函数 ⇒ **执行与递减属于同一代**,自洽。

### 验证(缺一不可)
1. 重编译:`PYBIND11_INC=<env>/lib/python3.12/site-packages/torch/include \
   PYTHON=<env>/bin/python scripts/build_engine_variants.sh`
2. `XIAOTU_MOE_POOL_TRACE=1 XIAOTU_MOE_SHARD_WD=60` 启动,**连续 3 次会话**跑
   `/tmp/repro.py` 的 24 请求序列:**全部通过**;`skipped_dec_` 恒 0;`exec==total && rem==0`。
3. 数值门禁 `scripts/check_engine_aligned.sh`(**R55 强制**)。
4. **一次 `C=8/N=32` 持续压测跑完并给出持续吞吐**(`scripts/tune_client.sh`)。
5. 若仍有模式 B(`exec` 缺口、`skipped_dec=0`),再按 §179(e) 查任务体阻塞
   (`entered`/`left` 计数器定位"进了没出"的 worker)。

### 反面约束(不要做的事)
- **不要**动 flat 路径的 `remaining_`;
- **不要**再依据 `miss=1` 或 `abandoned=120` 改代码(两者均为基线噪声,已三次确认);
- **不要**在前向热路径加同步埋点(R92);
- 改动必须有读数支撑(R94 的教训)。
