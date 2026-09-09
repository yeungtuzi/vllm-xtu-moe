# fork(lk/lvllm) vs vLLM 主线计算模块 决策分析

日期:2026-09-08
状态:决策分析(为下一步「走哪条路」提供证据;本机待办的三步=文档化+搭主线环境+调研 NUMA 控制权)
相关文档:ref/xiaotu_cpu_expert_mainline.md · ref/gpupre_design_xiaotu.md · results.txt

## 0. 问题
在并发 decode ~100 tok/s、支持 2 路 1M 上下文、平台 = 2×A100 + EPYC 9654 + 24 通道 DDR5-4800、
跑 DeepSeek-V4-Flash / GLM-5.3-Flash 等的前提下,该继续走 lk/lvllm fork(做其开源替代 xiaotu),
还是转投 vLLM 主线做它的计算模块?**第二路能不能达到我们现在的性能和效果?**

## 1. 决定性事实:decode 是带宽受限,不是核数/内核算术受限
- 用户实测(已多次,本会话前置):48 / 96 / 190 核性能都差不多。
- 推论:decode 早已不满内核算术,瓶颈在**从 DDR5 读 fp4 激活权重的带宽**。
- 带宽是**硬件属性,与谁写内核无关** → 同一内核、同一权重、同一根内存带宽,读到的字节一样,
  **上限必然一样**。任何能把 fp4 从 DDR5 以接近全带宽读出的内核都撞同一天花板。

## 2. 100 tok/s 的带宽账(验算)
- DS-V4-Flash:hidden 4096, moe_intermediate 2048, n_routed 256, top_k 6, 43 层。
- 单层全专家 fp4 = 3.221 GB;单层激活(topk6)= 75.5 MB;每 token decode 读激活权重 ≈ **3.246 GB**。
- 输出目标→所需 DRAM 读:21.5 tok/s≈70 GB/s;50 tok/s≈162 GB/s;**100 tok/s≈325 GB/s**。
- EPYC 9654 24 通道 DDR5-4800 理论 921 GB/s,现实 ~600-690 GB/s。100 tok/s ≈ 现实峰值 ~50%,可行。
- 结论:100 tok/s 不是"选哪条路线"决定的,是"decode MoE 是否从 DRAM 读 + 是否批处理"决定的。

## 3. 为什么现在只有 ~14-32 tok/s(旧基准)而非 100?
- results.txt [ROUND 横向]:总吞吐 31-33,输出吞吐 14-15(ShareGPT, conc-4, 整模型在 CPU)。
- 关键:当前**不是带宽受限**(仅 ~45 GB/s),是每次调用 ~2.3ms 的 MoE 延迟 × 43 层 ≈100ms/step,
  加 dense/attention/GPU-KV 又 ~125ms/step → TPOT ~225-300ms。
- 用户实测的 100 tok/s 是**另一套更优配置/口径**(并发解码 + 更彻底地把 decode MoE 从 CPU 串行
  挪走/批处理)达成的;本会话**不重复验证**(用户明示),以上为背景。

## 4. 两条路的真实差异(不是性能,是编排/维护/生态站位)
同一份计算内核,无论挂 fork 还是主线插件,读同样硬件同上的内存 → 计算性能一样。
真正不同:
- **编排责任**:fork = 你全管(serving/KV/scheduling/TP + 独有的流式 prefill);主线 = 标准编排
  (serving/KV/scheduling)有上游维护,但你得把独有流式 prefill 写进插件扩展点。
- **维护成本**:fork = 追上游每个 commit,重;主线 = 只维护薄接口 + 内核 + 自己的流式模块,轻。
- **生态**:主线模块化 MoE + offload orchestrator + CPU backend 都在快速演进,站主线肩上。
- **关键风险**:主线的量化 CPU 内核(MXFP4/FP8/INT4/INT8)**全要求 AMX(Intel 专有)**;
  本机 AMD(无 AMX)主线自己的量化 CPU 内核全用不了 → 只有我们的内核能在这台机器跑 MXFP4/FP8,
  **我们不是可有可无,是主线在这台机唯一可用的 CPU 量化内核**。

## 5. 唯一致命未知数:主线插件能否拿到 fork 相同的 NUMA 分片/线程绑定控制权
- 100 tok/s 靠 single-copy NUMA 分片(#9)+ 跨 socket 交叉点读打出(results 反复强调
  "cross-node weight reads hurt, Do NOT use single-copy")。
- 若主线插件拿不到对权重分片/线程绑定/NUMA 节点的控制权,325 GB/s 堆不满 → 100 tok/s 掉。
- **这是"第二路能不能到 100 tok/s"里唯一真可能导致失败的开关,必须实证(步骤③)**。
- 能保住 100 tok/s 的前提(必须一起搬进插件,不是只填 apply()):
  1. 非 unscan 内核 + 同一 lean fp4 内核;
  2. **NUMA 分片/线程绑定控制权**;
  3. 持久 pinned 缓冲 + decode 批处理(M≥2);
  4. decode MoE 继续从 CPU DRAM 读,不被主线"想当然"搬 GPU(138GB 装不下 GPU)。

## 6. 2×1M 上下文:不是风险
- 这是 KV/引擎特性(长上下文/KV 池/MLA 量化),不是计算模块特性;vLLM 主线只会更成熟,无风险。

## 7. 结论(给决策)
- **计算内核这条腿注定一样**(同一硬件同一带宽 → 同一上限)。
- **能否到 100 tok/s,不取决于"写不写主线计算模块",而取决于把 fork 的 NUMA 分片 + pinned
  缓冲编排按多大比例带进插件**。带全 = 到;只带内核 = 大概率到不了。
- **真诚建议**:以主线插件为主、两条腿共用同一内核,不二选一(计算内核共享 → 性能无差异;
  fork 唯一好处是"自己全管"但代价是维护整个栈;主线在需要的每一块都在迭代)。把标准编排外包
  给主线,把独有流式 prefill 编排写成插件模块。前提是先过步骤③的 NUMA 控制权关。

## 8. Novelty 论证(无论哪条路都值得先钉死)
- **计算内核不是护城河**(fp4/amx/avx512 内核,O SDI 论文 + vLLM 主线 MML 都有,都能到带宽上限附近)。
- **"单缓冲流式 prefill 的显存/带宽编排"才是护城河**:逐层 H2D 单缓冲(占 1 层 3.22GB)把 PCIe
  打满(1550-1590 tok/s vs lk 零拷贝 415)是别人没做到的:
  - vs lk:显存≈0 但慢 415;我们占 1 层但 ~3.8×快。
  - vs OSDI SLP:他们没有单缓冲压显存到 1 层的编排(双卡 64GB,不用极端省);我们的针对点是
    "超紧显存预算 + 单缓冲"。
- 一层 claim(exact upper):"在单卡显存装不下模型硬约束下,单缓冲逐层流式 prefill 把显存压到
  1 层的同时打出 PCIe 满带宽吞吐"——审稿人视角需与 SLP 划清界限。

## 9. 【步骤③实证】主线 NUMA 分片/线程绑定控制权调研结论
以 main 分支(commit 6c73b08,vLLM 0.28)源码为准,实测如下:
- **线程绑定:主线把 NUMA/绑核控制权完整暴露给进程/插件层。**
  `VLLM_CPU_OMP_THREADS_BIND`(默认 "auto")→ `ompmultiprocessing.py` `_get_autobind_cpu_ids()`:
  每个 local rank(TP worker 进程)绑定到**自己的一个 NUMA node** 的全部核;再按 Intel/非 Intel
  OpenMP 设 `KMP_AFFINITY`(granularity=fine,explicit,proclist=...)/ `GOMP_CPU_AFFINITY` /
  `OMP_PLACES+OMP_PROC_BIND`,并设 `OMP_NUM_THREADS` = 该 rank 核数。`nobind` 可禁绑。
- **计算内核的线程模型:`at::parallel_for`(torch intra-op 线程池/OpenMP)**。
  `csrc/cpu/sgl-kernels/moe_fp8.cpp` 等 `fused_experts_fp_kernel_impl` 对 M、M*topk 并行,
  依赖进程级 OpenMP 线程池。=> **插件 `apply()` 在进程内执行,自动继承该 rank 绑定的线程池 +
  NUMA 亲和性;插件无需/无法自行 `set_num_threads`(主线禁用)但**自动**拿到绑核结果。**
- **权重分片:`_load_w13/_load_w2` 按 `tp_rank` 分片**,每 TP rank 在**自己进程内存**持有本 rank
  的 expert/矩阵分片(= 本地 NUMA 存储,无跨 socket 复制)→ 与 fork 的 single-copy 语义对应。
  CPU MoE 各量化类 `_supports_parallel_config` 返回 True(支持 EP/TP;`apply()` 均带 `expert_map`)。
- **结论:主线 NUMA/绑核控制权充分暴露且机制等价。**
  要到我们 100 tok/s 的效果,标准做法 = `--tensor-parallel-size N`(每 rank 绑一 NUMA node,
  每 rank 持本地 expert 分片),配 `VLLM_CPU_OMP_THREADS_BIND` 选 node。这等价于 fork 的
  "single-copy NUMA 分片",只是**以多进程(TP/EP)而非单进程内分片**表达。
- **对插件开发者的含义**:我们的 lean 内核挂进 `FusedMoEExperts.apply()` 后,天然运行在
  该 rank 的绑核 OpenMP 线程池上;只要内核用 `at::parallel_for`(vLLM CPU kernel 常规做法)或
  自管线程但尊重 rank 的 NUMA 分区,就能吃到与 fork 相同的跨 socket 本地读 100 tok/s 上限。
- **注意点(诚实的边界)**:
  1) 主线 CPU 并行是"进程级 TP",不是 fork 的单进程 168-192 线程 + 自管 NUMA pool;要 2 socket
     = 开 2 个 TP rank(每 rank 一 NUMA node)。若目标是整机 24 通道,需把 TP/DP 设到覆盖全部
     可用 NUMA node,并让每层专家分片落本地 —— 这是 fork(单进程全核)与主线(多进程)表达差异,
     但带宽天花板相同。
  2) `cpu_experts` 的 `CPUExpertsMxfp4` 等仍是 Intel AMX-only;我们无 AMX,须写自己的 xiaotu
     内核类(放宽 `_supports_current_device` 为 x86),它按 `apply()` 契约在 rank 内跑、
     用 rank 的 `at::parallel_for` 线程池。
  3) 2×1M 上下文:KV/引擎特性,主线成熟,无风险。

---
## 修订记录
- **2026-09-08b(本版)** — §9 新增步骤③实证结论:主线把 NUMA/绑核控制权完整暴露
  (`VLLM_CPU_OMP_THREADS_BIND` → 每 TP rank 绑一 NUMA node + OMP env),计算内核用
  `at::parallel_for`(继承 rank 绑核线程池),权重按 tp_rank 本地分片(=single-copy 语义),
  CPU MoE 各量化类支持 EP。=> 主线能拿到与 fork 等价的 NUMA 控制,到 100 tok/s 的开关通过。
  依据:main 分支源码实测(vllm/utils/ompmultiprocessing.py、csrc/cpu/sgl-kernels/moe_fp8.cpp、
  vllm/model_executor/layers/fused_moe/experts/cpu_moe.py)。
- **2026-09-08(本版)** — 新建。综合本会话讨论:核数无关→带宽受限事实 + 100tok/s 带宽账验算 +
  两条路真实差异(编排/维护/生态)+ 唯一致命未知数(NUMA 控制权)+ novelty 论证。为下一步
  「搭主线环境 + 调研 NUMA 控制权」提供证据基线。
