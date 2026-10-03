# HANDOFF —— 2026-10-03 · 两条实验线的失败记录与已知问题

> 状态:**UVA 零拷贝 ping/pong 线 = 放弃 ✓;W4A8(int8/AVX512-VNNI)CPU prefill 线 = 放弃 ✓**
> 本文件是《失败交接》:记录做了什么、**什么被实测确立**、什么**没做完**、以及已知问题与坑。

---

## 一、当前仓库状态(动手前先看这里 ✓)

```
分支  main                              = 2dc9afe(含 UVA 0.2.5 的提交)
分支  exp/uva-w4a8-failed-2026-10-03    = d1ab48b ← 【两条线的全部改动】(已提交 ✓ 工作区干净)
标签  FAILED-uva-w4a8-2026-10-03        ← 标记为失败 ✓
分支  archive/main-before-rollback-2026-10-03 / tag archive-main-2dc9afe-2026-10-03 ← 回滚前保险 ✓

候选回滚点(⚠️ 两者不等价,相差 83 个文件):
  · v0.2.4                 = 02fb89f (2026-09-22)
  · pre-uva-exp-2026-09-30 = dd221ea (2026-09-30,"checkpoint before the uva zero copy experiment")
       ⇒ dd221ea → HEAD 只差 8 个文件 = 【UVA 实验本身的范围】✓
```

**回退方法(二选一,已确认后再执行 ✓)**
```bash
# A) 只撤 UVA 实验(推荐,不丢 09-23~09-30 的工作)
git checkout main && git reset --hard pre-uva-exp-2026-09-30

# B) 回 0.2.4(会丢 83 个文件的工作 ✗,执行前请确认 archive 分支存在 ✓)
git checkout main && git reset --hard v0.2.4
```

---

## 二、UVA 零拷贝 ping/pong 线:**失败,已放弃**

· 0.2.5 的结论(仓库里已有):host-shard prefill 路径与 baseline **等价**,默认关(default off)
· 后续讨论过的"微缓冲 ping/pong"(整层 2×7.3 GB → 2×256 MB)做了**可行性实测**,结论是**算术成立但无收益场景**:
  | 量 | 实测 |
  |---|---|
  | PCIe Gen4 H2D 带宽(pinned) | 26.8 GB/s(≥256 MB);16 MB 只有 19.2 GB/s |
  | 线性拟合 | time(ms) = 0.484 + 0.0385×size(MB) ⇒ 固定开销 484 µs(上界) |
  | 256 MB tile 传输 | 10.35 ms(固定开销占 4.7%)|
  | 双缓冲流水线(29 tile/层) | 290 ms vs 纯传输 272 ms ⇒ **额外开销仅 6%** ✓ |
  | compute 与 transfer 并发 | 可以并发,不串行 ✓ |
  | 掩盖阈值 | M ≥ ~73 token/tile(长 prefill 满足)✓ |
  | ⚠️ GPU0 的 PCIe | **只有 x8**(GPU1/2 是 x16)⇒ 传输要绑 GPU1/2 |
  ⇒ 即:**14.6 GB → 0.5 GB 可行,且掩盖能力保持** —— 但这条线在本项目中**未继续**(用户决定放弃)。

---

## 三、W4A8(int8/VNNI)CPU prefill 线:**内核成立,端到端不成立 ⇒ 放弃**

### 3.1 被实测确立的事实(有证据 ✓)

```
① int8 整数对齐内层(B2)内核实测:
     B  = 1.425x  ·  B2 = 1.582x(11 轮中位,两核两 seed ±0.5%,需 g++ 模板编译期 chunk)
   同量化 fp64 逐位校验:maxabs = 0.000e+00(0/32,6 种子)⇒ 【内层逐位精确 ✓】
② 决策关键:【"整数口免费"在 Zen4 上不成立】
     vpsllvd / vpmovzxbd / vpshufb 与 vpdpbusd / vfmadd 【同为 1 条 zmm/周期】
     ⇒ 省 FP0/1 的同时新增 13 条 1-周期级整数指令 ⇒ 净收益 28/20 = 1.40(实测 1.425 ✓)
③ FP0/1 上限是 【4.67x】,不是 2.0x(我一度算错:fp32 FOLD 是 4 条 fma + 4/6 条 mul / 每(行,64 MAC))
④ 端到端五次同口径实测(基线 2048=230 / 32k=345):
     fp32=1.00x · 旧 int8=1.02-1.04x · 方案1 B=1.03-1.08x(含 vpsllvd 语义 bug,假快)
     B2 修好=0.99-1.03x · B2+G1修正=0.95x
   ⇒ 【把 MoE 内层端口压力降 42%,端到端纹丝不动】✓
⑤ 路径计数(384,020,000 次):ALIGN=100% / OLD8=0
   ⇒ 【ALIGN 新内层确实在执行】✓ —— 推翻了"可能没进"的猜测 ✓
⑥ 干净环境下最好一次:32k = **379.6 tok/s**(prompt 28154,ttft 74.2s)✓
```

### 3.2 ⇒ 放弃的理由

```
MoE 内层快了 1.58x,端到端却不动 ⇒ 【瓶颈不在 MoE 内层】✓
⚠️ 但【MoE 占端到端多少】这个数,本会话【没有直接测到】✗
   (我一度反推"约 20%"⇒ 后来自己发现该反推依赖"内核真跑到 1.58x"这个未验证前提 ⇒ 【已撤回】✗)
⇒ 所以正确的表述是:【MoE 内层改动对端到端不敏感】是实测事实;
  "MoE 占比 X%" 仍是【未测量】✗ —— 这正是交接待办的第 1 项。
```

---

## 四、⚠️ 已知问题 / 坑(按严重程度)

### 4.1 代码级
```
① 【落码/撤回必须 grep 验证语义判据,不能只看"编译通过"】
   我宣布"β 已撤回、OOB 已消除"✗,实际 β 仍在(编译通过 + 我误读了同一份输出里的计数 ✗)
   正确:改完必须 grep 该改动的【专属判据】(如 sll_epi16==0 / beta==0)✓
② 【vpsllvd 不取低 5 bit】✗ —— 移位量 ≥32 ⇒ 整 lane 【清零】,静默错、不崩、不报错
   实测:1<<0x03030303 = 0、1<<32 = 0
   ⇒ 用"字节表当 lane 计数"必须【拓宽成 dword】(vpmovzxbd)✓
③ 【按槽缓存的指针必须在 命中/未命中/失败 三条路径全部设置或清空】✗
   否则读【上一个块的陈旧表】(本会话踩过两次:g_i8_WT2、以及 §457 的 En)✓
④ 【三态门控必须互斥】:旧 int8 / 新 int8 / fp32
   曾写错成"启动新路径的条件"与"跳过旧路径的条件"不互补 ⇒ 守卫失败时【两条都不跑 ⇒ C 无人写】✗
   正确:显式 i8_new_on_ / i8_old_on_,fp32 用 !(两者)✓
⑤ 【64B zmm 覆盖 64 元素;一个 MXFP4 32-块只有 32 字节】
   裸 _mm512_loadu_si512 取 32 项会把【下一块】读进来 ✗
   ⇒ 要么 g_ += 2 一次 64 项,要么 maskz_loadu_epi8(低 32B)
   ⚠️ masked load 仍做页存在性检查 ⇒ 缓冲尾要留 ≥64B ✓
⑥ 【宏内每一行(含注释)必须以 \ 结尾】,否则宏被截断(本会话踩过 4 次)✗
   自检:awk '/^#define XIAOTU_TILE_BODY/{f=1} f&&/^#undef XIAOTU_TILE_BODY/{f=0} f{ if ($0 !~ /\\$/ && $0 !~ /^#undef/) print NR": "$0 }' <file>  ⇒ 只应输出宏收尾那一行
```

### 4.2 测量/插桩级(我被用户当场纠正三次 ✗)
```
⑦ 【热路径禁止 IO/锁/malloc】—— 我把 fprintf 放进 3.84 亿次的分支入口 ✗
   ⇒ 日志 41.9 MB、192 线程抢 FILE 锁 ⇒ 【273s/279s 全是日志】⇒ 那次测量全毁 ✗
   正解:thread_local 环形缓冲(纯内存写)+ 退出/信号时一次 dump ✓
⑧ 【打点粒度】:优先打在【分支入口/深循环之前】;若必须在深循环里 ⇒ 【1/N 采样】✓
⑨ 【计数器不能用无锁 static long + tot%N==0】✗:192 线程竞态 ⇒ 打印放大约 28 倍 ✗
   正解:atomic fetch_add 返回旧值 ⇒ 只有【一个】线程命中该倍数 ✓
⑩ 【要插桩就编独立的 instrumented .so 到 /tmp】✓,绝不拿现役 .so 试 ✗
⑪ 【perf record -a 需要权限】✗(本机 perf_event_paranoid=1,`-a` 出来 0 字节)
   `perf record -p <pid>` 只采到主线程 ⇒ 3K 样本/89s(192 线程应几万)✗
   ⇒ 更好的工具:py-spy(--native,无需 root)✓
⑫ 【清场必须杀进程组,不能只杀端口派生 PID】✗
   本会话 VLLM::EngineCore(RSS 660 GB)在测量结束后【还活着】⇒ 污染了后续所有吞吐 ✗
   正解:bash scripts/proc.sh stop <TAG>(按 PID 文件 ⇒ 杀组 ✓)
⑬ 【按名字杀进程会自杀】✗(本会话两次 ✗):pkill -f "run_accept_safe.sh" / pgrep -f run_accept_safe
   都匹配到【自己的命令行】⇒ 把自己杀了 ✓
   正解:模式拼接 PAT="run_accept""_safe" ✓,或读 PID 文件 ✓
⑭ 【/proc/<pid>/stat 字段偏移】:切掉 "comm) " 后 utime=$12、stime=$13
   我用了 $14/$15 ✗ ⇒ 解析出 "S4"/"S5" ⇒ 占比全 0 ✗
```

### 4.3 流程级
```
⑮ 【不要循环论证】:我拿 CPU_MOE_CONCLUSION.md 的旧结论("345=天花板")来"印证"新结论 ✗
   而 W4A8 项目的全部理由就是【打破它】⇒ 两个不同命题 ✗(用户当场指出 ✓,已撤回)
⑯ 【引用旧数据前必须核原始口径】:我用一个台账已标注"不适用真实负载"的微基准(4,558 tok/s)
   去算 MoE 占比(得 7.6%)✗ ⇒ 已撤回 ✓
⑰ 【"改判据"类建议前先通读验收文档】:AB_W4A8_ACCEPTANCE.md 早已预注册"L1 不过"与处置协议
   ⇒ 我一度建议改判据 ✗ ⇒ 有"事后放宽通过线"嫌疑 ⇒ 已撤回 ✓
```

---

## 五、未完成 / 交接待办

```
① ⭐【prefill 全路径占比:IO / 计算 / 等待 / 空转】—— 这是用户最后一次明确要求,【尚未取得】✗
   已建好但未验证:
     · /tmp/vnni_phase.so(插桩版 ✓ 编译通过 ✓):阶段采样器(quant/predecode/tile_body)
       设计符合纪律:热路径零 IO/零锁 ✓、1/256 采样 ✓、每线程 shared slot(atomic 认领)✓、退出 dump ✓
     · /tmp/phase_run.sh:跑插桩版 + 每秒抓 EngineCore 全线程 utime/stime(字段已修 $12/$13 ✓)
   卡点:① [PHASE] dump 尚未在日志里确认 ② py-spy/perf 权限未开
   需要:装 py-spy(pip install py-spy)或 sudo sysctl -w kernel.perf_event_paranoid=-1
② W4A8 精度验收 ①②③(GSM8K / Vision / 1M 四针)【从未在本会话跑过】✗
③ ALIGN 路径相对 baseline 的正式 A/B(≥3 次复现)未做;唯一干净单点 = 379.6 tok/s ✓
④ 代码资产(默认关、可回退、零回归 ✓):源码快照 /tmp/p4.phase.hpp(md5 见 /tmp)
   · 门控:XIAOTU_MOE_INT8_ALIGN(默认 0)⇒ 不设时行为与基线【逐字相同】✓
   · 方法学工具:/tmp/avx512_align_bench.c(端口标定 + 六内核对照,含 4.67x 上限依据)
   · 台账 dev-docs/UVA_ZEROCOPY_EXPERIMENT.md §1~§477(含全部负结果与自我更正)
```

---

## 六、给接手者的一句话

> **MoE 内层是真快(1.58x,逐位精确),但端到端不动 —— 所以下一步绝不是继续优化内层 ✗,
> 而是先按第一节的待办 ①,把 prefill 的 IO/计算/等待/空转 占比【直接量出来】✓,
> 再决定力气该花在哪 70-80%。**

日期:2026-10-03 · 分支:exp/uva-w4a8-failed-2026-10-03 · 标签:FAILED-uva-w4a8-2026-10-03
