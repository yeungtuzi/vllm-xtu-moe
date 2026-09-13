# 里程碑:lvllm 编排链移植进 vLLM fork + 挂上 xiaotu_moe(实测版)

**日期**：2026-09-13 · **状态**：功能全绿、已实测、8070 上线冠军配置
**目标**：`goal-36fc65c0-16b0-4da3-94e8-7c2f612b4e72`（第 210 轮用户指定,已 complete）

---

## 1. 交付了什么

| 组件 | 位置 | 说明 |
|---|---|---|
| lk 编排链 | `Lvllmds4-x/vllm/model_executor/layers/fused_moe/routed_experts.py`(1770 行)<br>`.../runner/moe_runner.py`(1014 行)<br>`Lvllmds4-x/vllm/envs.py`(lk 段) | **与最新官方 `Lvllm/patches/01_lk_moe__3116c5d.patch` 逐行相同**,只把 `import lk_moe` 改名成 `xiaotu_moe`(commit `faf95dd5b`) |
| 计算引擎 | `xiaotu_moe/`(Apache-2.0) | 5 个 ISA 变体;实现 lk 的 `MOE_*`/`MOEConfigV2` 接口 |
| 启动器 | `scripts/serve_lk_port.sh` | 作者参数为默认;**自动识别 dspark 并把草稿钉在 GPU**;显存护栏(放不下就禁用 draft + 警告);`SPEC=auto/1/0`、`SPEC_JSON`、`EXTRA_ENV` 可调 |
| 引擎侧两处关键修复 | `xiaotu_moe/csrc/python_binding/binding.cpp`、`csrc/moe/numa_pool.hpp`+`moe_v2.hpp` | ①**CUDA 图捕获安全**(构造期预分配 pinned 缓冲,捕获期拒分配)②**TP=2 rank 切分**(核表按 rank 过滤 + 相对 node 下标 + 分片/节点对齐) |
| 文档 | `report/tuning/NOTES.md` §290-§299、`docs/PERFORMANCE_OPTIMIZATION.md`、`docs/UPSTREAM.md` §8.3/§10、`report/tuning/TRIED_AND_REVERTED.md` R102-R106 | 全部结论与回退都有出处 |

## 2. 怎么跑(两卡 A100-40GB,冠军配置)

```bash
ENV=/home/user/anaconda3/envs/lkxtu TAG=win TP=2 GPU_UTIL=0.80 MAXLEN=1048576 \
  SEQS=8 MBT=8192 MINBATCH=1024 PREFETCH=1 EAGER=0 THREADS=48 SPEC=0 \
  bash scripts/serve_lk_port.sh
# 开投机(自动识别 dspark + 草稿常驻 GPU):
#   ... SPEC=auto bash scripts/serve_lk_port.sh
```

## 3. 实测性能(本机两卡)

| 指标 | 数值 |
|---|---|
| 预填充 8192 冷 / 热 | 982 / **1675 t/s** |
| 预填充 32768 | **1287 t/s** |
| 解码 C=1(TP=2 / 图 / 不投机) | **20.29 t/s**(TPOT 37.33 ms) |
| 解码 C=2 聚合 | **34.29 t/s** |
| 解码 C=1 / C=4 / C=8(TP=1 / 图) | 19.42 / 38.34 / **45.46 t/s** |
| KV(1M 上下文) | 13.15 GiB = **201 万 token** |
| 每层成本(`XIAOTU_CD_TIMING=1`) | TP=1 1.06 ms(compute 0.53 + rest 0.53);TP=2 1.03 ms(0.37 + 0.66) |
| 投机接受长度 | 2.10(n=3/greedy)~2.75(n=5/probabilistic);**两卡净亏**(盈亏平衡 ≥3.03) |

## 4. 正确性

* 数值门禁 `scripts/test_block23_equiv.py` = 文档基线 **OK=7 BAD=1**(me=1,max_rel 1.873e-02)。
* 端到端 greedy 探针(`scripts/probe_greedy.py`,5 个 prompt)输出正确、连贯。
* 引擎对齐脚本 `scripts/check_engine_aligned.sh`、投机接受率均可测可读。

## 5. 合规

* **复用顺序**:上游 vLLM → lvllm →(都没有才)自研。移植路径下**没有新增任何 vLLM 功能**;
  我们的改动只在自家引擎与启动脚本(策略/参数)。
* **投机解码全序列来自 vLLM**(逐环节来源见 `docs/UPSTREAM.md` §8.3)。
* **draft 永远在 GPU**:`SPEC=auto` 自动从 ckpt config 识别 dspark,把草稿层号并进 lk 现成开关
  `LVLLM_GPU_RESIDENT_MOE_LAYERS`(实测日志 `[GPU]`);放不下时**禁用 draft 并警告**。

## 6. 尚未达成(等用户指令,已存档 `report/tuning/FUTURE_PLAN.md`)

**同机同配置、只换引擎的对照已完成**(2026-09-13,`lkref_tp2` vs `lkport39win`):

| | 参考 lk_moe | ours |
|---|---|---|
| 不投机 C=1 | 20.48 t/s(TPOT **25.36 ms**) | 20.29 t/s(TPOT 37.33 ms) |
| 不投机 C=2 聚合 | **52.71** | 34.29 |
| 不投机 C=4 聚合 | **66.10** | ~30-38 |
| 投机 C=1(5/probabilistic) | 5.18 | **9.68** |
| 预填充 8192 冷 / 32768 | 883 / 828 | **982 / 1287** |

* ⇒ **差距 100% 在引擎的每层 marshalling**,不在编排/参数/卡数:把 TPOT 按 `F + C·V` 拟合,
  我们的**每 token 引擎成本更好**(V=6.45 vs 8.84),但**每步固定开销几乎翻倍**
  (F=30.9 ms vs 16.5 ms)。`XIAOTU_CD_TIMING` 拆出每层 `rest` 0.53-0.66 ms,
  参考的等价值 ≈0.3 ms ⇒ **每层 ~0.3 ms × 43 ≈ 13 ms/token** 就是要追回的部分。
* **投机不是优化点**:一步要验证 `num_seqs×(1+spec)` 个 token,而这些 token 全落在 CPU MoE 上
  ⇒ 每接受一个 token 的 CPU 工作量 = (1+spec)/接受长度 ≈ 2.4×,结构性亏损(参考引擎更差:5.18)。
  用户记忆中的 "80-90 tok/s" 已确认是 `SpecDecoding metrics` 里的
  `Drafted/Accepted throughput` 字段,不是端到端速率。
* **下一步(按性价比)**:①staging 与计算重叠(第二流 + event 双缓冲)②3 次 D2H 合成 1 次
  ③用常驻线程 + event 替换 `cudaLaunchHostFunc` ④再用 KV 换若干常驻层。
