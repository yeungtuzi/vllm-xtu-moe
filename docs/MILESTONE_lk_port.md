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

**lk-moe 在同一台服务器上的解码基准是单流 30-35、投机 ≈50、C=4 ≈70 t/s**;
我们目前单流 20.29(不投机)、C=2 聚合 34.29。差距 1.6-1.7×,正在按下面顺序定位:

1. **同机同配置、只换引擎**(参考 env `lvllmds4-x` 里仍是原版专有 `lk_moe` 2.4.2)——
   这一次对照能直接判定差距在"我们的引擎 in-service 行为"还是"配置/编排参数";
2. 若在引擎:对照 `compute`/`rest` 拆分,优先看 ①线程数与绑定(每 rank 只用了 48/96 核)
   ②EP 归约(每层一次跨进程 shm barrier,参考实现是在引擎内部合并)③pinned staging 拷贝;
3. 若在配置:按 lk README 试 `LVLLM_GPU_RESIDENT_MOE_LAYERS`(用 KV 换常驻层)、
   `PREFETCH`/`MBT`/`THREADS` 的扫描。
