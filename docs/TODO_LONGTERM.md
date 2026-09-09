# LONG-TERM TODO — Performance Alignment: xiaotu-moe → lk-moe

**作者 Author:** 大河马 (dahema@me.com)，由 DeepSeek Harness 辅助 / with DeepSeek Harness assistance

> **本文件只覆盖"引擎性能对齐 lk-moe"这一条子线。**
> 全项目的待办/决策总账在 [`BACKLOG.md`](BACKLOG.md)(进行中项、上游 PR 策略、
> 已知阻塞、证据索引)。新会话请先读 BACKLOG,再按需读本文件。

This file records the long-running optimization goal tracked as a ToDo, so
progress is persistent and auditable across sessions/rounds. Target is the
**closed-source lk-moe** CPU MoE engine; reference for implementation is the
open-source **ktransformers / lktransformers** (Apache-2.0). We study lk-moe's
*approach* and copy/adapt ktransformers *code*, never lk-moe's own code/binaries.

---

## Primary objective (active)
持续对齐优化 xiaotu-moe 的 CPU MoE 计算内核，使同机(168核 + A100/GPU2)
50-prompts / conc-4 ShareGPT 基准性能追上原版 lk-moe，以
ktransformers/lktransformers (Apache-2.0) 为参考逐项对标优化，并持续在
`/home/user/lvllm/results.txt` / `xiaotu_vs_lkmoe.md` 更新对比结果。

[on] Keep aligning/optimizing xiaotu-moe's CPU MoE compute kernel so the
same-machine (168-core + A100/GPU2) 50-prompts/conc-4 ShareGPT benchmark
performance catches the original lk-moe, using ktransformers/lktransformers
(Apache-2.0) as reference, updating results in results.txt continuously.

**Definition of done (complete):** benchmark total throughput ≥ lk-moe's 51.0
tok/s (currently 25.95) and/or TPOT ≤ lk-moe's 177 ms (currently 293.6), with
50/50 correctness maintained. 0.11 moves the gap from 0.445x → 0.51x total and
TPOT 1.86x → 1.66x; still ~2x short.

---

## Progress ledger
| Date | Round | Change | Metric result | Verdict |
|---|---|---|---|---|
| 09-04 | – | v1 local vector batching | REGRESSED | ✗ reverted |
| 09-04 | – | #2 persistent batched kernel + persistent buffers | 25.50 / TPOT 297 | ✅ kept |
| 09-04 | – | #3 A2+B0 fusion | 24.01 / TPOT 340 | ✗ reverted |
| 09-04 | – | #4 weight-row prefetch (ktransformers) | 25.95 / TPOT 293.6 | ✅ kept (best) |
| 09-04 | – | #5 NUMA page interleave | 25.46 / TPOT 332 | ✗ reverted |
| 09-04 | – | #6 final build confirmation | 25.00 / TPOT 294.1 | ✅ stable |
| 0.11 tag | 09-04 | **Released as v0.11** | 0.51x / TPOT 1.66x | ✅ baseline for next |

---

## Remaining directions (candidates for future rounds)
Prioritized by expected value vs risk across 168 cores / 8 NUMA nodes.

### High value / high effort (structural kernel depth — this is where lk-moe wins)
- [ ] **Algorithm-level decode pipelining / instruction-depth tuning** inside
      `matmul_packed4_group`: analyse whether the single-token-decode regime
      (per-expert `me`≈1 at conc-4) is latency-bound and add deeper ILP /
      software pipelining of the group-decode + FMA stream rather than more
      N-slicing fan-out. This is the likeliest path toward lk-moe's ~177 TPOT.
- [ ] **Prefill batch pipeline** (TTFT gap is the largest, ~2.8x): improve the
      large-M batch-parallel split granularity, memory movement, and
      Gate/Up/Down 3-stage pipeline scheduling vs lk-moe. This is the biggest
      single-request end-to-end latency lever.
- [ ] **Per-socket weight replication + NUMA-nearest dispatch**: replicate the
      weight snapshot once per NUMA node so every worker's block reads are
      local DRAM; dispatch each expert's jobs to the node owning that copy.
      Distinct from page-interleave (which failed); only worth it if a probe
      shows the weight stream becomes bandwidth-bound at higher ILP.

### Lower value / explored (do not blindly retry without a new idea)
- [ ] A2+B0 fusion — **measured regression**, revisit only with a barrier-cost
      probe first.
- [ ] NUMA page-level interleave — **measured regression**; prefer per-socket
      replication over fine-grained interleave.
- [ ] Barrier-count reduction: pool is not currently bandwidth-saturated; probe
      pool/barrier idle time before spending effort here.

### Guardrails / constraints (must hold for every change)
- [ ] Correctness gate: 50/50 pass, 0 failed, 0 stall on the 50-prompt / conc-4
      bench; smoke "4\n\n4" correct.
- [ ] Prod **GPU0/GPU1 port-8070 EngineCore (external) must never be touched**;
      test only on **GPU2 / port 8071**. Kill servers only via explicit
      `kill <pid>` (never `pkill` a pattern present in one's own cmdline).
- [ ] Node placement of `w13_`/`w2_` snapshots: default constructor-node
      contiguous placement is the measured optimum today.
- [ ] Never copy lk-moe binaries/code; only reference its approach and copy
      ktransformers/lktransformers (Apache-2.0).

---

## How to run / verify
- Env: conda `xiaotumoe-vllm` at `/home/user/anaconda3/envs/xiaotumoe-vllm`
  (`export ENV=...`; use `$ENV/bin/python`; `PYTHONPATH=/home/user/lvllm/xiaotu-moe`).
- Test on GPU2 / TP1 / port 8071 / ml 8192 / gpu-mem 0.5; workload 50 prompts,
  conc 4, seed 0.
- Each kernel change requires killing/restarting the 8071 server
  (~170 s model load + ~5 min warm-up ≈ 213–330 s to READY).
