#!/usr/bin/env python
"""线程池丢票竞态(R113/§501)**快速复现器**。

背景:丢票表现为"`remaining_` 永不归零 ⇒ 调用方卡满 300s 才 `abort()`"(numa_pool.hpp 的看门狗)。
这个信号太慢,没法在压力测试里反复触发。所以配合 `XIAOTU_MOE_POOL_DEADLINE_MS=<小值>`
把检测时间降到毫秒级,再用**大量短调用**不停地冲那个"发布 ↔ 领票"窗口:
**进程非 0 退出 = 这次跑触发了竞态**。

为什么用合成权重:`cpu_prefill` 的池调用次数只取决于 (B, K, 活跃专家数, 线程数),
与权重数值无关;用小的 E/H/I 让单次调用做到毫秒级 ⇒ 每秒能冲上百次窗口。

用法:
  XIAOTU_MOE_POOL_DEADLINE_MS=1000 ENG=xiaotu XIAOTU_MOE_THREADS=120 \
    python scripts/stress_pool.py --iters 20000 --bs 1 --k 2 --e 64
  # 批量:N 个进程 × T 秒,数 aborts
  bash scripts/stress_pool.sh 8 60
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "xiaotu_moe", "build"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=1)      # 每次调用的 token 数
    ap.add_argument("--k", type=int, default=2)       # topk
    ap.add_argument("--e", type=int, default=64)      # 专家数
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--i", type=int, default=512)
    ap.add_argument("--dedup", type=int, default=0, help=">0 = 固定这么多活跃专家")
    ap.add_argument("--report", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import xiaotu_moe as m

    E, H, I, GK = args.e, args.h, args.i, 32
    rng = np.random.default_rng(args.seed)
    w13 = rng.integers(0, 256, size=(E, 2 * I, H // 2), dtype=np.uint8)
    w2 = rng.integers(0, 256, size=(E, H, I // 2), dtype=np.uint8)
    s13 = np.full((E, 2 * I, H // GK), 127, dtype=np.uint8)
    s2 = np.full((E, H, I // GK), 127, dtype=np.uint8)

    cfg = m.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = args.k
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 8192
    cfg.max_num_seqs = 256
    cfg.stride = 32
    cfg.group_min_len = 10
    cfg.group_max_len = 4096 + 128
    cfg.groupN = 1
    cfg.groupK = 32
    cfg.activation_type = 0
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)

    x = rng.standard_normal((args.bs, H)).astype(np.float16).view(np.uint16)
    out = np.zeros((args.bs, H), dtype=np.float32)
    if args.dedup > 0:
        picks = rng.integers(0, E, size=(args.dedup,)).astype(np.int32)
        ids = np.tile(picks, (args.bs * args.k) // args.dedup + 1)[: args.bs * args.k]
        ids = ids[rng.permutation(args.bs * args.k)].reshape(args.bs, args.k).astype(np.int32)
    else:
        ids = rng.integers(0, E, size=(args.bs, args.k)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(args.bs, args.k)).astype(np.float32)

    print(f"[stress] variant={getattr(m, '__variant__', '?')} E={E} H={H} I={I} "
          f"B={args.bs} K={args.k} iters={args.iters} "
          f"deadline_ms={os.environ.get('XIAOTU_MOE_POOL_DEADLINE_MS', '300000')}", flush=True)
    t0 = time.perf_counter()
    for it in range(args.iters):
        eng.cpu_prefill(args.bs, args.k, ids, wts, x, out)
        if args.report and (it + 1) % args.report == 0:
            dt = time.perf_counter() - t0
            print(f"[stress] it={it + 1} {dt:.1f}s {1000 * dt / (it + 1):.3f} ms/call", flush=True)
    dt = time.perf_counter() - t0
    print(f"[stress] DONE {args.iters} calls in {dt:.1f}s "
          f"({1000 * dt / args.iters:.3f} ms/call) — 无丢票", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
