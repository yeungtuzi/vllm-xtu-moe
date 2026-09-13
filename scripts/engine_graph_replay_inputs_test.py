#!/usr/bin/env python3
"""cpu_decode 的 CUDA graph **重放正确性**测试:host 回调是否每次 replay 都重算?

为什么必须有这个测试:
  `--compilation_config.cudagraph_mode FULL_DECODE_ONLY` 能不能用,取决于
  `binding.cpp` 里那个 `cudaLaunchHostFunc` 宿主回调在**每次 graph replay** 时
  是否重新执行 CPU MoE。如果它只在**捕获**时跑了一次,那么 replay 只会把
  `out_gpu` 写成捕获时那份陈旧结果 —— 输出是**静默错误**的(不报错、不变慢,
  只是答案错),这比崩溃危险得多。

  `scripts/engine_graph_test.py` 只验证了"输入不变时 replay == eager",这对
  "回调有没有重算"是**无区分力**的(陈旧结果也 == eager)。本脚本在每次 replay
  前**原地改写输入缓冲**,再用一次 eager 调用取该输入的参考值,逐一比对。

用法:  python3 scripts/engine_graph_replay_inputs_test.py [qlen] [rounds]
退出码 0 = 通过(回调确实每次重算)。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xiaotu_moe  # noqa: E402

E, H, I, K, GK = 256, 4096, 2048, 6, 32


def main() -> int:
    qlen = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    dev = torch.device("cuda:0")
    rng = np.random.default_rng(7)

    w13 = torch.from_numpy(rng.integers(0, 256, (E, 2 * I, H // 2), dtype=np.uint8)).contiguous()
    w2 = torch.from_numpy(rng.integers(0, 256, (E, H, I // 2), dtype=np.uint8)).contiguous()
    s13 = torch.from_numpy(rng.integers(100, 145, (E, 2 * I, H // GK), dtype=np.uint8)).contiguous()
    s2 = torch.from_numpy(rng.integers(100, 145, (E, H, I // GK), dtype=np.uint8)).contiguous()

    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes, cfg.process_id, cfg.gpu_id = 1, 0, 0
    cfg.has_gate_proj = True
    cfg.expert_num, cfg.top_k = E, K
    cfg.hidden_size, cfg.intermediate_size = H, I
    cfg.max_batch_size, cfg.max_num_seqs = 8192, 256
    cfg.stride, cfg.group_min_len, cfg.group_max_len = 32, 10, 8192
    cfg.groupN, cfg.groupK, cfg.activation_type = 1, GK, 0
    cfg.use_gpu_prefill = False
    eng = xiaotu_moe.MOE_MXFP4(cfg, w13.data_ptr(), w2.data_ptr(),
                               s13.data_ptr(), s2.data_ptr(), 0, 0)
    eng.prepare_decode_buffers(max(qlen, 64), K)   # 必须在捕获之前

    gen = torch.Generator(device="cpu").manual_seed(0)

    def new_inputs(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h = (torch.randn(qlen, H, generator=g) * 0.1).to(torch.bfloat16).view(torch.uint16)
        i = torch.randint(0, E, (qlen, K), generator=g, dtype=torch.int32)
        w = torch.rand(qlen, K, generator=g, dtype=torch.float32)
        return h, i, w

    h0, i0, w0 = new_inputs(1)
    hid = h0.to(dev).contiguous()
    ids = i0.to(dev).contiguous()
    wts = w0.to(dev).contiguous()
    out = torch.zeros(qlen, H, dtype=torch.float32, device=dev)

    def call():
        # 必须在调用时取当前流(捕获期间 torch 会切到 capture stream)。
        eng.cpu_decode(torch.cuda.current_stream().cuda_stream, qlen, K,
                       hid.data_ptr(), ids.data_ptr(), wts.data_ptr(), out.data_ptr())

    # 用第 0 组输入捕获。
    call()
    torch.cuda.synchronize()
    captured_ref = out.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    print("[capture] ok")

    # 捕获后立刻 replay 一次:输入未变,应当等于捕获时的结果。
    hid.copy_(h0.to(dev).view(torch.uint16))
    ids.copy_(i0.to(dev))
    wts.copy_(w0.to(dev))
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    print(f"[sanity] same-input replay identical={bool(torch.equal(out, captured_ref))}")

    rc = 0
    for r in range(rounds):
        seed = 100 + r
        h, i, w = new_inputs(seed)
        hid.copy_(h.to(dev).view(torch.uint16))
        ids.copy_(i.to(dev))
        wts.copy_(w.to(dev))
        # 参考值:同输入下的一次 eager 调用。
        out.zero_()
        call()
        torch.cuda.synchronize()
        ref = out.clone()
        # 待测:同输入下 graph replay。
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        same = bool(torch.equal(out, ref))
        stale = bool(torch.equal(out, captured_ref))
        print(f"[round {r} seed={seed}] replay==eager:{same}  "
              f"replay==capture-time:{stale}  max|d|={(out - ref).abs().max().item():.3e}")
        if not same:
            rc = 2

    if rc:
        print("[FAIL] host 回调没有在 replay 时重算 —— FULL_DECODE_ONLY 会静默算错")
    else:
        print("[PASS] host 回调每次 replay 都用当前输入重算 CPU MoE")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
