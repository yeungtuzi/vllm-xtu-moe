#!/usr/bin/env python3
"""引擎 CUDA graph 捕获回归测试(与 vLLM 解耦的独立复现)。

背景:开启 `--enforce-eager=false` 后,vLLM 的 breakable CUDAGraph 捕获在
`capture_end()` 处报 `markCaptureEnd called with no captures in progress`,
需要判断是引擎 binding 的捕获路径有问题,还是 vLLM 的 piecewise 捕获与
我们的 host 回调不兼容。这个脚本只做最小复现:

  1) 随机 MXFP4 权重建引擎(E/H/I/K 与 DS-V4-Flash 一致);
  2) eager 跑一次 cpu_decode → 参考输出;
  3) torch.cuda.CUDAGraph 捕获同一调用,replay 两次;
  4) 比较 replay 结果与 eager 结果(应当逐位一致),并检查没有新增 CUDA 错误。

用法:  python3 scripts/engine_graph_test.py [qlen]
退出码 0 = 通过。
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
    dev = torch.device("cuda:0")
    rng = np.random.default_rng(7)

    def u8(shape):
        return torch.from_numpy(rng.integers(0, 256, shape, dtype=np.uint8)).contiguous()

    w13, w2 = u8((E, 2 * I, H // 2)), u8((E, H, I // 2))
    # 缩放用真实的 e8m0 指数范围(100..145),否则随机字节会指数爆炸成 NaN
    def sc(shape):
        return torch.from_numpy(rng.integers(100, 145, shape, dtype=np.uint8)).contiguous()
    s13, s2 = sc((E, 2 * I, H // GK)), sc((E, H, I // GK))

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
    if hasattr(eng, "prepare_decode_buffers"):
        eng.prepare_decode_buffers(max(qlen, 64), K)   # ★ 必须在捕获之前
    else:
        print("[warn] engine has no prepare_decode_buffers (old .so)")

    torch.manual_seed(0)
    # 真实量级的 bf16 输入(随机 uint16 位型会含大量 NaN/Inf,导致输出不可比)
    hid = (torch.randn(qlen, H, device=dev) * 0.1).to(torch.bfloat16).view(torch.uint16).contiguous()
    ids = torch.randint(0, E, (qlen, K), dtype=torch.int32, device=dev).contiguous()
    wts = torch.rand(qlen, K, dtype=torch.float32, device=dev).contiguous()
    out = torch.zeros(qlen, H, dtype=torch.float32, device=dev)
    def call():
        # ★ 必须在**调用时**取当前流:捕获期间 torch 会把当前流切成 capture stream,
        #   提前缓存的默认流会让 graph 变成空的(实测 "The CUDA Graph is empty")。
        stream = torch.cuda.current_stream()
        eng.cpu_decode(stream.cuda_stream, qlen, K, hid.data_ptr(),
                       ids.data_ptr(), wts.data_ptr(), out.data_ptr())

    call()
    torch.cuda.synchronize()
    ref = out.clone()
    print(f"[eager] ok, out.sum={ref.sum().item():.6f}")

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            call()
    except Exception as exc:                       # noqa: BLE001
        print(f"[FAIL] capture raised: {type(exc).__name__}: {exc}")
        return 1
    print("[capture] ok")
    print(f"[capture] out.sum(after capture)={out.sum().item():.6f}")

    for i in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        ok = bool(torch.allclose(out, ref, rtol=0, atol=0, equal_nan=True))
        print(f"[replay {i}] identical={ok}  max|Δ|={(out - ref).abs().max().item():.3e}")
        if not ok:
            return 2
    err = torch.cuda.get_device_properties(0) and None
    print("[PASS] graph capture + 2 replays identical to eager")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
