#!/usr/bin/env python
"""`cpu_decode` 全链路 marshalling 微基准(不需要加载模型)。

动机:整机一次加载 ~20 分钟,而"每层 marshalling 比参考多 ~0.3 ms"这种问题
需要在**分钟级**反馈里迭代。本脚本用真实的一层权重造 N 个引擎(默认 8),
然后按服务里的方式**轮转**调用 `cpu_decode`(GPU bf16 输入 → D2H → CPU MoE
→ H2D fp32 输出),给出:

  * µs/layer(每次调用的墙钟)
  * ms/pass (N 层 = 一个"token")
  * 两种模式:
      - serial : 每次调用后 torch.cuda.synchronize()(量"串行往返"上限)
      - pipe   : 全部 enqueue 完再一次 sync(量"纯入队开销";与 serial 的差
                 就是 GPU/D2H/H2D 与 CPU 计算之间**没有重叠**的那部分)

用法:
  PATH=$ENV/bin:$PATH XIAOTU_LAYER1_NPZ=<ckpt_dir> NENGINES=8 REP=20 \
    python scripts/bench_cd_plumbing.py
环境变量:
  NENGINES(默认 8) REP(默认 20) LAYER(默认 3) THREADS(引擎线程) MODE=serial|pipe|both
  XIAOTU_CD_TIMING=1 时引擎自己会打印 period/compute(engine/ep)/rest 拆分
"""
import glob
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

H, I, GK, E, K = 4096, 2048, 32, 256, 6


def load_layer(ckpt: str, layer: int):
    shard = None
    for p in sorted(glob.glob(os.path.join(ckpt, "model-*.safetensors"))):
        with open(p, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hl))
        if f"layers.{layer}.ffn.experts.0.w1.weight" in hdr:
            shard = p
            break
    assert shard, f"shard with layers.{layer} not found under {ckpt}"
    from safetensors import safe_open
    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
    s2 = np.empty((E, H, I // GK), dtype=np.uint8)
    with safe_open(shard, "pt") as f:
        def g(e, sfx):
            return f.get_tensor(f"layers.{layer}.ffn.experts.{e}.{sfx}").cpu()
        for e in range(E):
            w13[e, 0:I] = g(e, "w1.weight").contiguous().numpy().copy()
            w13[e, I:2 * I] = g(e, "w3.weight").contiguous().numpy().copy()
            w2[e] = g(e, "w2.weight").contiguous().numpy().copy()
            s13[e, 0:I] = g(e, "w1.scale").contiguous().view(__import__("torch").uint8).numpy().copy()
            s13[e, I:2 * I] = g(e, "w3.scale").contiguous().view(__import__("torch").uint8).numpy().copy()
            s2[e] = g(e, "w2.scale").contiguous().view(__import__("torch").uint8).numpy().copy()
    return w13, w2, s13, s2


def main() -> int:
    import torch
    # ENGINE_MODULE=lk_moe 可在参考 env(lvllmds4-x)里跑**同一个微基准**做引擎对引擎对照
    _mod = os.environ.get("ENGINE_MODULE", "xiaotu_moe")
    xiaotu_moe = __import__(_mod)

    ckpt = os.environ.get("XIAOTU_LAYER1_NPZ", "")
    if not ckpt:
        raise SystemExit("set XIAOTU_LAYER1_NPZ=<dir with model-*.safetensors>")
    neng = int(os.environ.get("NENGINES", "8"))
    rep = int(os.environ.get("REP", "20"))
    layer = int(os.environ.get("LAYER", "3"))
    mode = os.environ.get("MODE", "both")
    qlen = int(os.environ.get("QLEN", "1"))

    w13, w2, s13, s2 = load_layer(ckpt, layer)

    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = int(os.environ.get("CFG_WORLD", "1"))
    cfg.process_id = int(os.environ.get("CFG_RANK", "0"))
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = K
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

    t0 = time.time()
    if _mod == "lk_moe":
        # 参考绑定的构造参数是**裸指针**(lk 侧传 .data_ptr());我们的绑定两者都收
        def _mk():
            return xiaotu_moe.MOE_MXFP4(cfg, w13.ctypes.data, w2.ctypes.data,
                                        s13.ctypes.data, s2.ctypes.data, 0, 0)
    else:
        def _mk():
            return xiaotu_moe.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    engines = [_mk() for _ in range(neng)]
    print(f"[plumb] module={_mod} variant={getattr(xiaotu_moe, '__variant__', 'n/a')} "
          f"engines={neng} built in {time.time()-t0:.1f}s", flush=True)

    dev = torch.device("cuda:0")
    stream = torch.cuda.current_stream().cuda_stream
    rng = np.random.default_rng(7)
    x = torch.frombuffer(bytearray(rng.integers(0, 256, size=qlen * H * 2, dtype=np.uint8).tobytes()),
                         dtype=torch.uint8).view(torch.bfloat16).reshape(qlen, H).to(dev)
    # DEDUP=N:所有 token 只用固定的 N 个专家(模拟"同一段文本连续 token 命中同一批专家",
    # 用来量 L3(768 MB)复用能不能突破 DRAM 墙)。
    dedup = int(os.environ.get("DEDUP", "0"))
    if dedup > 0:
        picks = rng.integers(0, E, size=(dedup,)).astype(np.int32)
        flat = np.tile(picks, (qlen * K) // dedup + 1)[: qlen * K].astype(np.int32)
        ids = torch.frombuffer(bytearray(flat.tobytes()), dtype=torch.int32).reshape(qlen, K).to(dev)
    else:
        ids = torch.frombuffer(bytearray(rng.integers(0, E, size=qlen * K, dtype=np.int32).tobytes()),
                               dtype=torch.int32).reshape(qlen, K).to(dev)
    wts = torch.rand(qlen, K, dtype=torch.float32, device=dev)
    out = torch.zeros(max(qlen, 64), H, dtype=torch.float32, device=dev)   # 行数要 >= qlen

    # GPU_MM>0:每层前插一个 GPU 矩阵乘,模拟服务里"该层注意力/dense"占用 GPU 的时间
    # (用来复现"96 个 worker 自旋 + GPU 工作"同时发生时 cpu_decode 往返变慢"的现象)
    mm_n = int(os.environ.get("GPU_MM", "0"))
    A = B = Cm = None
    if mm_n:
        A = torch.randn(mm_n, mm_n, dtype=torch.bfloat16, device=dev)
        B = torch.randn(mm_n, mm_n, dtype=torch.bfloat16, device=dev)
        Cm = torch.empty(mm_n, mm_n, dtype=torch.bfloat16, device=dev)
        torch.mm(A, B, out=Cm); torch.cuda.synchronize()

    def do_calls(sp):
        for e in engines:
            if mm_n:
                torch.mm(A, B, out=Cm)          # 模拟该层 GPU 侧工作
            e.cpu_decode(sp, qlen, K, x.data_ptr(), ids.data_ptr(), wts.data_ptr(), out.data_ptr())

    def one_pass(sync_each: bool):
        t = time.perf_counter()
        do_calls(stream)
        if sync_each:
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    # warmup
    for _ in range(3):
        one_pass(False)
    # DUMP=<path>:把最后一个引擎的输出写盘,用于 host-func / async 两条路径的逐位对拍
    _dump = os.environ.get("DUMP")
    if _dump:
        torch.cuda.synchronize()
        import numpy as _np
        _np.save(_dump, out.cpu().numpy())
    print(f"{'mode':>7} {'ms/pass':>9} {'us/layer':>9} {'t/s(N=layers)':>14}", flush=True)
    # GPU_GRAPH=1:把整串调用**捕获进一张 CUDA 图**再 replay —— 直接检验
    # "host-func 节点在图里会排空流水线(43 次/步)"这个假设(服务里就是图模式)。
    if os.environ.get("GPU_GRAPH"):
        sside = torch.cuda.Stream()
        sside.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sside):
            do_calls(sside.cuda_stream)
        torch.cuda.current_stream().wait_stream(sside)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            do_calls(torch.cuda.current_stream().cuda_stream)
        best = None
        for _ in range(rep):
            t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
            dt = time.perf_counter() - t
            best = dt if best is None else min(best, dt)
        print(f"{'graph':>7} {best*1e3:9.2f} {best*1e6/neng:9.1f} {neng/best:14.1f}", flush=True)
    for m in (["serial", "pipe"] if mode == "both" else [mode]):
        best = min(one_pass(m == "serial") for _ in range(rep))
        print(f"{m:>7} {best*1e3:9.2f} {best*1e6/neng:9.1f} {neng/best:14.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
