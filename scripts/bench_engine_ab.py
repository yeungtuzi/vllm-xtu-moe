#!/usr/bin/env python
"""CPU 引擎 A/B 基线(xiaotu-moe vs lk_moe)—— 目标第一条的验收工具。

同一同步 API(`cpu_prefill(B,K,ids,wts,x,out)`,numpy 与裸指针都零拷贝)、同一份真实层
权重、同线程数。**lk 必须给 `LK_THREADS`**(漏设会只用个位数线程 ⇒ 假慢 ~70×,见 R33)。

  # 我们
  ENG=xiaotu XIAOTU_MOE_THREADS=120 XIAOTU_LAYER1_NPZ=<model> BS=6 DEDUP=12 REP=60 \
    /home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python scripts/bench_engine_ab.py
  # lk(必须用它自己的 env 与线程变量)
  ENG=lk LK_THREADS=120 CUDA_VISIBLE_DEVICES=0 XIAOTU_LAYER1_NPZ=<model> BS=6 DEDUP=12 REP=60 \
    /home/user/anaconda3/envs/lvllmds4-x/bin/python scripts/bench_engine_ab.py

基线值(BS=6/K=6/THREADS=120/真实层权重):
  起点 2026-09-11:DEDUP=12 xiaotu **1.22** ms/层(124 GB/s)vs lk 0.57 ms/层(265 GB/s)
  现在(轮 67-73 之后):DEDUP=12 xiaotu **0.66-0.68** ms/层(229 GB/s、1.9 GB/s·线程)
                       DEDUP=23 xiaotu 0.82-0.85 ms/层 vs lk 0.67
  ⇒ 验收 ① `≤0.70 ms/层` **已达标**;`每线程 ≥2.2 GB/s` 在 DEDUP=12 差 15%(已完整归因:
  内核向量化维度,lane=K vs lk 的 lane=输出列;服务端真实形状 na≈32 时已达 2.92 GB/s·线程
  超过 lk)。结论鏈见 dev-docs/report/tuning/NOTES.md §113-§128。

**门禁**:`scripts/check_engine_aligned.sh` 一条命令跑"数值对拍 + 本基准"(R55:任何动
numa_pool 同步结构或内层循环的改动都必须先过对拍,轮 68/70 两次事故都是 bench 全过而对拍挂)。

原 docstring: CPU xiaotu engine microbench: per-layer cost vs batch size (real weights).

Used to A/B compiler flags for the native extension (report/bench_engine_flags.sh).
Pure CPU: engine.cpu_prefill = forward_many compute, no GPU copies.
Env: LAYER (default 3), BS (comma list), REP (default 3), THREADS.
"""
import glob
import json
import os
import sys
import time

import numpy as np
from safetensors import safe_open

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
ENG = os.environ.get("ENG", "xiaotu")
if ENG == "lk":
    # 【§622f】lk_moe 有**两个同名同版本但二进制不同**的安装(lvllm 与 lvllmds4-x,
    # md5 不同)。要做"与服务级 A/B 同一个二进制"的对比,必须能指定用哪一个 ⇒
    # 用 `LK_PY_PATH` 覆盖(留空则保持历史默认 lvllmds4-x)。
    _lkp = os.environ.get("LK_PY_PATH",
                         "/home/user/anaconda3/envs/lvllmds4-x/lib/python3.12/site-packages")
    if _lkp:
        sys.path.insert(0, _lkp)

MODEL = os.environ.get("XIAOTU_LAYER1_NPZ", "")
if not MODEL:
    raise SystemExit("set XIAOTU_LAYER1_NPZ=<real layer-1 npz fixture> (see docs/BENCHMARKS.md)")
# ---- 维度:**从 checkpoint 的 config.json 自动探测**(不再写死 V4 的 4096/2048)
# 为什么:写死 V4 维度后,指向 V4.1-Flash(H=5120/I=2304/E=384/40 层)会直接
# `ValueError: could not broadcast input array from shape (2304,2560) into shape (2048,2048)`
# —— 2026-09-17 做 NPS4 分片 A/B 时又踩一次(bench_cpu_engine.py 早已修过,本文件漏了)。
# 仍可用 env 覆盖:HID/I/E/K/GK/NLAYERS。
_tc = {}
_cfg_path = os.path.join(MODEL, "config.json")
if os.path.isfile(_cfg_path):
    try:
        with open(_cfg_path) as _f:
            _j = json.load(_f)
        _tc = _j.get("text_config", _j) or {}
    except Exception:  # noqa: BLE001
        _tc = {}
H = int(os.environ.get("HID", _tc.get("hidden_size", 4096)))
I = int(os.environ.get("I", _tc.get("moe_intermediate_size", 2048)))
E = int(os.environ.get("E", _tc.get("n_routed_experts", 256)))
K = int(os.environ.get("K", _tc.get("num_experts_per_tok", 6)))
GK = int(os.environ.get("GK", 32))
N_LAYERS = int(os.environ.get("NLAYERS", _tc.get("num_hidden_layers", 43)))
print(f"[dims] H={H} I={I} E={E} K={K} GK={GK} layers={N_LAYERS} (auto)")


def f32_to_bf16_bits(x):
    v = x.astype(np.float32).view(np.uint32)
    lsb = (v >> 16) & 1
    return ((v + 0x7FFF + lsb) >> 16).astype(np.uint16)



def _pf(engine, B, K, ids, wts, x, out):
    """lk 的绑定只收裸指针;我们的两种都收。"""
    if ENG == "lk":
        import torch as _t
        def P(a):
            return _t.from_numpy(np.ascontiguousarray(a)).data_ptr()
        engine.cpu_prefill(B, K, P(ids), P(wts), P(x), P(out))
    else:
        engine.cpu_prefill(B, K, ids, wts, x, out)

def main():
    import torch  # noqa: F401  (engine needs torch loaded)
    mod = __import__('lk_moe' if ENG == 'lk' else 'xiaotu_moe')

    layer = int(os.environ.get("LAYER", "3"))
    bs = [int(v) for v in os.environ.get("BS", "512,2048,8192").split(",")]
    rep = int(os.environ.get("REP", "3"))

    shard = None
    for p in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
        with open(p, "rb") as f:
            hl = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hl))
        if f"layers.{layer}.ffn.experts.0.w1.weight" in hdr:
            shard = p
            break
    assert shard, "shard not found"

    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
    s2 = np.empty((E, H, I // GK), dtype=np.uint8)
    with safe_open(shard, "pt") as f:
        def get(e, sfx):
            return f.get_tensor(f"layers.{layer}.ffn.experts.{e}.{sfx}").cpu()
        for e in range(E):
            w1 = get(e, "w1.weight").contiguous().numpy().copy()
            w3 = get(e, "w3.weight").contiguous().numpy().copy()
            w13[e, 0:I] = w1
            w13[e, I:2 * I] = w3
            w2[e] = get(e, "w2.weight").contiguous().numpy().copy()
            s13[e, 0:I] = get(e, "w1.scale").contiguous().view(torch.uint8).numpy().copy()
            s13[e, I:2 * I] = get(e, "w3.scale").contiguous().view(torch.uint8).numpy().copy()
            s2[e] = get(e, "w2.scale").contiguous().view(torch.uint8).numpy().copy()

    m = mod.load() if ENG != "lk" else mod
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 8192; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
    # NENGINES>1:像真实服务那样建多个引擎(每层一个),只对最后一个计时。
    # 目的:验证"服务内单次调用比单实例微基准慢 3x"是否来自**多实例的内存放置**
    # (服务里 43 层 x ~3.2 GB 分片 ≈ 139 GB,而微基准只有 3.2 GB)。
    neng = int(os.environ.get("NENGINES", "1"))
    keep = []
    for _ in range(max(1, neng)):
        if ENG == "lk":
            import torch as _t
            _w = [_t.from_numpy(x) if hasattr(x, "data_ptr") is False or not _t.is_tensor(x) else x
                  for x in (w13, w2, s13, s2)]
            _w = [_t.from_numpy(x) if not _t.is_tensor(x) else x for x in (w13, w2, s13, s2)]
            keep.append(m.MOE_MXFP4(cfg, _w[0].data_ptr(), _w[1].data_ptr(),
                                    _w[2].data_ptr(), _w[3].data_ptr(), 0, 0))
        else:
            keep.append(m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0))
    engine = keep[-1]
    # ROUNDROBIN=1:像真实服务那样**轮流**调用这 neng 个引擎(每层一个),这样每个
    # 引擎的 scratch 缓冲在一次调用后就被换掉 ⇒ scratch 常驻缓存的效果消失。
    # 用来验证"服务内单次调用比单实例微基准慢 2x"是否来自 scratch 变冷。
    rr = int(os.environ.get("ROUNDROBIN", "0"))
    print(f"[cpu-bench] eng={ENG} variant={getattr(mod, '__variant__', '?')} layer={layer} nengines={len(keep)}",
          flush=True)

    rng = np.random.default_rng(7)
    ids = rng.integers(0, E, size=(1, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(1, K)).astype(np.float32)

    print(f"{'B':>6} {'ms/layer':>10} {'43L ms':>9} {'tok/s':>8} {'TFLOP/s':>8}",
          flush=True)
    for B in bs:
        x = f32_to_bf16_bits(rng.standard_normal((B, H)).astype(np.float32))
        out = np.zeros((B, H), dtype=np.float32)
        # Random per-token routing: with identical ids for every token the batch
        # collapses onto a handful of experts, which is NOT what the real model
        # does (256 experts x ~48 rows each) and changes the weight working set
        # from 3.2 GB (DRAM) to a few tens of MB (L3).
        #
        # DEDUP=N 控制"去重后的活跃专家数"(不设 = 全随机,36 个都不同)。
        # 真实解码步(qlen=6/topk=6、6 个 draft token 属同一段文本)实测去重后只有
        # ~12 个专家 ⇒ 每个专家平均被 3 个 token 命中(me=3)。内核只有 me>=4 才走
        # "解码一次喂 4 行"的快路径,me=1..3 走单行路径(每行都把权重重新解码一遍)
        # ⇒ 不设 DEDUP 就测不到真实情形。
        dedup = int(os.environ.get("DEDUP", "0"))
        if dedup > 0:
            picks = rng.integers(0, E, size=(dedup,)).astype(np.int32)
            ids_b = np.tile(picks, (B * K) // dedup + 1)[: B * K].reshape(B, K)
            ids_b = ids_b.reshape(-1)[rng.permutation(B * K)].reshape(B, K)
        else:
            ids_b = rng.integers(0, E, size=(B, K)).astype(np.int32)
        wts_b = rng.uniform(-1, 1, size=(B, K)).astype(np.float32)
        # 【§566】**预热次数可调**(默认 1)。为什么必须能调:实测同一份权重/同一线程数下,
        # 我们的引擎在**前几百次调用**里逐次变快(REP=30→1.15、60→1.02、120→0.85、
        # 240→0.77、480→0.74、960→0.71 ms/层),而 **lk 与 REP 无关(恒 0.58)**
        # ⇒ 短跑会把暂态灌进均值,导致"同日比值门"给出**假 FAIL**(REP=60 报 1.62×,
        # 同尺子 REP=480 才是 1.28×)。服务里引擎每秒被调用上万次,始终处于热态,
        # 所以**验收口径必须用热态**:`WARMUP=200 REP=60`,或大 REP。
        warmup = int(os.environ.get("WARMUP", "1"))
        for _ in range(max(0, warmup)):
            _pf(engine, B, K, ids_b, wts_b, x, out)
        t0 = time.perf_counter()
        if rr:
            for i in range(rep):
                _pf(keep[i % len(keep)], B, K, ids_b, wts_b, x, out)
        else:
            for _ in range(rep):
                _pf(engine, B, K, ids_b, wts_b, x, out)
        dt = (time.perf_counter() - t0) / rep
        # per token: K experts x (2*I*H + H*I) MACs
        flop = B * K * (2.0 * I * H + H * I) * 2 / 1e12
        print(f"{B:6d} {dt*1e3:10.2f} {dt*1e3*N_LAYERS:9.1f} "
              f"{B/(dt*N_LAYERS):8.1f} {flop/dt:8.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
