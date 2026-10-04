#!/usr/bin/env python3
"""W4A8 离线 A/B —— 用**结构相似替代模型** Qwen3.8-Flash-Next-MXFP4-FP8 的真实权重。

为什么用它(用户 2026-10-04 明令 + AGENTS.md §5):
  * 与 DS4.1F 同构:路由专家 + 主机侧大表;而且它的专家张量**就是引擎的规范布局**:
      gate_up_proj_packed [E, 2I, H/2] U8   ← 引擎的 w13
      gate_up_proj_scale  [E, 2I, H/32] U8  ← 引擎的 s13(e8m0, groupK=32)
      down_proj_packed    [E, H, I/2] U8    ← 引擎的 w2
      down_proj_scale     [E, H, I/32] U8   ← 引擎的 s2
    ⇒ **无需重排**;一层仅 ~1.34 GB(全量模型的 1/3),读取 I/O 也小
  * 形状:E=512 H=2560 I=640 topk=10,layer 6。ALIGN 要求 K%64==0: 2560%64=0 ✓、640%64=0 ✓

⚠️ 资源纪律(AGENTS.md §4):
  * **探测阶段在沙箱内即可,不起任何服务** ⇒ 与生产零竞争
  * 调用方负责 `taskset` + `nice -n 19` + `XIAOTU_MOE_THREADS<=64`
  * 本脚本**不写** `oom_score_adj`(无需:它不加载巨模型)

用法:
  XIAOTU_MOE_THREADS=72 taskset -c 120-191 nice -n 19 \
    python dev-docs/w4a8-assets/qfn_bench.py --tokens 8192 --iters 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("XIAOTU_MOE_NO_AUTO_EP", "1")
sys.path.insert(0, "/home/user/lvllm/vllm-xiaotu-moe")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import xiaotu_moe  # noqa: E402

QFN = os.path.expanduser(
    "~/.cache/modelscope/models/Qwen3.8-Flash-Next-MXFP4-FP8"
)
KEYS = ("gate_up_proj_packed", "gate_up_proj_scale",
        "down_proj_packed", "down_proj_scale")


def load_layer(layer: int, tp: int = 1, rank: int = 0):
    wm = json.load(open(os.path.join(QFN, "model.safetensors.index.json")))["weight_map"]
    base = f"model.language_model.layers.{layer}.mlp.experts."
    need = {n: base + n for n in KEYS}
    byfile: dict[str, list] = {}
    for n, k in need.items():
        byfile.setdefault(wm[k], []).append((n, k))
    got: dict[str, torch.Tensor] = {}
    for f, items in byfile.items():
        with safe_open(os.path.join(QFN, f), "pt") as g:
            for n, k in items:
                got[n] = g.get_tensor(k).contiguous()
    w13 = got["gate_up_proj_packed"]                    # [E, 2I, H/2]
    s13 = got["gate_up_proj_scale"].view(torch.uint8)   # [E, 2I, H/32]
    w2 = got["down_proj_packed"]                        # [E, H, I/2]
    s2 = got["down_proj_scale"].view(torch.uint8)       # [E, H, I/32]
    if tp > 1:
        # 与 uva_extract_layer.py 的权威切片规则一致(见该文件注释)
        E, n2, Hh = w13.shape
        I_full = n2 // 2
        sh = I_full // tp
        r = rank
        w13 = torch.cat([w13[:, r * sh:(r + 1) * sh, :],
                         w13[:, I_full + r * sh:I_full + (r + 1) * sh, :]], dim=1).contiguous()
        s13 = torch.cat([s13[:, r * sh:(r + 1) * sh, :],
                         s13[:, I_full + r * sh:I_full + (r + 1) * sh, :]], dim=1).contiguous()
        w2 = w2[:, :, (r * (w2.shape[2] // tp)):((r + 1) * (w2.shape[2] // tp))].contiguous()
        s2 = s2[:, :, (r * (s2.shape[2] // tp)):((r + 1) * (s2.shape[2] // tp))].contiguous()
    return w13, s13, w2, s2


def build(w13, s13, w2, s2, tp: int, rank: int, qlen_max: int, topk: int):
    E = int(w13.shape[0])
    H = int(w13.shape[2]) * 2
    I = int(w13.shape[1]) // 2
    cfg = xiaotu_moe.MOEConfigV2()
    cfg.num_processes = tp
    cfg.process_id = rank
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = topk
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 8192
    cfg.max_num_seqs = 256
    cfg.stride = 32
    cfg.group_min_len = 10
    cfg.group_max_len = max(4096, qlen_max) + 128
    # QFN 的 config **没有 swiglu_limit** ⇒ plain SiLU(activation_type=0)
    cfg.activation_type = 0
    cfg.use_gpu_prefill = False
    cfg.groupN = 1
    cfg.groupK = 32
    eng = xiaotu_moe.MOE_MXFP4(cfg, w13.data_ptr(), w2.data_ptr(),
                               s13.data_ptr(), s2.data_ptr(), 0, 0)
    return eng, dict(E=E, H=H, I=I, cfg=cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--act-scale", type=float, default=1.0)
    ap.add_argument(
        "--routing", default="real",
        choices=("uniform", "real", "zipf"),
        help="real = 团队 SKEW 模型(一个热专家 + 均匀摊开)")
    ap.add_argument("--skew-hot", type=int, default=0)
    ap.add_argument("--skew-pool", type=int, default=0)
    ap.add_argument("--dump-out", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    t0 = time.perf_counter()
    w13, s13, w2, s2 = load_layer(a.layer, a.tp, a.rank)
    eng, k = build(w13, s13, w2, s2, a.tp, a.rank, a.tokens, a.topk)
    E, H = k["E"], k["H"]
    print(f"[qfn] {a.label} variant={xiaotu_moe.__variant__} "
          f"load+build={time.perf_counter()-t0:.1f}s "
          f"E={E} H={H} I={k['I']} topk={a.topk} qlen={a.tokens} "
          f"THREADS={os.environ.get('XIAOTU_MOE_THREADS','<unset>')}", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    qlen = a.tokens
    if a.routing == "uniform":
        ids = torch.randint(0, E, (qlen, a.topk), generator=g).to(torch.int32)
    else:
        tot = qlen * a.topk
        hot = a.skew_hot or tot // a.topk          # 默认:热专家吃 1/topk 的 assignment
        pool = a.skew_pool or max(1, int(E * 0.38))  # 默认按 V4.1 记录的 147.7/384 比例
        n_hot = min(hot, tot)
        nrest = tot - n_hot
        rest = torch.arange(1, pool + 1, dtype=torch.int64).repeat(nrest // pool + 1)[:nrest]
        flat = torch.cat([torch.zeros(n_hot, dtype=torch.int64), rest])
        ids = flat[torch.randperm(tot, generator=g)].reshape(qlen, a.topk).to(torch.int32)
    ids = ids.contiguous()
    wts = torch.rand((qlen, a.topk), generator=g).to(torch.float32).contiguous()
    hid = (torch.randn(qlen, H, generator=g) * a.act_scale).to(torch.bfloat16).contiguous()
    out = torch.zeros(qlen, H, dtype=torch.float32).contiguous()

    args6 = (qlen, a.topk, ids.data_ptr(), wts.data_ptr(), hid.data_ptr(), out.data_ptr())
    for _ in range(a.warmup):
        eng.cpu_prefill(*args6)
    ts = []
    for _ in range(a.iters):
        t = time.perf_counter()
        eng.cpu_prefill(*args6)
        ts.append(time.perf_counter() - t)
    ts.sort()
    med = ts[len(ts) // 2]
    flop = 2.0 * a.topk * 3.0 * H * k["I"] * qlen
    tag = f"route={a.routing}"
    if a.routing == "real":
        tag += f"(hot={a.skew_hot or qlen},pool={a.skew_pool or max(1,int(E*0.38))})"
    print(f"[qfn] {a.label} {tag} median={med*1e3:.4f} ms min={ts[0]*1e3:.4f} "
          f"max={ts[-1]*1e3:.4f} eff={flop/med/1e12:.5f} TFLOPS", flush=True)
    if a.dump_out:
        torch.save({"out": out.clone(), "qlen": qlen, "seed": a.seed}, a.dump_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
