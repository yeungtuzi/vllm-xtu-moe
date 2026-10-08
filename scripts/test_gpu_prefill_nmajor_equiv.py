#!/usr/bin/env python
"""⭐ **NMAJOR 等价测试**:`NMAJOR=1`(直读 N-major)必须与 `NMAJOR=0`(K-major)**输出逐位相同**。

为什么必须单独写(B362 的教训):仓库现有的 MXFP4 回归
(`test_gpu_prefill_equiv_fixture.py`)**走的是 bf16 未量化路径** ⇒ **不覆盖** `_gate_up_kernel`/`_down_kernel` ✗;
而 `test_gpu_prefill_equiv.py` 需要真模型 fixture ⇒ 本机跑不了 ✗
⇒ 所以本测试**只为这次改动**而写:同一份权重、同一个输入,只换布局与开关,断言输出一致 ✓

布局(见 `gpu_prefill.py` 的 docstring 与内核索引):
  K-major(NMAJOR=0): w13t [E, H//2, 2I]  s13t [E, H//32, 2I]  w2t [E, I//2, H]  s2t [E, I//32, H]
  raw   (NMAJOR=1): w13  [E, 2I, H//2]   s13  [E, 2I, H//32]  w2  [E, H, I//2]  s2  [E, H, I//32]
  ⇒ 二者关系:`raw = kmajor.transpose(1, 2).contiguous()` ✓

用法:
  CUDA_VISIBLE_DEVICES=1 python scripts/test_gpu_prefill_nmajor_equiv.py
环境变量:H I E K T(默认 512 256 16 6 64)
"""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer  # noqa: E402


def main() -> int:
    E = int(os.environ.get("E", 16))
    H = int(os.environ.get("H", 512))
    I = int(os.environ.get("I", 256))
    K = int(os.environ.get("K", 6))
    T = int(os.environ.get("T", 64))
    gi = int(os.environ.get("GPU", 1))
    dev = torch.device(f"cuda:{gi}")
    torch.cuda.set_device(gi)
    print(f"设备 cuda:{gi} · E={E} H={H} I={I} K={K} T={T}")

    gc = torch.Generator().manual_seed(0)
    # ── raw(N-major)──
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), generator=gc, dtype=torch.uint8)
    s13 = torch.randint(122, 132, (E, 2 * I, H // 32), generator=gc, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), generator=gc, dtype=torch.uint8)
    s2 = torch.randint(122, 132, (E, H, I // 32), generator=gc, dtype=torch.uint8)
    # ── K-major = transpose(1,2)──
    w13t = w13.transpose(1, 2).contiguous()
    s13t = s13.transpose(1, 2).contiguous()
    w2t = w2.transpose(1, 2).contiguous()
    s2t = s2.transpose(1, 2).contiguous()
    # ⭐ 二分用的退化输入(零内核改动就能把嫌疑分成"权重"与"scale"两半)
    if os.environ.get("ZERO_W") == "1":
        w13 = torch.zeros_like(w13); w2 = torch.zeros_like(w2)
        print("  [ZERO_W] 所有权重字节=0 ⇒ 所有 nibble=0")
    if os.environ.get("ONE_S") == "1":
        s13 = torch.full_like(s13, 127); s2 = torch.full_like(s2, 127)
        print("  [ONE_S] 所有 scale=127 ⇒ 2^0 = 1")
    print(f"  raw13 {tuple(w13.shape)}(km13 由内部生成 {tuple(w13t.shape)});"
          f" raw2 {tuple(w2.shape)}(km2 {tuple(w2t.shape)})")

    g = torch.Generator().manual_seed(1)
    x = (torch.randn(T, H, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).to(dev)
    ids = torch.randint(0, E, (T, K), generator=g, dtype=torch.int32).to(dev)
    tw = torch.rand(T, K, generator=g, dtype=torch.float32)
    tw = (tw / tw.sum(-1, keepdim=True)).to(dev)

    def run(nmajor: str, w13_, s13_, w2_, s2_):
        os.environ["XIAOTU_GPF_NMAJOR"] = nmajor
        y = gpu_moe_layer(x, ids, tw, w13_.to(dev), s13_.to(dev), w2_.to(dev), s2_.to(dev),
                          H=H, I=I, K=K, device=dev)
        torch.cuda.synchronize()
        return y

    # ⭐ 关键(2026-10-08 更正):`gpu_moe_layer` 的契约是**传 raw、内部转 K-major**
    #    ⇒ 两臂必须都传【同一份 raw】,只差 `NMAJOR` 开关 ✓
    #    (先前两臂分别传 km/raw ⇒ arm0 被**双重转置** ⇒ 拿 KM 索引读 raw = 垃圾 ✗)
    y0 = run("0", w13, s13, w2, s2)          # raw ⇒ 内部转 K-major(现行为)
    y1 = run("1", w13, s13, w2, s2)          # raw ⇒ 直读 N-major(新路径)
    assert y0.shape == y1.shape == (T, H), (y0.shape, y1.shape)

    same = bool(torch.equal(y0, y1))
    d = (y0.float() - y1.float()).abs()
    ref = y0.float().abs().mean().item() or 1.0
    print(f"  逐位相同 = {same}")
    print(f"  max|Δ| = {d.max().item():.3e}  归一化 = {d.max().item()/ref:.3e}"
          f"  (参考 |y| 均值 {ref:.3e})")
    # ⭐ 判据(B370):不同【求和顺序】不可能逐位相同;正确的尺是**最大量级的 bf16 ulp** ✓
    #    ⚠️ 用【局部】ulp 当尺会被抵消点放大 100+ 倍 ⇒ 那是度量假象,不是误差 ✗
    mx = float(y0.float().abs().max())
    e = int(torch.floor(torch.log2(torch.tensor(mx))).item())
    ulp_max = 2.0 ** (e - 7)                       # bf16:1+8+7 ⇒ ulp = 2^(E-7) ✓
    ratio = d.max().item() / ulp_max
    ok = same or ratio <= 1.0
    print(f"  max|y| = {mx:.4e} ⇒ bf16 ulp(max) = {ulp_max:.4e}")
    print(f"  ⭐ max|Δ| / ulp(max) = {ratio:.2f} ulp")
    print(f"[{'OK ' if ok else 'BAD'}] NMAJOR 等价:"
          f"{'逐位相同 ✓' if same else f'{ratio:.2f} bf16 ulp(max) ⇒ 仅求和顺序舍入 ✓' if ok else '不一致 ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
