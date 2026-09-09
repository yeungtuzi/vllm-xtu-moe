#!/usr/bin/env python
"""Debug: dequant one weight row n=4 for expert e over all K; kernel vs torch."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","2")
import torch
import triton
import triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_xiaotu_moe import gpu_prefill as G

@triton.jit
def _deq_row(w13_ptr, w13_ld, s13_ptr, s13_ld, n, out_ptr, H: tl.constexpr, BK: tl.constexpr):
    # output [H] fp32 = dequant of row n
    out = tl.zeros((H,), dtype=tl.float32)
    k = tl.arange(0, H)
    byte = tl.load(w13_ptr + n * w13_ld + (k >> 1))   # [H] bytes (vectorized)
    nib = tl.where((k & 1) == 0, byte & 0x0F, (byte >> 4) & 0x0F)
    val = G._dequant_e2m1_nibble(nib)
    sc = tl.exp2(tl.load(s13_ptr + n * s13_ld + (k >> 5)).to(tl.float32) - 127.0)
    tl.store(out_ptr + k, val * sc)

H = 64
dev = torch.cuda.current_device()
gc = torch.Generator(device='cpu').manual_seed(7)
w13 = torch.randint(0,16,(4,2*32,H//2),generator=gc,dtype=torch.uint8).to(dev)  # E=4, 2I=64
s13 = torch.randint(120,127,(4,2*32,H//32),generator=gc,dtype=torch.uint8).to(dev)
n = 4
e = 0
out = torch.zeros(H, dtype=torch.float32, device=dev)
_deq_row[(1,)](w13[e], w13[e].stride(0), s13[e], s13[e].stride(0), n, out, H=H, BK=32)

# torch reference dequant of row n
lo=(w13[e,n]&0x0F).to(torch.int64); hi=((w13[e,n]>>4)&0x0F).to(torch.int64)
vals=G._E2M1.to(dev)
even=vals[lo]; odd=vals[hi]
w = torch.stack([even,odd],dim=-1).reshape(H)
sc = torch.exp2(s13[e,n].float()-127.0).repeat_interleave(32)[:H]
ref = w*sc
d=(out-ref).abs()
print("dequant row n=4: max_abs", d.max().item())
print("  kernel", out[:8].tolist())
print("  torch ", ref[:8].tolist())
