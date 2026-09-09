#!/usr/bin/env python
"""Debug inter (gate+up) kernel vs torch for a single expert, K=1."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","2")
import torch
import triton
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_xiaotu_moe import gpu_prefill as G
from vllm_xiaotu_moe.gpu_prefill import _gate_up_kernel

H, I, E = 64, 32, 4
T, K = 16, 1
dev = torch.cuda.current_device()
gc = torch.Generator(device='cpu').manual_seed(7)
gg = torch.Generator(device=dev).manual_seed(7)
w13 = torch.randint(0,16,(E,2*I,H//2),generator=gc,dtype=torch.uint8).to(dev)
s13 = torch.randint(120,127,(E,2*I,H//32),generator=gc,dtype=torch.uint8).to(dev)
x = torch.randn(T,H,generator=gg,dtype=torch.bfloat16,device=dev)
topk_ids = torch.randint(0,E,(T,K),dtype=torch.int64,device=dev)

# torch gate+up for all tokens (per their single expert)
lo=(w13&0x0F).to(torch.int64); hi=((w13>>4)&0x0F).to(torch.int64)
vals=G._E2M1.to(dev)
gates = torch.empty(T,I,dtype=torch.bfloat16,device=dev)
ups = torch.empty(T,I,dtype=torch.bfloat16,device=dev)
for e in range(E):
    idx = (topk_ids[:,0]==e).nonzero().squeeze(1)
    if idx.numel()==0: continue
    w_e = torch.stack([vals[lo[e,:I]], vals[hi[e,:I]]],dim=-1).reshape(I,H).float()
    w_u = torch.stack([vals[lo[e,I:]], vals[hi[e,I:]]],dim=-1).reshape(I,H).float()
    sc_g = torch.exp2(s13[e,:I].float()-127.0).repeat_interleave(32,dim=1)[:,:H]
    sc_u = torch.exp2(s13[e,I:].float()-127.0).repeat_interleave(32,dim=1)[:,:H]
    w_g = (w_e*sc_g).to(x.dtype); w_u=(w_u*sc_u).to(x.dtype)
    gates[idx] = x[idx]@w_g.t()
    ups[idx] = x[idx]@w_u.t()

# Build segmentation + run kernel for all experts, then read inter
from vllm_xiaotu_moe.gpu_prefill import _build_segmentation
idsc, twsc, seg_start, A = _build_segmentation(topk_ids, topk_ids.to(torch.float32), E, dev)
print("seg_start:", seg_start.tolist())
inter = torch.zeros((A,2*I),dtype=torch.bfloat16,device=dev)
BM,BN,BK=16,32,32
w13f=w13; s13f=s13
for e in range(E):
    st,en = int(seg_start[e]), int(seg_start[e+1])
    M=en-st
    if M==0: continue
    _gate_up_kernel[(H//BN, triton.cdiv(M,BM))](
        x, x.stride(0), idsc[st:en], w13f[e], w13f.stride(0), s13f[e], s13f.stride(0),
        inter, inter.stride(0), M=M, base=st, H=H, I=I, BM=BM, BN=BN, BK=BK,
    )
print("[gate_up debug] per-token max|got-ref| gate, up:")
got_gate = inter[:, :I].float()
got_up = inter[:, I:].float()
print("  gate maxdiff", (got_gate-gates).abs().max().item(), " up maxdiff", (got_up-ups).abs().max().item())
print("  gate t0 cols with abs>1e3:", (got_gate[0].abs()>1e3).nonzero().flatten().tolist())
bad = (got_gate[0].abs()>1e3).nonzero().flatten()
for b in bad.tolist()[:6]:
    print(f"    col {b}: got={got_gate[0,b].item():.3e} torch={gates[0,b].item():.3e}")
