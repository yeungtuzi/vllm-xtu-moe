#!/usr/bin/env python
"""Focused debug: K=1, weights=1 -> out[t] = f_{e_t}(x[t]); compare per token."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","2")
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer, torch_reference_layer, _E2M1

H, I, E = 64, 32, 4
T, K = 16, 1
dev = torch.cuda.current_device()
gc = torch.Generator(device='cpu').manual_seed(7)
gg = torch.Generator(device=dev).manual_seed(7)
w13 = torch.randint(0,16,(E,2*I,H//2),generator=gc,dtype=torch.uint8)
w2  = torch.randint(0,16,(E,H,I//2),generator=gc,dtype=torch.uint8)
s13 = torch.randint(120,127,(E,2*I,H//32),generator=gc,dtype=torch.uint8)
s2  = torch.randint(120,127,(E,H,I//32),generator=gc,dtype=torch.uint8)
x = torch.randn(T,H,generator=gg,dtype=torch.bfloat16,device=dev)
topk_ids = torch.randint(0,E,(T,K),dtype=torch.int64,device=dev)
tw = torch.ones(T,K,device=dev)
w13g=w13.to(dev);s13g=s13.to(dev);w2g=w2.to(dev);s2g=s2.to(dev)

got = gpu_moe_layer(x, topk_ids, tw, w13g, s13g, w2g, s2g, H=H, I=I, K=K, device=dev)
ref = torch_reference_layer(x, topk_ids, tw, w13g, s13g, w2g, s2g, H, I)
print("topk_ids:", topk_ids[:,0].tolist())
d = (got.float()-ref.float()).abs()
print("per-token max diff:", d.max(dim=1).values.tolist()[:16])
for t in range(min(4,T)):
    print(f"t={t} e={int(topk_ids[t,0])}")
    print("  got", got[t].float().tolist())
    print("  ref", ref[t].float().tolist())
