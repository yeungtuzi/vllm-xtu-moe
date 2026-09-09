#!/usr/bin/env python
"""Debug: compare GPU gate+up (via new kernel) to torch, and inter."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","2")
import torch
import triton
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vllm_xiaotu_moe.gpu_prefill as G

H,I,E=64,32,4
T,K=16,1
dev=torch.cuda.current_device()
gc=torch.Generator(device='cpu').manual_seed(7); gg=torch.Generator(device=dev).manual_seed(7)
w13=torch.randint(0,16,(E,2*I,H//2),generator=gc,dtype=torch.uint8).to(dev)
s13=torch.full((E,2*I,H//32),122,dtype=torch.uint8).to(dev)
x=torch.randn(T,H,generator=gg,dtype=torch.bfloat16,device=dev)
topk=torch.randint(0,E,(T,K),dtype=torch.int64,device=dev)

# torch gate+up per token (each single expert)
vals=G._E2M1.to(dev)
gates=torch.empty(T,I,dtype=torch.bfloat16,device=dev); ups=torch.empty(T,I,dtype=torch.bfloat16,device=dev)
W13 = None
for e in range(E):
    idx=(topk[:,0]==e).nonzero().squeeze(1)
    if idx.numel()==0: continue
    lo=(w13[e]&0x0F).to(torch.int64); hi=((w13[e]>>4)&0x0F).to(torch.int64)
    w=torch.stack([vals[lo],vals[hi]],dim=-1).reshape(2*I,H)
    sc=torch.exp2(s13[e].float()-127.0).repeat_interleave(32,dim=1)[:,:H]
    W=(w*sc).to(x.dtype)
    gates[idx]=x[idx]@W[:I].t()
    ups[idx]=x[idx]@W[I:].t()

# kernel path
w13_t=G._dequant_layer_device(w13,s13,H,I,dev)   # [E,H,2I]
tok,wts,seg_start,A=G._build_segmentation(topk,topk.to(torch.float32),E,dev)
inter=torch.zeros((A,2*I),dtype=torch.bfloat16,device=dev)
BM,BN,BK=16,32,64
for e in range(E):
    st,en=int(seg_start[e]),int(seg_start[e+1]); M=en-st
    if M==0: continue
    G._gate_up_bf16_kernel[(triton.cdiv(2*I,BN),triton.cdiv(M,BM))](
        x,x.stride(0),tok[st:en],w13_t[e],w13_t.stride(1),inter,inter.stride(0),
        M=M,base=st,H=H,BM=BM,BN=BN,BK=BK)
print("seg:",seg_start.tolist())
gmask = inter[:,:I].float() - gates[tok]   # rows sorted by expert; gate per token
print("gate maxdiff vs torch (correct compare):", gmask.abs().max().item())
print("up   maxdiff vs torch:", (inter[:,I:].float()-ups).abs().max().item())
d=(inter[:,:I].float()-gates).abs()
flat=(d[0]).sort(descending=True)
g13=(inter[13,:I].float()-gates[13]).abs()
wc=(g13>0.5).nonzero().flatten().tolist()
print("t13 gate cols with diff>0.5:", wc)
for c in wc[:10]:
    print(f"  col{c}: got={inter[13,c].item():.4f} torch={gates[13,c].item():.4f}")
