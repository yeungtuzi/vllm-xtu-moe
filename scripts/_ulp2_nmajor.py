import os,sys,torch
sys.path.insert(0,'/home/user/lvllm/vllm-xiaotu-moe')
from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer
E,H,I,K,T=8,256,128,4,32
dev=torch.device("cuda:0"); torch.cuda.set_device(0)
g=torch.Generator().manual_seed(0)
w13=torch.randint(0,256,(E,2*I,H//2),generator=g,dtype=torch.uint8)
s13=torch.randint(122,132,(E,2*I,H//32),generator=g,dtype=torch.uint8)
w2 =torch.randint(0,256,(E,H,I//2),generator=g,dtype=torch.uint8)
s2 =torch.randint(122,132,(E,H,I//32),generator=g,dtype=torch.uint8)
x=(torch.randn(T,H,generator=g,dtype=torch.float32)*0.02).to(torch.bfloat16).to(dev)
ids=torch.randint(0,E,(T,K),generator=g,dtype=torch.int32).to(dev)
tw=torch.rand(T,K,generator=g); tw=(tw/tw.sum(-1,keepdim=True)).to(dev)
def run(nm):
    os.environ["XIAOTU_GPF_NMAJOR"]=nm
    y=gpu_moe_layer(x,ids,tw,*[t.to(dev) for t in (w13,s13,w2,s2)],H=H,I=I,K=K,device=dev)
    torch.cuda.synchronize(); return y
A=run("0"); B=run("1")
d=(A.float()-B.float()).abs()
mag=A.float().abs().clamp(min=1e-6)
r=(d/(2.0**(torch.floor(torch.log2(mag))-7)))
bad=(r>8)
print(f"  >8 ulp 的元素 {int(bad.sum())}/{A.numel()}")
rows=bad.any(dim=1).nonzero().flatten().tolist()
cols=bad.any(dim=0).nonzero().flatten().tolist()
print(f"  涉及行({len(rows)}):{rows[:20]}")
print(f"  涉及列({len(cols)}):{cols[:24]}")
# 列是否有规律(2 的幂 / 32 的倍数 / H 附近)?
print(f"  列的模 2 = {sorted(set(c%2 for c in cols))} · 模 4 = {sorted(set(c%4 for c in cols))} · 模 32 = {sorted(set(c%32 for c in cols))}")
print(f"  行是否集中在某些 token?频次前 6 = {sorted(((rows.count(v),v) for v in set(rows)),reverse=True)[:6]}")
# 这些行对应的专家
bad_rows=sorted(set(rows))
print(f"  这些行的专家 id(token→expert 每 token 取 top1)= {[int(ids[t,0]) for t in bad_rows[:12]]}")
# ⭐ 最差点的【绝对量级】(判定是不是小量级上的抵消噪声)
flat=d.flatten(); top=flat.topk(8).indices
import numpy as np
for i in top.tolist():
    rr,cc = i//H, i%H
    print(f"    最差: 行{rr} 列{cc}  |y0|={A[rr,cc].float().item():.6e}  |y1|={B[rr,cc].float().item():.6e}  |Δ|={d[rr,cc].item():.6e}  相对该点={d[rr,cc].item()/max(abs(A[rr,cc].float().item()),1e-12):.3f}")
print(f"  ⇒ 若最差点的 |y| 都很小 ⇒ 是【抵消噪声】,不是 bug ✓")
print(f"  全矩阵 |y| 的 1% 分位 = {A.float().abs().flatten().kthvalue(max(1,int(0.01*A.numel()))).values:.3e}")
