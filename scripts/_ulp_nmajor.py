import os,sys,torch
sys.path.insert(0,'/home/user/lvllm/vllm-xiaotu-moe')
from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer
E,H,I,K,T=8,256,128,4,32
dev=torch.device("cuda:0"); torch.cuda.set_device(0)
os.environ.pop("XIAOTU_GPF_DBG",None)
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
n=A.numel(); nz=int((d>0).sum())
# bf16 在该量级下的 1 ulp
mag=A.float().abs().clamp(min=1e-6)
ulp=(2.0**(torch.floor(torch.log2(mag))-7))       # ⭐ bf16:1+8+7 ⇒ ulp = 2^(e-7) ✓
r=(d/ulp)
print(f"  元素 {n} · 不同 {nz} ({100*nz/n:.1f}%) · |y|均值 {A.float().abs().mean():.3e}")
print(f"  |Δ| max={d.max():.3e}  p99={d.flatten().kthvalue(max(1,int(0.99*n))).values:.3e}")
print(f"  ⭐ 以 bf16 ulp 计:中位={r.median():.2f} · p99={r.flatten().kthvalue(max(1,int(0.99*n))).values:.2f} · max={r.max():.2f}")
for th in (0.5,1,2,4,8):
    print(f"     ≤{th:>4} ulp 的元素占比 = {100*(r<=th).float().mean():6.2f}%")
# ⭐ 以【最大量级】的 bf16 ulp 为尺:不同求和顺序的差异本就相对于各项量级 ✓
mx = A.float().abs().max().item()
emax = int(torch.floor(torch.log2(torch.tensor(mx))).item())
ulp_max = 2.0**(emax-7)
print(f"  max|y| = {mx:.4e} ⇒ bf16 ulp(max) = {ulp_max:.4e}")
print(f"  ⭐ max|Δ| / ulp(max) = {d.max().item()/ulp_max:.2f} ulp")
print(f"  ⇒ {'✅ 差异 ≤ 2 ulp(max) ⇒ 【仅求和顺序造成的舍入】,与逐位相等等价' if d.max().item()/ulp_max<=2.0 else '⛔ 超过 2 ulp(max) ⇒ 仍需查'}")
