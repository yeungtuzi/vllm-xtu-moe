import os,sys,torch
sys.path.insert(0,'/home/user/lvllm/vllm-xiaotu-moe')
from vllm_xiaotu_moe import gpu_prefill as G
E,H,I,K,T=8,256,128,4,32
dev=torch.device("cuda:0"); torch.cuda.set_device(0)
os.environ["XIAOTU_GPF_DBG"]="1"; os.environ["XIAOTU_GPU_PREFILL_KERNEL"]="base"
g=torch.Generator().manual_seed(0)
w13=torch.randint(0,256,(E,2*I,H//2),generator=g,dtype=torch.uint8)
s13=torch.randint(122,132,(E,2*I,H//32),generator=g,dtype=torch.uint8)
w2 =torch.randint(0,256,(E,H,I//2),generator=g,dtype=torch.uint8)
s2 =torch.randint(122,132,(E,H,I//32),generator=g,dtype=torch.uint8)
x=(torch.randn(T,H,generator=g,dtype=torch.float32)*0.02).to(torch.bfloat16).to(dev)
ids=torch.randint(0,E,(T,K),generator=g,dtype=torch.int32).to(dev)
tw=torch.rand(T,K,generator=g); tw=(tw/tw.sum(-1,keepdim=True)).to(dev)
BK,BN=64,64; N=max(4096,BK)
def run(nm):
    G.dbg_buffer(dev,N).zero_()
    os.environ["XIAOTU_GPF_NMAJOR"]=nm
    y=G.gpu_moe_layer(x,ids,tw,*[t.to(dev) for t in (w13,s13,w2,s2)],H=H,I=I,K=K,device=dev)
    torch.cuda.synchronize()
    buf=G.dbg_buffer(dev,N)
    w=buf[:BK//2*BN].reshape(BK//2,BN).clone()
    sc=buf[BK//2*BN:(BK//2+BK//32)*BN].reshape(BK//32,BN).clone()
    return y, w, sc
y0,d0,s0=run("0"); y1,d1,s1=run("1")
print(f"  ⭐ 两臂 b_lo 整块相同 = {bool(torch.equal(d0,d1))}")
print(f"     arm0[0,:6] = {d0[0,:6].tolist()}")
print(f"     arm1[0,:6] = {d1[0,:6].tolist()}")
exp = w13[0,:BN,:BK//2].to(torch.int8).t().to(dev)   # [kp, n] = raw[0, n, kp]
print(f"  ⭐ arm0 == 期望(raw[0,n,kp])? {bool(torch.equal(d0,exp))}")
print(f"  ⭐ arm1 == 期望?            {bool(torch.equal(d1,exp))}")
print(f"  ⭐ 两臂 scale 块相同 = {bool(torch.equal(s0,s1))}")
print(f"     arm0 scale[0,:6] = {s0[0,:6].tolist()}")
print(f"     arm1 scale[0,:6] = {s1[0,:6].tolist()}")
exp_s = s13[0,:BN,:BK//32].to(torch.int8).t().to(dev)   # [kb, n] = raw_s[0, n, kb]
print(f"  ⭐ arm0 scale == 期望? {bool(torch.equal(s0,exp_s))}  arm1? {bool(torch.equal(s1,exp_s))}")
print(f"  输出逐位相同 = {bool(torch.equal(y0,y1))}")
i0=G._DBG_STATE.get("inter"); i1=G._DBG_STATE.get("inter")
# ⭐ 重跑两臂,分别取 inter
def run_i(nm):
    os.environ["XIAOTU_GPF_NMAJOR"]=nm
    y=G.gpu_moe_layer(x,ids,tw,*[t.to(dev) for t in (w13,s13,w2,s2)],H=H,I=I,K=K,device=dev)
    torch.cuda.synchronize()
    return y, G._DBG_STATE["inter"].clone(), G._DBG_STATE["out"].clone()
ya,ia,oa=run_i("0"); yb,ib,ob=run_i("1")
print(f"  ⭐⭐ inter(gate_up 输出)逐位相同 = {bool(torch.equal(ia,ib))}")
if not torch.equal(ia,ib):
    d=(ia.float()-ib.float()).abs()
    print(f"       inter |Δ| max={d.max():.3e} 不同元素 {int((d>0).sum())}/{d.numel()}")
print(f"  ⭐⭐ out 逐位相同 = {bool(torch.equal(oa,ob))}")
print(f"  ⇒ 结论:{"祸在【_gate_up_kernel】" if not torch.equal(ia,ib) else "gate_up 一致 ⇒ 祸在【_down_kernel】"}")
