"""Isolate xiaotu engine cpu_decode on GPU tensors."""
import numpy as np, torch, sys
sys.path.insert(0, "/home/user/lvllm/vllm-xiaotu-moe")
import xiaotu_moe
d = np.load('/home/user/lvllm/xiaotu-moe/scripts/real_layer1_model.npz')
w13,w2,g13,g2 = d['w13'],d['w2'],d['g13'],d['g2']
E,I,H = int(d['E']),int(d['I']),int(d['H']); K=6
def f2e(s): return np.clip(np.round(np.log2(np.maximum(s,1e-30))).astype(np.int32)+127,0,255).astype(np.uint8)
s13=f2e(g13); s2=f2e(g2)
m=xiaotu_moe.load(); cfg=m.MOEConfigV2()
cfg.num_processes=1;cfg.process_id=0;cfg.gpu_id=0;cfg.has_gate_proj=True
cfg.expert_num=E;cfg.top_k=K;cfg.hidden_size=H;cfg.intermediate_size=I
cfg.max_batch_size=256;cfg.max_num_seqs=256;cfg.stride=32;cfg.group_min_len=10
cfg.group_max_len=4096+128;cfg.groupN=1;cfg.groupK=32;cfg.activation_type=0
eng=m.MOE_MXFP4(cfg,w13,w2,s13,s2,0,0)
print("engine built; testing cpu_decode on GPU tensors", flush=True)
qlen=4
hidden = torch.randn(qlen,H,dtype=torch.bfloat16,device='cuda')
ids = torch.randint(0,E,(qlen,K),dtype=torch.int32,device='cuda')
wts = torch.rand(qlen,K,dtype=torch.float32,device='cuda')
out = torch.zeros(qlen,H,dtype=torch.float32,device='cuda')
s = torch.cuda.Stream()
print("stream:", s.cuda_stream, "hidden:", tuple(hidden.shape), hidden.dtype, flush=True)
eng.cpu_decode(s.cuda_stream, qlen, K, hidden.data_ptr(), ids.data_ptr(), wts.data_ptr(), out.data_ptr())
torch.cuda.synchronize()
print("cpu_decode OK; out nonzero:", int((out!=0).sum()), "/", out.numel(), flush=True)
