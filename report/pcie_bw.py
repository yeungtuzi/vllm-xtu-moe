import torch, time
dev = torch.device("cuda:2")
torch.cuda.set_device(dev)
def bw(buf, n, iters=10, direction="h2d"):
    s = torch.cuda.Stream(dev)
    for _ in range(3):
        if direction == "h2d": buf.copy_(src, non_blocking=True)
        else: dst.copy_(buf, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        if direction == "h2d": buf.copy_(src, non_blocking=True)
        else: dst.copy_(buf, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return buf.numel() * buf.element_size() * iters / dt / 1e9

n = 256 * 1024 * 1024  # 256 MiB
src = torch.empty(n, dtype=torch.uint8, pin_memory=True)
dst = torch.empty(n, dtype=torch.uint8, pin_memory=True)
g   = torch.empty(n, dtype=torch.uint8, device=dev)
pg  = torch.empty(n, dtype=torch.uint8, pin_memory=True)
print(f"pinned H2D  {bw(g, n, direction='h2d'):8.2f} GB/s")
print(f"pinned D2H  {bw(g, n, direction='d2h'):8.2f} GB/s")
# pageable reference
src2 = torch.empty(n, dtype=torch.uint8)
g2 = torch.empty(n, dtype=torch.uint8, device=dev)
for _ in range(2): g2.copy_(src2); torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(5): g2.copy_(src2)
torch.cuda.synchronize(); dt=time.perf_counter()-t0
print(f"pageable H2D{n*5/dt/1e9:8.2f} GB/s")
# HBM bandwidth (device-to-device, large)
a = torch.empty(512*1024*1024, dtype=torch.uint8, device=dev)
b = torch.empty(512*1024*1024, dtype=torch.uint8, device=dev)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(10): b.copy_(a)
torch.cuda.synchronize(); dt=time.perf_counter()-t0
print(f"HBM D2D copy {2*512*1024*1024*10/dt/1e9:8.2f} GB/s (r+w)")
