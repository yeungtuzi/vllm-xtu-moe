import os, torch
def roll(tag):
    d={}
    for ln in open('/proc/self/smaps_rollup'):
        if ':' in ln:
            k,v=ln.split(':',1); d[k.strip()]=v.strip()
    keys=['Rss','Pss','Shared_Clean','Shared_Dirty','Private_Clean','Private_Dirty',
          'Anonymous','Locked','RssAnon','RssFile','RssShmem']
    st={}
    for ln in open('/proc/self/status'):
        if ln.startswith(('RssAnon','RssFile','RssShmem','VmRSS','VmSwap')):
            k,v=ln.split(':',1); st[k.strip()]=v.strip().split()[0]
    print(f"--- {tag}")
    print("   smaps_rollup:", {k:d.get(k) for k in keys if k in d})
    print("   status      :", st)
roll("baseline")
n = 512*1024*1024  # 2 GiB of float32
a = torch.empty(n, dtype=torch.float32); a.fill_(1.0)
roll("anon 2GiB filled")
p = torch.empty(n, dtype=torch.float32, pin_memory=True); p.fill_(1.0)
roll("PINNED 2GiB filled")
s = torch.empty(n, dtype=torch.float32).share_memory_(); s.fill_(1.0)
roll("shm 2GiB filled")
