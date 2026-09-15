import torch, re
n = 512*1024*1024
p = torch.empty(n, dtype=torch.float32, pin_memory=True); p.fill_(1.0)
best=[]
cur=None
for ln in open('/proc/self/smaps'):
    m=re.match(r'^([0-9a-f]+)-([0-9a-f]+) (\S+) (\S+) (\S+) (\S+)\s*(.*)$', ln)
    if m:
        if cur: best.append(cur)
        cur={'range':m.group(1)+'-'+m.group(2),'perm':m.group(3),
             'size':(int(m.group(2),16)-int(m.group(1),16))//1024,'name':m.group(7).strip(),'Rss':0,'Anon':0,'PD':0}
    elif cur:
        if ln.startswith('Rss:'): cur['Rss']=int(ln.split()[1])
        elif ln.startswith('Anonymous:'): cur['Anon']=int(ln.split()[1])
        elif ln.startswith('Private_Dirty:'): cur['PD']=int(ln.split()[1])
if cur: best.append(cur)
print("VMAs with Rss > 200 MB:")
for v in sorted(best,key=lambda x:-x['Rss'])[:8]:
    if v['Rss']>200*1024:
        print(f"  Rss={v['Rss']/1048576:6.2f}G Anon={v['Anon']/1048576:6.2f}G PD={v['PD']/1048576:6.2f}G size={v['size']/1048576:6.2f}G {v['perm']} name={v['name']!r}")
