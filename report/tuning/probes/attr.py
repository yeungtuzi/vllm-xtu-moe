import re, sys, collections
pid = sys.argv[1]
st = {}
for ln in open(f'/proc/{pid}/status'):
    k = ln.split(':',1)[0]
    if k in ('RssAnon','RssFile','RssShmem','VmRSS','VmSize'):
        st[k] = int(ln.split(':',1)[1].split()[0])
print(f"pid {pid} status(MiB): " + " ".join(f"{k}={v>>10}" for k,v in st.items()))
agg = collections.Counter(); anon = collections.Counter(); cnt = collections.Counter()
cur=None
for ln in open(f'/proc/{pid}/smaps'):
    m = re.match(r'^([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s*(.*)$', ln)
    if m:
        cur = m.group(4).strip() or f"[{m.group(3)} anon]"
    elif cur is not None and ln.startswith('Rss:'):
        r = int(ln.split()[1]); agg[cur]+=r; cnt[cur]+=1
    elif cur is not None and ln.startswith('Anonymous:'):
        anon[cur]+=int(ln.split()[1])
print("\ntop VMAs by aggregated Rss (MiB):")
for name, r in agg.most_common(12):
    print(f"  {r>>10:8d} MiB  anon={anon[name]>>10:8d} MiB  n={cnt[name]:4d}  {name[:80]}")
