import json, glob, os, re
rows=[]
for f in sorted(glob.glob('/tmp/rnd2/*.json')):
    d=json.load(open(f))
    b=os.path.basename(f)[:-5]
    m=re.match(r'([a-z0-9]+)_L(\d+)_C(\d+)', b)
    if not m: continue
    tag,L,C=m.group(1),int(m.group(2)),int(m.group(3))
    comp=d.get('completed') or 0; n=d.get('num_prompts') or 0
    tot=d.get('total_input_tokens') or 0
    ttft=d.get('mean_ttft_ms') or 0; tpot=d.get('mean_tpot_ms') or 0
    if comp and ttft:
        pre=tot/(comp*ttft/1000.0)          # tok/s
        dec=1000.0/tpot if tpot else 0
        rows.append((tag,L,C,pre,dec,comp,n))
    else:
        rows.append((tag,L,C,None,None,comp,n))
name={'glm':'GLM-5.3-Flash','mimo':'MiMo-V2.5','v41':'DeepSeek-V4.1'}
print(f"{'model':<17}{'L':>6}{'C':>3}{'prefill':>10}{'decode':>9}   ok")
for tag,L,C,pre,dec,comp,n in sorted(rows):
    if pre is None: print(f"{name.get(tag,tag):<17}{L:>6}{C:>3}{'FAIL':>10}{'--':>9}   {comp}/{n}")
    else:           print(f"{name.get(tag,tag):<17}{L:>6}{C:>3}{pre:>10.1f}{dec:>9.1f}   {comp}/{n}")
print(f"\n共 {len(rows)}/12 格")
