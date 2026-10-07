#!/usr/bin/env bash
# ⭐ 元话语率检查器(A53):只看会话存储,不碰任何服务 ✓
# 用法: bash scripts/meta_discourse_rate.sh [session-id...]   (省略则扫最近 5 个)
set -u
R=/home/user/lvllm/vllm-xiaotu-moe
python3 - "$@" <<'PY'
import json,os,sys,zstandard as z,re,glob,datetime
base=os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--")
META=re.compile(r"\b(I (?:now )?(?:have|understand|fully understand|see)|"
                r"I have (?:a |the )?(?:full|complete|very complete) picture|"
                r"Now I (?:have|understand|need|know)|I have enough|I now have)\b", re.I)
def turn_texts(sid):
    f=os.path.join(base,sid,"session.v4.jsonl.zstd")
    if not os.path.exists(f): return None
    with open(f,"rb") as fh: d=z.ZstdDecompressor().stream_reader(fh).read()
    out=[]
    for l in d.decode("utf-8",errors="replace").split("\n"):
        if not l.strip(): continue
        r=json.loads(l)
        if r.get("type")=="assistant/message":
            m=(r.get("data") or {}).get("message") or {}
            t="".join((p.get("text","") or "")+" " for p in (m.get("content") or []) if isinstance(p,dict))
            out.append(t)
    return out
args=[a for a in sys.argv[1:] if a.startswith("session-")]
if not args:
    rows=[]
    for d in glob.glob(base+"/*"):
        f=os.path.join(d,"session.v4.jsonl.zstd")
        if os.path.exists(f): rows.append((os.path.getmtime(f),os.path.basename(d)))
    rows.sort(reverse=True); args=[s for _,s in rows[:5]]
print("  %-44s %-6s %-11s %-11s %s"%("会话","轮","前1/3","后1/3","判断"))
for sid in args:
    ts=turn_texts(sid)
    if not ts or len(ts)<6: print("  %-44s (轮太少或不存在)"%sid[:44]); continue
    v=[len(META.findall(t)) for t in ts]; n=len(v); q=max(1,n//3)
    a=sum(v[:q])/q; c=sum(v[2*q:])/max(1,len(v[2*q:]))
    if c>1.0 and c>a*1.3: verdict="🔴 元话语率上升且>1.0 ⇒ 建议告警(写 handoff)"
    elif c>a*1.3:        verdict="🟡 上升中(未过 1.0)"
    else:                verdict="✓ 平稳"
    print("  %-44s %-6d %-11.2f %-11.2f %s"%(sid[:44],n,a,c,verdict))
PY
