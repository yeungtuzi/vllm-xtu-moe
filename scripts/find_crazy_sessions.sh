#!/usr/bin/env bash
# 按【标题里的"疯"字】查找用户标记的退化会话,并给出快速体检
# 用法: bash scripts/find_crazy_sessions.sh [关键字,默认 疯]
set -u
KW="${1:-疯}"
python3 - "$KW" <<'PY'
import json,os,glob,sys,zstandard as z,datetime,re
KW=sys.argv[1]
PC=os.path.expanduser("~/.dsh/storages/session_projcache/sessions")
SESS=os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--")
CORRUPT=re.compile(r"vxiaotuo|lvmlm|xiaomju|vluo|lvlm-|vlm-xiaotu|vllm-xiaom|vllm-xiaotuo")
def title_of(f):
    try:
        d=json.load(open(f,encoding="utf-8"))
        rows=(d.get("record") or {}).get("rows") or {}
        t=rows.get("title")
        if isinstance(t,dict): t=t.get("val")
        return t if isinstance(t,str) else None
    except Exception: return None
hits=[]
for f in glob.glob(PC+"/*.json"):
    t=title_of(f)
    if t and KW in t:
        sid=os.path.basename(f)[:-5]
        hits.append((sid,t))
if not hits:
    print("  没有找到标题含 %r 的会话"%KW); raise SystemExit
print("  ⭐ 标题含 %r 的会话:%d 个\n"%(KW,len(hits)))
print("  %-12s %-30s %-7s %-7s %-6s %-8s %s"%("最后修改","标题","调用","轮","损坏","首次损坏","会话 id"))
for sid,t in hits:
    f=os.path.join(SESS,sid,"session.v4.jsonl.zstd")
    if not os.path.exists(f):
        print("  %-12s %-30s (会话文件不存在)"%("—",t[:30])); continue
    mt=datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%m-%d %H:%M")
    with open(f,"rb") as fh: data=z.ZstdDecompressor().stream_reader(fh).read()
    recs=[json.loads(l) for l in data.decode("utf-8",errors="replace").split("\n") if l.strip()]
    calls=[r for r in recs if r.get("type")=="tool/call"]
    turns=sum(1 for r in recs if r.get("type")=="turn/start")
    bad=0; first=None
    for r in calls:
        dd=r.get("data") or {}
        try: a=json.loads(dd.get("arguments") or "{}")
        except Exception: a={}
        c=str(a.get("command",""))
        if "re.compile" in c or "xiaotuo|lvmlm" in c or "CORRUPT=" in c or "cat >" in c: continue
        if CORRUPT.search(c):
            bad+=1
            if first is None:
                tt=r.get("time"); first=datetime.datetime.fromtimestamp(tt/1000).strftime("%m-%d %H:%M") if tt else "?"
    print("  %-12s %-30s %-7d %-7d %-6d %-8s %s"%(mt,t[:30],len(calls),turns,bad,first or "—",sid[:24]))
PY
