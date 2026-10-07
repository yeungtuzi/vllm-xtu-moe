#!/usr/bin/env python3
"""同口径会话指标(只读 ✓)。用法:_sess_metrics.py <session-id> [label]"""
import json, os, sys, re, datetime
import zstandard as z
base = os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--")
sid = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else sid[:20]
f = os.path.join(base, sid, "session.v4.jsonl.zstd")
if not os.path.exists(f): print("%s\t不存在"%label); sys.exit(0)
META = re.compile(r"\b(I (?:now )?(?:have|understand|fully understand|see)|"
                  r"I have (?:a |the )?(?:full|complete|very complete) picture|"
                  r"Now I (?:have|understand|need|know)|I have enough|I now have)\b", re.I)
CORR = re.compile(r"vxiaotuo|lvmlm|xiaomju|lvlm-|vllm-xiaom|vllm-xiaotuo|lvlml|lvvm|xao-l-e")
with open(f, "rb") as fh: d = z.ZstdDecompressor().stream_reader(fh).read()
rr = [json.loads(l) for l in d.decode("utf-8", errors="replace").split("\n") if l.strip()]
A, calls = [], []
for r in rr:
    t = r.get("type")
    if t == "assistant/message":
        m = (r.get("data") or {}).get("message") or {}
        A.append("".join((p.get("text","") or "") + " " for p in (m.get("content") or []) if isinstance(p, dict)))
    elif t == "tool/call":
        try: a = json.loads((r.get("data") or {}).get("arguments") or "{}")
        except Exception: a = {}
        c = str(a.get("command",""))
        if "re.compile" in c or "CORRUPT=" in c: continue
        calls.append(c)
v = [len(META.findall(x)) for x in A]; n = len(v); q = max(1, n // 3)
bad = sum(1 for c in calls if CORR.search(c))
print("%s\t%d\t%d\t%.2f\t%.2f\t%.2f\t%d\t%.2f" % (
    label, n, len(calls), sum(v[:q])/q if v else 0, sum(v)/max(1,n),
    sum(v[2*q:])/max(1,len(v[2*q:])) if v else 0, bad, 100*bad/max(1,len(calls))))
