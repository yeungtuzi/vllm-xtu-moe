#!/usr/bin/env bash
# 退化取证器:盯 DSH 会话存储 + 引擎日志,一旦发现【路径损坏】立刻抓现场
# 用法:bash scripts/watch_agent_degeneration.sh          (建议用 proc.sh 起)
# ⚠️ 只读会话存储与生产日志;只会发出【一次】极短的对照请求(用于区分"会话态/进程态")
set -u
R=/home/user/lvllm/vllm-xiaotu-moe
LOGDIR="$R/dev-docs/report/tuning/logs"
INTERVAL="${INTERVAL:-30}"
OUT="$LOGDIR/deg_captures"
STATE="$OUT/.captured_sessions"      # ⭐ 已抓取过的会话,避免重复触发
mkdir -p "$OUT"
mkdir -p "$OUT"
echo "[$(date '+%F %T')] 取证器启动:每 ${INTERVAL}s 检查一次,输出到 $OUT"
echo "  触发条件:① 会话标题含【疯】(用户标记)② 工具参数出现路径损坏"

# ⭐ 标题标记检测:用户把标题改成含"疯"即视为标记
TITLE_PY='
import json,os,glob,sys
KW=sys.argv[1] if len(sys.argv)>1 else "疯"
PC=os.path.expanduser("~/.dsh/storages/session_projcache/sessions")
out=[]
for f in glob.glob(PC+"/*.json"):
    try:
        d=json.load(open(f,encoding="utf-8"))
        rows=(d.get("record") or {}).get("rows") or {}
        t=rows.get("title")
        if isinstance(t,dict): t=t.get("val")
        if isinstance(t,str) and KW in t:
            out.append((os.path.basename(f)[:-5], t))
    except Exception: pass
for sid,t in out: print("MARK\t%s\t%s"%(sid,t))
'


PY='
import json,os,glob,zstandard as z,datetime,re,sys
base=os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--")
CORRUPT=re.compile(r"vxiaotuo|lvmlm|xiaomju|vluo|lvlm-|vlm-xiaotu|vllm-xiaom|vllm-xiaotuo")
def scan():
    hits=[]
    for d in sorted(glob.glob(base+"/*"), key=os.path.getmtime, reverse=True)[:8]:
        f=os.path.join(d,"session.v4.jsonl.zstd")
        if not os.path.exists(f): continue
        sid=os.path.basename(d)
        if sid.startswith("session-ecd2c6e6"): continue          # 我自己的会话:排除
        try:
            with open(f,"rb") as fh: data=z.ZstdDecompressor().stream_reader(fh).read()
            recs=[json.loads(l) for l in data.decode("utf-8",errors="replace").split("\n") if l.strip()]
        except Exception: continue
        calls=[r for r in recs if r.get("type")=="tool/call"]
        bad=0; first=None
        for r in calls:
            dd=r.get("data") or {}
            try: a=json.loads(dd.get("arguments") or "{}")
            except Exception: a={}
            cmd=str(a.get("command",""))
            # ⭐ 修掉"自匹配":跳过本身含有检测模式的命令(我自己的脚本)
            if "re.compile" in cmd or "xiaotuo|lvmlm" in cmd or "CORRUPT=" in cmd: continue
            if CORRUPT.search(cmd):
                bad+=1
                if first is None:
                    t=r.get("time"); first=datetime.datetime.fromtimestamp(t/1000).strftime("%H:%M:%S") if t else "?"
        if bad: hits.append((sid,bad,first,len(calls)))
    return hits
h=scan()
for sid,bad,first,n in h:
    print("HIT\t%s\t%d\t%s\t%d"%(sid,bad,first or "?",n))
'
while true; do
  # ⭐ A28 后新增:崩溃关键词告警(aten::new_empty / EngineDeadError ⇒ 说明 A28 修法未生效,需回滚)
  CL=$(ls -t "$LOGDIR"/v41_8070.[0-9]*.log 2>/dev/null | head -1)
  # ⭐ 只看【新增行】:记录已扫过的 offset ⇒ 历史命中不会每轮重报 ✓
  OFFS="$OUT/.crash_scan_offset"
  CUR=0; [ -n "$CL" ] && CUR=$(stat -c %s "$CL" 2>/dev/null || echo 0)
  PREV=$(cat "$OFFS" 2>/dev/null || echo 0)
  if [ "$CUR" -lt "$PREV" ]; then PREV=0; fi     # 日志轮转 ⇒ 从头扫
  if [ -n "$CL" ] && [ "$CUR" -gt "$PREV" ] && tail -c +$((PREV+1)) "$CL" 2>/dev/null | grep -qaE "aten::new_empty|EngineDeadError"; then
    TS=$(date '+%Y%m%d-%H%M%S'); D="$OUT/CRASH-$TS"; mkdir -p "$D"
    tail -c +$((PREV+1)) "$CL" 2>/dev/null | grep -aE "aten::new_empty|EngineDeadError" | tail -20 > "$D/crash_lines.txt"
    echo "崩溃关键词命中(新增部分)" > "$D/README.txt"
    cp -f "$CL" "$D/engine.log" 2>/dev/null
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > "$D/gpu.txt" 2>/dev/null
    echo "⚠️ A28 修法后仍出现崩溃关键词 ⇒ 需回滚 ACT_RESERVE 或继续排查" > "$D/README.txt"
    echo "[$(date '+%F %T')] 🚨 崩溃关键词命中 ⇒ 现场 $D"
  fi
  [ -n "$CL" ] && echo "$CUR" > "$OFFS"     # ⭐ 记下已扫位置 ✓
  HITS=$(python3 -c "$PY" 2>/dev/null)
  MARKS=$(python3 -c "$TITLE_PY" 2>/dev/null)
  # 过滤掉已经抓过的会话(避免对老标记反复触发)
  if [ -n "$HITS" ] || [ -n "$MARKS" ]; then
    HITS=$(printf '%s\n%s\n' "$HITS" "$MARKS" | python3 -c "
import sys,os
state=set()
if os.path.exists('$STATE'):
    state={l.strip() for l in open('$STATE') if l.strip()}
for l in sys.stdin:
    l=l.rstrip('\n')
    if not l.strip(): continue
    parts=l.split('\t')
    if len(parts)>=2 and parts[1] not in state: print(l)
" 2>/dev/null)
  fi
  if [ -n "$HITS" ]; then
    TS=$(date '+%Y%m%d-%H%M%S')
    D="$OUT/$TS"; mkdir -p "$D"
    echo "$HITS" > "$D/hits.tsv"
    echo "[$(date '+%F %T')] ⚠️ 检测到退化,抓取现场 → $D"
    : > "$D/summary.txt"
    # 统一两种格式:HIT<TAB>sid<TAB>bad<TAB>first<TAB>n   /   MARK<TAB>sid<TAB>title
    while IFS=$'\t' read -r tag a b c d; do
      [ -n "${tag:-}" ] || continue
      sid="$a"; base=$(basename "$sid")
      if [ "$tag" = "MARK" ]; then
        echo "【用户标记】标题: $b" >> "$D/summary.txt"
      else
        echo "【路径损坏】损坏 $b 次,首次 $c,共 $d 次调用" >> "$D/summary.txt"
      fi
      cp -f "$HOME/.dsh/sessions/--home-user-lvllm--/$sid/session.v4.jsonl.zstd" "$D/${sid}.jsonl.zstd" 2>/dev/null
      # ⭐ 记住已抓取,避免对同一个老标记反复触发
      echo "$sid" >> "$STATE"
    done <<< "$HITS"
    # 2) 引擎日志窗口(前后 10 分钟)
    L=$(ls -t "$R"/dev-docs/report/tuning/logs/v41_8070.*.log 2>/dev/null | head -1)
    [ -n "$L" ] && tail -4000 "$L" > "$D/engine_tail.log"
    # 3) 引擎状态快照
    python3 - <<'MPY' > "$D/metrics.txt" 2>/dev/null
import urllib.request
try:
    print(urllib.request.urlopen("http://127.0.0.1:8070/metrics", timeout=8).read().decode())
except Exception as e:
    print("metrics 读取失败:", e)
MPY
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv > "$D/gpu.txt" 2>/dev/null
    # 4) ⭐ 进程态对照:一个【极短】的新请求(若它也异常 ⇒ 进程被污染 ⇒ 需重启)
    /home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python - <<PY >> "$D/control_request.txt" 2>&1
import requests
r=requests.post("http://127.0.0.1:8070/v1/chat/completions",
  json={"model":"DeepSeek-V4.1-Flash","max_tokens":16,"temperature":0.0,
        "messages":[{"role":"user","content":"只回答:ok"}]},timeout=300)
print("HTTP",r.status_code)
print(str(r.text)[:400])
PY
    echo "[$(date '+%F %T')] ✅ 现场已保存:$D"
    break     # 抓一次即退出(避免重复)
  fi
  sleep "$INTERVAL"
done
