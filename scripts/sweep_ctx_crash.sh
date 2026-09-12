#!/usr/bin/env bash
# 长上下文 decode 崩溃定性:按长度阶梯发 max_tokens=1 请求,每档后用一次短请求探活。
# 用法: scripts/sweep_ctx_crash.sh [PORT] [长度列表,空格分隔]
set -uo pipefail
PORT="${1:-8070}"; shift || true
LENS=("$@"); [ ${#LENS[@]} -eq 0 ] && LENS=(8192 16384 24576 32768)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for N in "${LENS[@]}"; do
  F="$ROOT/report/tuning/datasets/nat$N.jsonl"
  [ -f "$F" ] || { echo "len=$N SKIP(无数据集)"; continue; }
  python3 - "$PORT" "$F" "$N" <<'PY'
import json,sys,time,urllib.request,urllib.error
port,path,n=sys.argv[1],sys.argv[2],sys.argv[3]
p=json.loads(open(path).read().strip().split("\n")[0])["prompt"]
def call(prompt,mt):
    b=json.dumps({"model":"DeepSeek-V4-Flash-xiaotu","prompt":prompt,"max_tokens":mt,
                  "temperature":0.0,"stream":True,"ignore_eos":True}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",data=b,
                             headers={"Content-Type":"application/json"})
    t0=time.time(); ttft=None
    with urllib.request.urlopen(r,timeout=1800) as resp:
        for line in resp:
            if line.startswith(b"data: ") and b"[DONE]" not in line and ttft is None:
                ttft=time.time()-t0
    return ttft,time.time()-t0
try:
    ttft,tot=call(p,1)
    print(f"len={n} OK ttft={ttft*1000:.0f}ms wall={tot*1000:.0f}ms rate={int(n)/ttft:.0f}t/s",flush=True)
    # 探活:紧跟一次短 decode,验证引擎仍存活(崩溃发生在 decode 的 indexer)
    t2,_=call("hello world",4)
    print(f"len={n} probe-after OK ttft={t2*1000:.0f}ms",flush=True)
except urllib.error.HTTPError as e:
    print(f"len={n} **CRASH** HTTP {e.code}",flush=True)
except Exception as e:
    print(f"len={n} **CRASH** {type(e).__name__}: {e}",flush=True)
PY
  # 引擎若已死,后面的档位没有意义
  curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { echo "[sweep] 引擎已死,停止于 len=$N"; exit 2; }
done
echo "[sweep] 全部完成:引擎存活"
