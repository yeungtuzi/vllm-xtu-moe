#!/usr/bin/env bash
# 线程池丢票竞态(§501/R113)**批量复现**:并行跑 N 个进程,每个跑固定时长,数 aborts。
#
# 判定:进程非 0 退出(或日志里出现 `[pool] WATCHDOG fired`)= 触发了丢票。
# 为什么能快速判定:配合 `XIAOTU_MOE_POOL_DEADLINE_MS` 把"卡满 300s 才 abort"降到毫秒级。
#
# 用法:
#   bash scripts/stress_pool.sh [进程数=8] [每进程秒数=60] [每次调用的 BS=1] [topk=2]
# 产出:/tmp/stress_pool/<pid>.log,末尾打印 aborts/N。
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
N="${1:-8}"
SECS="${2:-60}"
BS="${3:-1}"
K="${4:-2}"
THREADS="${THREADS:-120}"
DEADLINE_MS="${DEADLINE_MS:-1000}"
OUT=/tmp/stress_pool
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[stress_pool] N=$N secs=$SECS BS=$BS K=$K threads=$THREADS deadline_ms=$DEADLINE_MS out=$OUT"

pids=()
for i in $(seq 1 "$N"); do
  ( XIAOTU_MOE_POOL_DEADLINE_MS="$DEADLINE_MS" ENG=xiaotu XIAOTU_MOE_THREADS="$THREADS" \
      timeout "$SECS" "$PY" "$ROOT/scripts/stress_pool.py" \
        --iters 100000000 --bs "$BS" --k "$K" --report 20000 --seed "$i" \
        > "$OUT/$i.log" 2>&1
    echo "$?" > "$OUT/$i.rc" ) &
  pids+=($!)
done
wait "${pids[@]}" 2>/dev/null

aborts=0; timeouts=0; ok=0
for i in $(seq 1 "$N"); do
  rc="$(cat "$OUT/$i.rc" 2>/dev/null || echo 99)"
  if grep -q "WATCHDOG" "$OUT/$i.log" 2>/dev/null; then
    aborts=$((aborts+1)); echo "  [$i] **丢票**(WATCHDOG) rc=$rc"
    grep -a -A2 "WATCHDOG fired" "$OUT/$i.log" | head -4 | sed 's/^/       /'
  elif [ "$rc" = "124" ]; then timeouts=$((timeouts+1)); echo "  [$i] 正常(跑满超时被 kill)"
  else echo "  [$i] rc=$rc $(tail -1 "$OUT/$i.log" 2>/dev/null | cut -c1-90)"; ok=$((ok+1)); fi
done
echo "[stress_pool] aborts=$aborts  ran-full=$timeouts  other=$ok  (共 $N)"
[ "$aborts" = "0" ]
