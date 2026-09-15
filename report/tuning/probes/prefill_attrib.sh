#!/usr/bin/env bash
# Prefill attribution for V4.1: fire one long, unique (cache-hostile) prompt at a
# running server and split the per-layer time into compute (CPU MoE) vs rest
# (GPU attention/indexer/hc + D2H/H2D + host-fn dispatch) using [cd-timing].
#
# WHY: the CPU microbench (bench_cpu_engine.py) says 40 layers of CPU MoE cost
# only ~13.6 s at B=4096/192 threads, while cold prefill was measured at ~60 s.
# So the CPU engine cannot be the whole story and we must measure the split
# INSIDE the service instead of extrapolating.
#
# The server must have been started with XIAOTU_CD_TIMING=1. qlen tags each
# line, so prefill (qlen = prompt length) and decode (qlen = 1) are separable.
#
# Usage: bash report/tuning/probes/prefill_attrib.sh <PORT> <TAG> [SYNTH] [MAXTOK]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
PORT="${1:?port}"
TAG="${2:?tag}"
SYNTH="${3:-4000}"
MAXTOK="${4:-4}"
LOG="$ROOT/report/tuning/logs/$TAG.log"
WAIT="${WAIT:-5400}"

echo "[attrib] waiting for readiness in $LOG (up to ${WAIT}s)"
DEADLINE=$(( SECONDS + WAIT ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  grep -q "Application startup complete" "$LOG" 2>/dev/null && break
  sleep 10
done
if ! grep -q "Application startup complete" "$LOG" 2>/dev/null; then
  echo "[attrib] NOT READY; tail:"; tail -25 "$LOG"; exit 1
fi
echo "[attrib] READY. sending synth=$SYNTH prompt (prefix cache is OFF, so this is a real prefill)"

MARK=$(wc -l < "$LOG")
"$PY" "$ROOT/report/tuning/probes/bench_v41.py" "$PORT" "prefill$SYNTH" \
  --synth "$SYNTH" --synth-seed 11 --maxtok "$MAXTOK" --repeat 1

echo
echo "================= [cd-timing] attribution (new log lines only) ================="
tail -n +"$((MARK + 1))" "$LOG" | grep -o '\[cd-timing\].*' | tail -120 > /tmp/attrib_lines.txt || true
wc -l < /tmp/attrib_lines.txt | xargs echo "cd-timing lines captured:"
# Split by qlen (prefill vs decode) and average period/compute/rest over layers.
awk '
  match($0, /layers=[0-9]+/){}
  {
    q=""; p=""; c=""; r="";
    if (match($0, /qlen=[0-9]+/))     q=substr($0,RSTART+5,RLENGTH-5);
    if (match($0, /period=[0-9.]+/))  p=substr($0,RSTART+7,RLENGTH-7);
    if (match($0, /compute=[0-9.]+/)) c=substr($0,RSTART+8,RLENGTH-8);
    if (match($0, /rest=[0-9.]+/))    r=substr($0,RSTART+5,RLENGTH-5);
    if (q=="") next;
    n[q]++; sp[q]+=p; sc[q]+=c; sr[q]+=r;
  }
  END{
    printf "%-8s %6s %12s %12s %12s\n","qlen","n","period_ms","compute_ms","rest_ms";
    for (k in n) printf "%-8s %6d %12.2f %12.2f %12.2f\n", k, n[k], sp[k]/n[k], sc[k]/n[k], sr[k]/n[k];
  }' /tmp/attrib_lines.txt | sort -k1,1
echo "================================================================================"
