#!/usr/bin/env bash
# Test the "amplification" hypothesis for the service-vs-isolated 3.3x gap (§433).
#
# The engine runs one worker per PHYSICAL core (192). A parallel region ends at a
# barrier that waits for the SLOWEST worker, so if even a few other runnable
# threads exist, a few workers get preempted and every barrier pays a full
# scheduling-latency penalty -- while the median worker is fine. A tiny
# oversubscription can therefore cost far more than its share of CPU. That is
# exactly the shape of the observed gap: no effect at B=1 (short regions), 3.3x at
# B=1400 (many long regions).
#
# This simulates vLLM's background threads (EngineCore, CUDA/driver, GPU worker,
# sampler, ...) with N busy processes and re-measures the SAME microbench.
#
# Usage: bash report/tuning/probes/ab_cpu_contention.sh [B] [SPINNERS...]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
B="${1:-1400}"
shift || true
SPIN="${*:-0 4 12 24}"
export XIAOTU_LAYER1_NPZ="${XIAOTU_LAYER1_NPZ:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
OUT="$ROOT/report/tuning/logs/ab_contention_$(date +%H%M%S).txt"
echo "writing $OUT"

for n in $SPIN; do
  pids=()
  for _ in $(seq 1 "$n"); do
    "$PY" -c "while True: pass" & pids+=($!)
  done
  sleep 2
  {
    echo "############ SPINNERS=$n (of 192 physical cores) B=$B THREADS=${T:-192}  $(date -Is)"
    ( cd "$ROOT" && XIAOTU_MOE_THREADS="${T:-192}" BS="$B" REP="${REP:-3}" LAYER="${LAYER:-3}" \
        "$PY" scripts/bench_cpu_engine.py ) 2>&1 | grep -E "^ *[0-9]+ "
  } | tee -a "$OUT"
  for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
  wait 2>/dev/null
  sleep 2
done
echo "### done $(date -Is)" | tee -a "$OUT"
