#!/usr/bin/env bash
# Why is the CPU MoE 3.3x slower inside the service than in isolation?
#
# Measured (NOTES §427): service `compute` at qlen=1400 = 427 ms/layer, while the
# single-layer microbench at the same dims/threads is ~128 ms/layer.
# bench_cpu_engine.py has two knobs built for this question:
#   NENGINES=N   build N engines (like the service's one-per-layer), which makes
#                the resident weight footprint ~N*6.8 GB instead of 6.8 GB.
#   ROUNDROBIN=1 call them in rotation, so each call's weights/scratch are COLD
#                (the service visits 40 different layers per forward, never
#                re-using a layer's warm state).
# ROUNDROBIN=0 + NENGINES=N tests footprint alone; ROUNDROBIN=1 adds cold cycling.
#
# Usage: bash report/tuning/probes/sweep_cpu_nengines.sh [B] [THREADS]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
B="${1:-1400}"
T="${2:-192}"
export XIAOTU_LAYER1_NPZ="${XIAOTU_LAYER1_NPZ:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
OUT="$ROOT/report/tuning/logs/sweep_nengines_$(date +%H%M%S).txt"
echo "writing $OUT"
for cfg in "1 0" "8 0" "8 1" "40 0" "40 1"; do
  set -- $cfg
  {
    echo "############ NENGINES=$1 ROUNDROBIN=$2 B=$B THREADS=$T  $(date -Is)"
    ( cd "$ROOT" && XIAOTU_MOE_THREADS="$T" NENGINES="$1" ROUNDROBIN="$2" \
        BS="$B" REP="${REP:-2}" LAYER="${LAYER:-3}" "$PY" scripts/bench_cpu_engine.py )
  } 2>&1 | tee -a "$OUT"
done
echo "### done $(date -Is)" | tee -a "$OUT"
