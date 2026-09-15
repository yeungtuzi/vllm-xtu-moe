#!/usr/bin/env bash
# Sweep XIAOTU_MOE_THREADS on the pure-CPU engine microbench (real weights).
#
# WHY: the engine caps threads at 120 (numa_pool.hpp:826) on the strength of
# *decode* bandwidth experiments ("每 CCD 4-5 核"), and serve_v41.sh pins 60.
# Prefill is COMPUTE-bound (NASS >> 1), so the decode-tuned thread count is a
# prime suspect for the ~59 tok/s cold-prefill. This measures TFLOP/s directly,
# with no GPU copies, so it isolates the CPU MoE kernel.
#
# Usage: bash report/tuning/probes/sweep_cpu_threads.sh [THREADS...]
#   env: BS (default 1,64,512,2048,8192) REP (default 3) LAYER (default 3)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
export XIAOTU_LAYER1_NPZ="${XIAOTU_LAYER1_NPZ:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
export BS="${BS:-1,64,512,2048,8192}"
export REP="${REP:-3}"
export LAYER="${LAYER:-3}"
OUT="${OUT:-$ROOT/report/tuning/logs/sweep_threads_$(date +%H%M%S).txt}"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

for t in "${@:-60 120 192 384}"; do
  echo "############ THREADS=$t  $(date -Is)" | tee -a "$OUT"
  ( cd "$ROOT" && XIAOTU_MOE_THREADS="$t" "$PY" scripts/bench_cpu_engine.py ) 2>&1 | tee -a "$OUT"
done
echo "### done $(date -Is)" | tee -a "$OUT"
