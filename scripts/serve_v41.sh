#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 3xA100-40GB (SM80) — bring-up / smoke runner.
#
# V4.1 is a 475 GiB model on 120 GB of VRAM, so nothing fits without offload.
# The split vLLM already supports (verified 2026-09-14, NOTES §369):
#   * routed experts (269 GiB)  -> host, computed by the xiaotu CPU engine
#                                  (VLLM_EXPERTS_LOAD_DEVICE=cpu + this plugin)
#   * Engram tables (188.8 GiB) -> pinned host memory (engram_config.cpu_offload,
#                                  on by default)
#   * everything else (~23 GiB)  -> GPU
# The one thing that does NOT work out of the box on A100 is attention: V4.1
# ships only FlashMLA (SM90+) and FlashInfer (SM100/SM120) paths. This script is
# the end-to-end acceptance test for the SM80 fallback.
#
# Usage:
#   bash scripts/serve_v41.sh                 # dummy weights (fast bring-up)
#   LOAD=auto bash scripts/serve_v41.sh       # real weights (~long load)
#
# Env: TAG PORT GPUS TP MAXLEN LOAD GPU_UTIL EXTRA_ENV
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
TAG="${TAG:-v41}"
PORT="${PORT:-8077}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
MAXLEN="${MAXLEN:-2048}"
LOAD="${LOAD:-dummy}"          # dummy = no disk read, exercises the kernels
GPU_UTIL="${GPU_UTIL:-0.85}"
EXTRA_ENV="${EXTRA_ENV:-}"

OUTDIR="$ROOT/report/tuning/logs"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"

if [ ! -f "$CKPT/config.json" ]; then
  echo "[v41] checkpoint not found: $CKPT" >&2; exit 2
fi

{
  echo "tag=$TAG port=$PORT gpus=$GPUS tp=$TP maxlen=$MAXLEN load=$LOAD util=$GPU_UTIL"
  echo "ckpt=$CKPT"; date -Is
} > "$OUTDIR/$TAG.env"

echo "[v41] starting (load=$LOAD maxlen=$MAXLEN tp=$TP gpus=$GPUS) log=$LOG"

# Optional memory sampler. V4.1 peaks near ~1.4 TB of host memory on a box whose
# 8 NUMA nodes are 193 GB each, so the failure mode is *one node* exhausting
# (kernel: "constraint=CONSTRAINT_MEMORY_POLICY, nodemask=N"), not total RAM.
# MEMTRACE=1 records per-node free memory plus the fattest process's
# Rss/Shmem/Pss so the next run says exactly which component grows where.
if [ "${MEMTRACE:-0}" = "1" ]; then
  MEMLOG="$OUTDIR/$TAG.mem"
  (
    while true; do
      echo "### $(date -Is)" >> "$MEMLOG"
      numactl --hardware 2>/dev/null | grep -E "^node [0-9]+ free" >> "$MEMLOG"
      fat=$(ps -eo pid,rss --sort=-rss --no-headers 2>/dev/null | head -1 | awk '{print $1}')
      if [ -n "$fat" ] && [ -r "/proc/$fat/smaps_rollup" ]; then
        echo "-- fattest pid=$fat" >> "$MEMLOG"
        grep -E "^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Dirty|Anonymous):" \
          "/proc/$fat/smaps_rollup" >> "$MEMLOG" 2>/dev/null
      fi
      sleep "${MEMTRACE_INTERVAL:-60}"
    done
  ) &
  echo $! > "$OUTDIR/$TAG.mem.pid"
  echo "[v41] memtrace -> $MEMLOG"
fi

cd /tmp
export CUDA_VISIBLE_DEVICES="$GPUS"
nohup env \
  HF_HUB_OFFLINE=1 \
  VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  VLLM_HANDSHAKE_TIMEOUT_MINS=120 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_EXPERTS_LOAD_DEVICE=cpu \
  XIAOTU_MOE_THREADS="${XIAOTU_MOE_THREADS:-60}" \
  XIAOTU_MOE_NSLICE_SMALL=0 \
  XIAOTU_MOE_ASYNC=0 \
  XIAOTU_MOE_SPIN_IDLE_US=0 \
  OMP_NUM_THREADS=1 \
  $EXTRA_ENV \
  numactl --interleave=all "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name dsv41 \
    --load-format "$LOAD" \
    --max-model-len "$MAXLEN" --tensor-parallel-size "$TP" --max-num-seqs 1 \
    --gpu-memory-utilization "$GPU_UTIL" --enforce-eager --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --kernel-config '{"enable_jit_warmup": false}' \
    --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[v41] pid=$(cat "$OUTDIR/$TAG.pid")"

# Wait for readiness. Model construction alone took ~23 min with dummy weights.
DEADLINE=$(( SECONDS + ${READY_TIMEOUT:-3600} ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "[v41] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[v41] server exited early; tail:"; tail -25 "$LOG"; exit 1
  fi
  sleep 10
done
echo "[v41] TIMEOUT waiting for readiness; tail:"; tail -25 "$LOG"; exit 1
