#!/usr/bin/env bash
# Re-A/B MADV_HUGEPAGE on the NUMA shards, under conditions that actually
# reproduce the service.
#
# The flag is OFF by default. The comment at moe_v2.hpp:1300-1307 gives two
# reasons, and BOTH have since moved:
#   1. the A/B was run at B=6 (decode) on a warm single engine -- we now know the
#      service state is COLD (ROUNDROBIN) and costs 2.29x (NOTES §428), so a warm
#      B=6 A/B cannot see the TLB effect;
#   2. the memory blowup it caused ("3.2GB/layer -> 15-19GB/layer, node7 ~5GB")
#      came from the OLD sparse-span layout that mmap'd the whole E*stride block
#      and wrote sparse spans. shard_region is now COMPACT ("maps EXACTLY what it
#      stores", moe_v2.hpp:1329-1332), so the waste is ~1 hugepage per expert part
#      instead of a whole 2MB page per sparse 512KB span.
#
# So: measure ms/layer AND peak RSS with the flag off vs on, in COLD mode, at both
# prefill and decode batch sizes. If cold prefill speeds up and RSS stays sane,
# this is a free win with no kernel change.
#
# Usage: bash report/tuning/probes/ab_shard_hugepage.sh [B_PREFILL] [NENGINES]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
BP="${1:-1400}"
NENG="${2:-8}"
export XIAOTU_LAYER1_NPZ="${XIAOTU_LAYER1_NPZ:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
OUT="$ROOT/report/tuning/logs/ab_hugepage_$(date +%H%M%S).txt"
echo "writing $OUT"

run() {  # $1=label $2=hugepage(0/1) $3=BS $4=roundrobin $5=nengines
  local hpflag=""
  [ "$2" = "1" ] && hpflag="XIAOTU_MOE_SHARD_HUGEPAGE=1"
  {
    echo "############ $1  hp=$2 BS=$3 RR=$4 NENG=$5  $(date -Is)"
    ( cd "$ROOT" && env $hpflag XIAOTU_MOE_THREADS="${T:-192}" \
        NENGINES="$5" ROUNDROBIN="$4" BS="$3" REP="${REP:-2}" LAYER="${LAYER:-3}" \
        /usr/bin/time -v "$PY" scripts/bench_cpu_engine.py ) 2>&1 \
      | grep -E "cpu-bench|^ *[0-9]+ |Maximum resident|Elapsed \(wall"
    echo "-- AnonHugePages now: $(grep AnonHugePages /proc/meminfo)"
  } | tee -a "$OUT"
}

# Cold prefill is the target; warm prefill and cold decode guard against regressions.
run "cold-prefill  off" 0 "$BP" 1 "$NENG"
run "cold-prefill  ON " 1 "$BP" 1 "$NENG"
run "warm-prefill  off" 0 "$BP" 0 "$NENG"
run "warm-prefill  ON " 1 "$BP" 0 "$NENG"
run "cold-decode   off" 0 "1" 1 "$NENG"
run "cold-decode   ON " 1 "1" 1 "$NENG"
echo "### done $(date -Is)" | tee -a "$OUT"
