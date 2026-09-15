#!/usr/bin/env bash
# One-shot V4.1 performance instrument: decode, cold prefill, and the
# compute/rest attribution between them. Waits for readiness, then measures.
#
# IMPORTANT (methodology, learned the hard way -- NOTES §420):
#   * the server MUST run with PREFIX_CACHE=0 for prefill numbers, otherwise a
#     long prompt can be served almost entirely from the prefix cache and
#     reports absurd values (2-9 k tok/s);
#   * the server MUST run with XIAOTU_CD_TIMING=1 for the attribution;
#   * prefill uses a unique-per-position prompt (--synth) for the same reason;
#   * decode uses REPEAT>1 and we read the min/max spread, not just the mean.
#
# Usage: bash report/tuning/probes/measure_v41.sh <PORT> <TAG> [MAXSEQS]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
PORT="${1:?port}"
TAG="${2:?tag}"
MAXSEQS="${3:-1}"
LOG="$ROOT/report/tuning/logs/$TAG.log"

bash "$ROOT/report/tuning/probes/prefill_attrib.sh" "$PORT" "$TAG" "${SYNTH:-4000}" "${MAXTOK:-4}" || exit 1

echo
echo "================= serial decode (short prompt, 64 tok) ================="
"$PY" "$ROOT/report/tuning/probes/bench_v41.py" "$PORT" "decode$TAG" \
  --maxtok 64 --repeat "${REP:-3}"

if [ "$MAXSEQS" -ge 8 ] 2>/dev/null; then
  echo
  echo "================= concurrency C=8 ================="
  "$PY" "$ROOT/report/tuning/probes/bench_v41.py" "$PORT" "conc8$TAG" \
    --maxtok 64 --conc 8 --reqs 8
else
  echo
  echo "[measure] MAXSEQS=$MAXSEQS < 8 -> skipping the concurrency test"
  echo "[measure] (relaunch with MAXSEQS=8 to measure aggregate throughput)"
fi
