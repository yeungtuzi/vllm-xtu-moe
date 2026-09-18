#!/usr/bin/env bash
# 端到端 A-B-A:旧内核 vs 新内核(block_23 + 单行 4 路 ILP),同窗口交替。
#
# 本机负载在 30~130 之间漂移,先后测的端到端数字不可比 ⇒ 必须换二进制重启、
# 在同一时间窗口内交替测,并在每段记录 loadavg。
#
# 用法:scripts/ab_serve_kernel.sh            # 需要 /tmp/{old,new}_vnni.so
# 产出:dev-docs/report/tuning/ab_serve_kernel.txt + raw/*_aba_*.json
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/dev-docs/report/tuning/logs"
SO="xiaotu_moe/build/_xiaotu_moe_C_avx512_vnni.cpython-312-x86_64-linux-gnu.so"
OUT="$ROOT/dev-docs/report/tuning/ab_serve_kernel.txt"
SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","model":"/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master"}'
PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python

stop_server() {
  local pid
  pid="$(cat "$LOGS/$1.pid" 2>/dev/null || true)"
  [ -n "${pid:-}" ] || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -KILL "$pid" 2>/dev/null || true
  pkill -9 -P "$pid" 2>/dev/null || true
}

run_leg() {
  local which="$1" tag="aba_$1"
  cp "/tmp/${which}_vnni.so" "$SO"
  stop_server "$tag"; stop_server "${2:-none}"; sleep 8
  MODE=dsv4 TAG="$tag" PORT=8070 TP=1 MAXLEN=262144 SEQS=128 MAX_NBT=8192 \
    GPU_UTIL=0.85 KV_DTYPE=fp8_ds_mla KV_MEM_BYTES=8589934592 \
    THREADS=192 OMP=96 EAGER=0 EP=0 GPUS=2 ENV_EXTRA="XIAOTU_CD_TIMING=1" \
    SPEC="$SPEC" scripts/tune_serve.sh >/dev/null 2>&1 || { echo "serve failed $which"; return 1; }
  L=128 C=1 N=4 OUT=128 TAG="aba_${which}_c1" SERVER_TAG="$tag" "$PY" scripts/bench_nat_client.py \
    | sed "s/^\[nat-client\] /leg=$which load=$(awk '{print $1}' /proc/loadavg) /;s/ per-req.*//" | tee -a "$OUT"
  grep -h "cd-timing" "$LOGS/$tag.log" | tail -3 | sed "s/^/leg=$which /" | tee -a "$OUT"
}

: > "$OUT"
run_leg old ""
run_leg new "aba_old"
echo "[aba] done (服务当前运行的是 new 内核)" | tee -a "$OUT"
