#!/usr/bin/env bash
# 目标场景基准:单卡 256K + DSpark(k=5)+ CUDA graph,自然文本、精确上下文长度。
# 一条命令跑完 上下文 × 并发 的曲线,并抓取服务端逐位置接受率。
#
# 用法:scripts/run_nat_curve.sh          # 起服务 + 跑曲线 + 关服务
#      SKIP_SERVE=1 scripts/run_nat_curve.sh   # 复用已在跑的服务
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/dev-docs/report/tuning/logs"
TAG="${TAG:-nat_k5}"
PORT="${PORT:-8070}"
SKIP_SERVE="${SKIP_SERVE:-0}"
MODEL_DIR="${MODEL_DIR:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
SPEC="${SPEC:-{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"model\":\"$MODEL_DIR\"}}"

if [ "$SKIP_SERVE" != "1" ]; then
  pid="$(cat "$LOGS/$TAG.pid" 2>/dev/null || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -KILL "$pid" 2>/dev/null || true
    [ -n "$pid" ] && pkill -9 -P "$pid" 2>/dev/null || true
  fi
  MODE=dsv4 TAG="$TAG" PORT="$PORT" TP=1 MAXLEN=262144 SEQS=128 MAX_NBT=8192 \
    GPU_UTIL=0.85 KV_DTYPE=fp8_ds_mla KV_MEM_BYTES=8589934592 \
    THREADS=192 OMP=96 EAGER=0 EP=0 GPUS=2 SPEC="$SPEC" \
    ENV_EXTRA="XIAOTU_CD_TIMING=1" \
    scripts/tune_serve.sh || { echo "[nat-curve] serve failed"; exit 1; }
fi

for L in 128 512 1024 4096; do
  L="$L" C=1 N=8 OUT=128 PORT="$PORT" SERVER_TAG="$TAG" TAG="nat${L}_c1_$(date +%H%M%S)" scripts/bench_nat.sh || echo "[nat-curve] c1 L=$L failed"
done
for L in 512 1024; do
  L="$L" C=2 N=8 OUT=128 PORT="$PORT" SERVER_TAG="$TAG" TAG="nat${L}_c2_$(date +%H%M%S)" scripts/bench_nat.sh || echo "[nat-curve] c2 L=$L failed"
done

{
  echo "### $(date -Is) tag=$TAG acceptance"
  grep -h "Mean acceptance length" "$LOGS/$TAG.log" 2>/dev/null | tail -25
} > "$LOGS/$TAG.acceptance"
echo "[nat-curve] done; acceptance -> $LOGS/$TAG.acceptance"
tail -4 "$LOGS/$TAG.acceptance"
