#!/usr/bin/env bash
# 目标场景基准:单卡 256K + DSpark(k=5)+ CUDA graph,自然文本、精确上下文长度。
# 一条命令跑完 上下文 × 并发 的曲线,并抓取服务端逐位置接受率。
#
# 用法:scripts/run_nat_curve.sh          # 起服务 + 跑曲线 + 关服务
#      SKIP_SERVE=1 scripts/run_nat_curve.sh   # 复用已在跑的服务
#
# License: Apache-2.0
set -uo pipefail
# ⭐ 唯一真源:归属证据 + 安全按 PID 停止(2026-10-07 审计修 F2)
_SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$_SD/lib_proc_identity.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/dev-docs/report/tuning/logs"
TAG="${TAG:-nat_k5}"
PORT="${PORT:-8096}"
SKIP_SERVE="${SKIP_SERVE:-0}"
MODEL_DIR="${MODEL_DIR:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
SPEC="${SPEC:-{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\",\"model\":\"$MODEL_DIR\"}}"

if [ "$SKIP_SERVE" != "1" ]; then
  pid="$(cat "$LOGS/$TAG.pid" 2>/dev/null || true)"
  # ⭐ N2 修:本行在【顶层】(不在函数里),`return 1` 只会报错、**不会中止** ✗
#   ⇒ 必须显式 exit 1(否则"停不掉/被拒"之后仍会继续启动,与 F2 的原意相反)
pi_stop_pid_safe "$pid" || { echo "[nat-curve] ⛔ 停止旧实例被拒/未停 ⇒ 中止,不启动" >&2; exit 1; }
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
