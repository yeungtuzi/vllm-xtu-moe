#!/usr/bin/env bash
# 官方 `vllm bench serve` 扫描:**prompt 长度 × 并发**(用户 2026-09-17 指定的最终验收口径)。
#
# 用法:
#   PORT=8700 TAG=acc1 bash scripts/bench_serve_sweep.sh
#   LENS="32,256,1024" CONC="1,2" PORT=8700 TAG=smoke bash scripts/bench_serve_sweep.sh
#
# 设计要点(**都是踩过的坑**):
#   * `--ignore-eos` + 固定 `--random-output-len` ⇒ 输出长度可控,ITL/TPOT 才有意义;
#   * 每个组合用**独立 seed**,但长度/并发之间不共享 prompt(避免前缀缓存把数字变好看);
#   * `--num-prompts` 按长度自适应:`L<=4096` 用 `2*C`,更长的用 `C`
#     (32K×8 单轮就是 256K token 的预填充,再多跑不完);
#   * 结果落 `--save-result` 的 json,便于汇总;
#   * **必须充分预热**:先跑一轮丢弃(`WARMUP=1` 默认),否则首个形状的 Triton JIT 会污染 TTFT。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PORT="${PORT:-8700}"
TAG="${TAG:-acc}"
HOST="${HOST:-127.0.0.1}"
MODEL="${MODEL:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
SERVED="${SERVED:-dsv41-xtu}"   # ⚠️ 必须与服务的 --served-model-name 一致,否则全部请求 4xx(Successful=0)
LENS="${LENS:-32,256,1024,4096,16384,32768}"
CONC="${CONC:-1,2,4,8}"
OUTLEN="${OUTLEN:-128}"
WARMUP="${WARMUP:-1}"
OUTDIR="$ROOT/report/tuning/logs/bench_serve_$TAG"
mkdir -p "$OUTDIR"
PY="$ENV/bin/python"

run_one() {  # $1=len $2=conc $3=n $4=warm
  local L="$1" C="$2" N="$3" W="$4" RN=""
  [ "$W" = "1" ] && RN="warm_"
  XIAOTU_ENV_FILE=/dev/null "$PY" -m vllm.entrypoints.cli.main bench serve \
    --backend openai-chat --host "$HOST" --port "$PORT" \
    --model "$MODEL" --served-model-name "$SERVED" \
    --endpoint /v1/chat/completions \
    --dataset-name random --random-input-len "$L" --random-output-len "$OUTLEN" \
    --num-prompts "$N" --max-concurrency "$C" \
    --ignore-eos --seed "$((L + C))" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUTDIR" \
    --result-filename "${RN}L${L}_C${C}.json" 2>&1 | grep -E 'Successful|Benchmark duration|Request throughput|Output token throughput|Mean TTFT|Mean TPOT|Mean ITL' | sed "s/^/  [L=$L C=$C]/"
}

echo "[sweep] port=$PORT tag=$TAG lens=$LENS conc=$CONC outlen=$OUTLEN out=$OUTDIR"
if [ "$WARMUP" = "1" ]; then
  echo "[sweep] === 预热轮(丢弃:覆盖所有要报的形状,消掉 JIT)==="
  for L in ${LENS//,/ }; do
    [ "$L" -le 4096 ] && N=1 || N=1
    run_one "$L" 1 "$N" 1
  done
fi
echo "[sweep] === 正式轮 ==="
for L in ${LENS//,/ }; do
  for C in ${CONC//,/ }; do
    if [ "$L" -le 4096 ]; then N=$(( C * 2 )); else N="$C"; fi
    run_one "$L" "$C" "$N" 0
  done
done
echo "[sweep] 全部完成 → $OUTDIR"
