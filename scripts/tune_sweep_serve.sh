#!/usr/bin/env bash
# 串行参数扫描驱动:每个配置 = 启一次服务 + 一轮 ShareGPT 压测 + 关闭。
#
# 用法:
#   SWEEP="THREADS=48 THREADS=96 THREADS=192" bash scripts/tune_sweep_serve.sh
#   SWEEP="NBT=4096 NBT=16384" BASE="MODE=dsv4 MAXLEN=262144" C=64 OUT=128 \
#     bash scripts/tune_sweep_serve.sh
#
# 每个变体写成 report/tuning/summary.jsonl 的一行(server_tag 标明配置),
# 便于事后按 server_tag 汇总。BASE 里的键值对会作为所有变体的默认环境变量。
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin:$PATH

SWEEP="${SWEEP:-THREADS=96}"
BASE="${BASE:-MODE=dsv4 MAXLEN=262144 KV_DTYPE=fp8_ds_mla GPU_UTIL=0.90 KV_MEM_BYTES=12884901888 SEQS=64 MAX_NBT=8192 PREFILL_MIN=384 OMP=48}"
C="${C:-64}"; N="${N:-64}"; OUT="${OUT:-128}"
PORT="${PORT:-8081}"
# 每个变体用独立端口 + 先杀掉占用该端口的旧进程:否则新服务可能因为端口被上一轮
# 残留进程占着而起不来,客户端却连到旧进程上(实测导致 "failed=64")。
PORT_BASE="${PORT_BASE:-$PORT}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-xiaotu}"
TOKENIZER="${TOKENIZER:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"

kill_servers() {
  local pids used i
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  for p in $pids; do kill -9 "$p" 2>/dev/null; done
  # vLLM 在启动时会检查"空闲显存 >= gpu_memory_utilization",必须等上一轮的
  # context 真正释放,否则新服务直接起不来(实测踩过两次)。
  for i in $(seq 1 60); do
    used=$(nvidia-smi --id="${GPUS:-2}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1024 ] && break
    sleep 5
  done
}

VI=0
for variant in $SWEEP; do
  VI=$((VI + 1))
  PORT=$((PORT_BASE + VI - 1))
  _holder=$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $NF}' | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
  [ -n "${_holder:-}" ] && kill -9 "$_holder" 2>/dev/null && sleep 3
  # BASE 与 variant 都是 "K=V" 形式,逐个 export
  for kv in $BASE $variant; do export "$kv"; done
  TAG="dsv4_sweep_$(echo "$variant" | tr 'A-Z=' 'a-z-')"
  echo "=== [$(date +%H:%M:%S)] variant=$variant tag=$TAG"
  kill_servers
  MODE="${MODE:-dsv4}" TAG="$TAG" PORT="$PORT" TP="${TP:-1}" GPUS="${GPUS:-2}" \
    MAXLEN="$MAXLEN" KV_DTYPE="$KV_DTYPE" GPU_UTIL="$GPU_UTIL" KV_MEM_BYTES="$KV_MEM_BYTES" \
    SEQS="$SEQS" MAX_NBT="$MAX_NBT" PREFILL_MIN="$PREFILL_MIN" THREADS="$THREADS" OMP="$OMP" \
    EAGER="${EAGER:-1}" ENV_EXTRA="${ENV_EXTRA:-}" \
    bash "$ROOT/scripts/tune_serve.sh" > "$ROOT/report/tuning/logs/$TAG.serve.log" 2>&1
  # 自己再确认一次就绪(tune_serve 可能因为端口上还有旧进程而提前返回)
  READY=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then READY=1; break; fi
    sleep 5
  done
  if [ "$READY" != "1" ]; then
    echo "!!! $TAG failed to start (see logs/$TAG.log)"; continue
  fi
  TAG="${TAG}_c${C}_out${OUT}" PORT="$PORT" MODEL="$MODEL_NAME" SERVER_TAG="$TAG" \
    C="$C" N="$N" OUT="$OUT" TOKENIZER="$TOKENIZER" \
    bash "$ROOT/scripts/tune_client.sh" 2>&1 | grep "tune_client\]" || true
done
kill_servers
echo "=== sweep done"
