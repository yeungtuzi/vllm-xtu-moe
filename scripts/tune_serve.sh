#!/usr/bin/env bash
# 调参统一服务端启动器(vllm-xtu-moe · 混合模式)
#
# 所有可变项都从环境变量进来,便于把"一次实验"记录成一行可复现的命令。
# 启动后会等待 /v1/models 就绪,并把关键日志行(加载耗时、KV 容量、后端/引擎形状)
# 写到 report/tuning/logs/<TAG>.meta。
#
# 用法(MODE=dsv4|qwen38):
#   MODE=dsv4 TAG=dsv4_tp1_fp8kv PORT=8081 TP=1 MAXLEN=262144 KV_DTYPE=fp8_ds_mla \
#     GPU_UTIL=0.90 SEQS=8 MAX_NBT=8192 scripts/tune_serve.sh
#
# 常用可选变量:
#   EP=1 启用专家并行;CPU_OFFLOAD_GB=N(仅 qwen38 需要);EAGER=0 关闭 enforce-eager;
#   PREFILL_MIN=N  → VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS;PREFILL_FILE=path 走运行期阈值文件;
#   THREADS=N → XIAOTU_MOE_THREADS;OMP=N → OMP_NUM_THREADS;SPEC='{...}' → --speculative-config;
#   GPUS=2 / GPUS=0,1 选择 GPU;KV_MEM_BYTES=N 显式限制 KV cache 字节数;
#   EXTRA='...' 追加任意参数。
#
# License: Apache-2.0
set -euo pipefail

MODE="${MODE:-dsv4}"
TAG="${TAG:-${MODE}_$(date +%m%d_%H%M%S)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$ROOT/report/tuning/logs"
mkdir -p "$OUTDIR"

PORT="${PORT:-8081}"
MAXLEN="${MAXLEN:-262144}"
SEQS="${SEQS:-8}"
MAX_NBT="${MAX_NBT:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
KV_DTYPE="${KV_DTYPE:-auto}"
EP="${EP:-0}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
EAGER="${EAGER:-1}"
TP="${TP:-1}"
PREFILL_MIN="${PREFILL_MIN:-384}"
PREFILL_FILE="${PREFILL_FILE:-}"
THREADS="${THREADS:-96}"
OMP="${OMP:-48}"
SPEC="${SPEC:-}"
EXTRA="${EXTRA:-}"
ENV_EXTRA="${ENV_EXTRA:-}"
KV_MEM_BYTES="${KV_MEM_BYTES:-}"
# 选择性地把参数段 offload 到 CPU(vLLM 的 UVA offloader):Qwen3.8 的 PLE n-gram
# 表(51.2 GB)在 CPU 内存里做查找,显存只留 dense/attention(~11 GB)→ 单卡可跑。
OFFLOAD_PARAMS="${OFFLOAD_PARAMS:-}"

case "$MODE" in
  dsv4)
    MODEL="${MODEL:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
    SERVED="${SERVED:-DeepSeek-V4-Flash-xiaotu}"
    GPUS="${GPUS:-2}"
    ;;
  qwen38)
    MODEL="${MODEL:-/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13}"
    SERVED="${SERVED:-Qwen3.8-Flash-Next}"
    GPUS="${GPUS:-0,1}"
    ;;
  *) echo "MODE must be dsv4 or qwen38" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export XIAOTU_MOE_SINGLECOPY="${XIAOTU_MOE_SINGLECOPY:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="$OMP"
[ -n "$THREADS" ] && export XIAOTU_MOE_THREADS="$THREADS"
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$PREFILL_MIN"
[ -n "$PREFILL_FILE" ] && export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE="$PREFILL_FILE"
export PATH=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin:$PATH
# ENV_EXTRA 形如 "XIAOTU_MOE_PROFILE=1 XIAOTU_TIMING=1":临时打开埋点
for _kv in $ENV_EXTRA; do export "$_kv"; done

ARGS=(
  "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --max-num-batched-tokens "$MAX_NBT"
  --gpu-memory-utilization "$GPU_UTIL"
  --no-enable-prefix-caching
  --kernel-config.enable_jit_warmup=false
  --served-model-name "$SERVED"
)
[ "$EP" = "1" ] && ARGS+=(--enable-expert-parallel)
[ "$KV_DTYPE" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
# 显式给 KV cache 设上限:256K 上下文只需要 ~8 GiB,其余显存要留给
# GPU prefill 的 ping-pong staging(每层 ~2 GiB)与激活,否则会 OOM。
[ -n "$KV_MEM_BYTES" ] && ARGS+=(--kv-cache-memory-bytes "$KV_MEM_BYTES")
[ "$CPU_OFFLOAD_GB" != "0" ] && ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
if [ -n "$OFFLOAD_PARAMS" ]; then
  # shellcheck disable=SC2206
  _opts=($OFFLOAD_PARAMS)
  ARGS+=(--cpu-offload-params "${_opts[@]}")
fi
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

# 记录本次实验的"环境快照"(机器负载会显著影响数字)
{
  echo "tag=$TAG mode=$MODE port=$PORT tp=$TP ep=$EP maxlen=$MAXLEN seqs=$SEQS nbt=$MAX_NBT"
  echo "gpu_util=$GPU_UTIL kv_dtype=$KV_DTYPE cpu_offload_gb=$CPU_OFFLOAD_GB eager=$EAGER"
  echo "prefill_min=$PREFILL_MIN prefill_file=${PREFILL_FILE:-none} threads=$THREADS omp=$OMP"
  echo "spec=${SPEC:-none} extra=${EXTRA:-none} env_extra=${ENV_EXTRA:-none} gpus=$GPUS"
  date -Is
  uptime
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} > "$OUTDIR/$TAG.env"

# 等显存真正释放(上一轮进程可能还在退出中),避免 "Free memory ... less than desired"
for _g in $(echo "$GPUS" | tr ',' ' '); do
  for _i in $(seq 1 60); do
    _used=$(nvidia-smi --id="$_g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$_used" ] && [ "$_used" -lt 1024 ] && break
    sleep 5
  done
done
LOG="$OUTDIR/$TAG.log"
nohup vllm serve "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[tune_serve] tag=$TAG pid=$(cat "$OUTDIR/$TAG.pid") log=$LOG"

READY=0
for _ in $(seq 1 720); do          # 最长等 60 分钟(超大模型加载)
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[tune_serve] server exited early; tail:"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done
if [ "$READY" != "1" ]; then
  echo "[tune_serve] timeout waiting for readiness; tail:"; tail -30 "$LOG"; exit 1
fi

{
  grep -E "init engine .* took|GPU KV cache size|Using CPU .*MoE backend|xiaotu MOE_" "$LOG" | tail -8
  echo "ready_at=$(date -Is)"
} > "$OUTDIR/$TAG.meta"
echo "[tune_serve] READY tag=$TAG"
cat "$OUTDIR/$TAG.meta"
