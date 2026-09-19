#!/usr/bin/env bash
# GLM-5.3-Flash on mainline vLLM + the `vllm-xiaotu-moe` plugin (A100 / SM80).
#
# What this configuration depends on (all verified in dev-docs/GLM53_SM80_PLAN.md):
#   * The 11 sparse-MLA (DSA) layers need the SM8x Triton backend, which the
#     model binds only when `sm8x_sparse_mla_enabled()` is true -> this requires
#     **--kv-cache-dtype bfloat16** (the SM8x route is bf16-only; fp8/fp4 KV
#     stays on the upstream SM90+ pool and fails closed).
#   * Routed expert weights live on the host and are computed by the xiaotu
#     engine: VLLM_EXPERTS_LOAD_DEVICE=cpu + the plugin's CPU backend swap.
#     The native FP8 checkpoint's block-128 experts take the FP8 kernel.
#   * The DSA (full-attention) projections are FP8 in the checkpoint but the
#     model keeps them BF16 and dequantizes on load (glm5next model.py).
#
# Usage:
#   bash scripts/serve_glm53_mainline.sh
#   TAG=glm53_t1 PORT=8073 MAXLEN=8192 MBT=2048 SEQS=8 bash scripts/serve_glm53_mainline.sh
#   COMPILE=1 ...   # add a compilation config (slow first start; off for bring-up)
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/ZhipuAI--GLM-5.3-Flash/snapshots/master}"

TAG="${TAG:-glm53_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8073}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXLEN="${MAXLEN:-4096}"
MBT="${MBT:-1024}"
SEQS="${SEQS:-4}"
THREADS="${THREADS:-60}"
KV_DTYPE="${KV_DTYPE:-bfloat16}"
# GPU-prefill threshold. The plugin's own default is 4096, which is *above*
# GLM-5.3's prefill chunk size: the KDA (mamba-like) state is only written at
# `block_size = 2176` boundaries, so the scheduler trims every chunk to a
# multiple of 2176 no matter how large --max-num-batched-tokens is. With the
# side-stream assembly overlap the measured break-even is ~1300 tokens/chunk
# (1580 tokens: CPU 10.9 s vs GPU 10.2 s; 2176 tokens: 1.2x; ~4.1k: 1.26x), so
# 1500 is the right default here. Lower it further only if you measure it.
GPU_PREFILL_MIN="${GPU_PREFILL_MIN:-1500}"
INTERLEAVE="${INTERLEAVE:-1}"
COMPILE="${COMPILE:-0}"
OUTDIR="${OUTDIR:-$ROOT/logs}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"

ARGS=(
  --model "$CKPT"
  --served-model-name GLM-5.3-Flash
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP"
  --dtype bfloat16
  --kv-cache-dtype "$KV_DTYPE"
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "$MBT"
  --max-num-seqs "$SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-chunked-prefill
  --disable-custom-all-reduce
)
if [ "$COMPILE" = "1" ]; then
  ARGS+=(--compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY"}')
fi

# per-TAG env bridge (see vllm_xiaotu_moe/__init__.py): the EngineCore child does
# not inherit every XIAOTU_* var, so write the ones we need to a TAG-local file.
XTU_ENV_FILE="${XTU_ENV_FILE_OVERRIDE:-$OUTDIR/$TAG.envfile}"
{
  echo "XIAOTU_MOE_THREADS=$THREADS"
  echo "XIAOTU_MOE_SPIN_IDLE_US=${SPIN_IDLE_US:-300}"
  # Only the FILE is authoritative (XIAOTU_ENV_FILE is set explicitly below),
  # so the threshold has to be written here to reach the EngineCore child.
  echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GPU_PREFILL_MIN"
  [ -n "${XIAOTU_MOE_NSLICE_SMALL:-}" ] && echo "XIAOTU_MOE_NSLICE_SMALL=$XIAOTU_MOE_NSLICE_SMALL"
  [ -n "${XIAOTU_MOE_ASYNC:-}" ] && echo "XIAOTU_MOE_ASYNC=$XIAOTU_MOE_ASYNC"
  [ -n "${XIAOTU_MOE_RANK_SPLIT:-}" ] && echo "XIAOTU_MOE_RANK_SPLIT=$XIAOTU_MOE_RANK_SPLIT"
  [ -n "${XIAOTU_MOE_NOSHARD:-}" ] && echo "XIAOTU_MOE_NOSHARD=$XIAOTU_MOE_NOSHARD"
  [ -n "${XIAOTU_MOE_GEMM_FP8_SCALE:-}" ] && echo "XIAOTU_MOE_GEMM_FP8_SCALE=$XIAOTU_MOE_GEMM_FP8_SCALE"
} > "$XTU_ENV_FILE"
# Pass through every other XIAOTU_*/VLLM_XIAOTU_* variable from the caller's
# environment. `XIAOTU_ENV_FILE` is set explicitly below, so the plugin takes the
# file as authoritative and *overwrites* os.environ with it -- anything not listed
# above is therefore dropped on the floor for the EngineCore child. That silently
# broke runtime A/Bs: exporting VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE here and
# expecting the worker to see it measured the same path twice (the same class of
# bug the plugin's own comment at `gpu_prefill_min_tokens` warns about).
while IFS='=' read -r _k _v; do
  case "$_k" in
    XIAOTU_ENV_FILE) continue ;;
    XIAOTU_*|VLLM_XIAOTU_*) ;;
    *) continue ;;
  esac
  grep -q "^${_k}=" "$XTU_ENV_FILE" 2>/dev/null || echo "${_k}=${_v}"
done < <(env) >> "$XTU_ENV_FILE"
export XIAOTU_ENV_FILE="$XTU_ENV_FILE"
echo "[glm53] env-bridge -> $XTU_ENV_FILE"

if [ "$INTERLEAVE" = "1" ] && command -v numactl >/dev/null 2>&1; then
  NCTL=(numactl --interleave=all)
else
  NCTL=()
fi
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "[glm53] tag=$TAG port=$PORT gpus=$GPUS tp=$TP maxlen=$MAXLEN mbt=$MBT seqs=$SEQS"
echo "[glm53] kv-cache-dtype=$KV_DTYPE (SM8x sparse-MLA is bf16-only)"
echo "[glm53] log=$LOG"

nohup env \
  HF_HUB_OFFLINE=1 \
  VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
  VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-60}" \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_EXPERTS_LOAD_DEVICE=cpu \
  XIAOTU_MOE_THREADS="$THREADS" \
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$GPU_PREFILL_MIN" \
  XIAOTU_MOE_SPIN_IDLE_US="${SPIN_IDLE_US:-300}" \
  OMP_NUM_THREADS=1 \
  "${NCTL[@]}" "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[glm53] pid=$(cat "$OUTDIR/$TAG.pid")"

for _ in $(seq 1 1080); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[glm53] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[glm53] server exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[glm53] timeout; tail:"; tail -40 "$LOG"; exit 1
