#!/usr/bin/env bash
# Test server for the vllm-xiaotu-moe plugin on GPU2 / port 8071.
#
# Hard constraints (user directive):
#   * GPU2 ONLY (never GPU0/1, never prod 8070)
#   * offline (HF_HUB_OFFLINE=1)
#   * CPU parallelism <= 96 threads
#   * does NOT touch the production server; do not call prod while measuring perf
set -euo pipefail

MODEL="${MODEL:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
PORT="${PORT:-8071}"
MAXLEN="${MAXLEN:-1024}"
SEQS="${SEQS:-64}"
OMP="${OMP:-96}"

source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate vllm-xiaotu-moe

export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export OMP_NUM_THREADS="$OMP"
export CUDA_VISIBLE_DEVICES=2
# Engine memory: hold the weights exactly once (no per-socket replicas / shard
# copies). Matches the reference lk engine footprint instead of ~2-3x copies.
export XIAOTU_MOE_SINGLECOPY="${XIAOTU_MOE_SINGLECOPY:-1}"
# Per-phase (A/A2/B0/B/C) breakdown, printed to stderr every 40 engine calls.
export XIAOTU_MOE_PROFILE="${XIAOTU_MOE_PROFILE:-1}"
# Long-prefill GPU MoE threshold (tokens). 0 disables; >= this a layer's MoE is
# computed on GPU by streaming that layer's weights H2D. DS-V4 benefits from a
# few-K threshold (per-layer H2D cost), matching ktransformers
# KT_GPU_PREFILL_TOKEN_THRESHOLD / fork LVLLM_GPU_PREFILL_MIN_BATCH_SIZE.
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="${VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS:-0}"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$SEQS" \
  --max-num-batched-tokens "${MAX_NBT:-4096}" \
  --kernel-config.enable_jit_warmup=false \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  ${SPEC_ARGS:-} \
  --served-model-name DeepSeek-V4-Flash-xiaotu \
  --trust-remote-code
