#!/usr/bin/env bash
# **我们自己环境**(vllm 主线 + 插件 vllm_xiaotu_moe)下 DeepSeek-V4.1-Flash 的常驻内存足迹。
#
# 与 `ref_v41_mem.sh`(参考实现)用**同一把尺子**(mem_footprint.sh = 服务进程树总 RSS),
# 但跑在 `vllm-xiaotu-moe` env、用本仓库的插件(edditable 安装,不走 .pth)。
#
# env/参数逐条对齐 `scripts/serve_v41.sh` 的默认,只把三处改成"可测量"的形态:
#   LOAD: dummy -> **auto**(真实权重;dummy 量不出内存问题)
#   TP:   1 -> **2**(要看的正是"两个 rank 是否各存一份")
#   MAXLEN: 2048 -> **8192**(足够真实又不让 KV 主导)
#
# 用法:
#   TAG=xtuown_w1 WCOPY=1 bash report/tuning/probes/xtu_own_v41_mem.sh
#   TAG=xtuown_w0 WCOPY=0 bash report/tuning/probes/xtu_own_v41_mem.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8210}"
TP="${TP:-2}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXLEN="${MAXLEN:-8192}"
SEQS="${SEQS:-1}"
THREADS="${THREADS:-60}"   # 与 serve_v41.sh 的新默认一致(TP=2 ⇒ 每 rank 12 CCD × 5;184 是单进程拐点,TP=2 下会 368 线程超订)
TAG="${TAG:-xtuown}"
WCOPY="${WCOPY:-}"          # 留空 = 用代码的条件默认(GPU 预填充开则 1,关则 0)
RELEASE_SOURCE="${RELEASE_SOURCE:-1}"
SPIN="${SPIN:-0}"
GP_MIN="${GP_MIN:-0}"      # VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS:>0 时 qlen>=该值 的层走 GPU 预填充
RESIDENT="${RESIDENT:-}"  # XIAOTU_MOE_GPU_RESIDENT_LAYERS,如 0-3,20-22
SPEC="${SPEC:-0}"          # 1 = 开 DSpark 投机(draft=mtp,默认常驻 GPU)
MBT="${MBT:-}"            # --max-num-batched-tokens(留空 = 不传)   # serve_v41.sh 的默认;0 = 完全不自旋(serve_mainline.sh 的默认,§355 证明 5000 是正反馈灾难)
RANK_SPLIT="${RANK_SPLIT:-2}"   # 与代码默认一致(§505:按 socket 分片 + CCD 交错)   # 1=按 node 子集切(默认;NPS1+TP2 会退化成 2×socket 副本) 2=CCD 交错(每 rank 覆盖全部 node ⇒ nshard=node 数 ⇒ 1 份权重)

# per-TAG env 桥(§497:不许再让全局 /tmp/xiaotu_env 污染)
XV="$ROOT/report/tuning/logs/$TAG.envfile"
{
  echo "XIAOTU_MOE_THREADS=$THREADS"
  echo "XIAOTU_MOE_NSLICE_SMALL=0"
  echo "XIAOTU_MOE_ASYNC=0"
  echo "XIAOTU_MOE_SPIN_IDLE_US=$SPIN"
  echo "XIAOTU_MOE_GPU_RESIDENT_LAYERS="
  echo "XIAOTU_MOE_RESIDENT_BUDGET_GB=0"
  echo "XIAOTU_RELEASE_SOURCE=$RELEASE_SOURCE"
  echo "${WCOPY:+XIAOTU_GPUPREFILL_WCOPY=$WCOPY}"
  echo "XIAOTU_MOE_RANK_SPLIT=$RANK_SPLIT"
  echo "XIAOTU_MOE_GPU_RESIDENT_LAYERS=$RESIDENT"
  echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GP_MIN"
} > "$XV"
export TAG PORT

RUN_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS \
XIAOTU_ENV_FILE=$XV VLLM_EXPERTS_LOAD_DEVICE=cpu OMP_NUM_THREADS=1 \
XIAOTU_MOE_THREADS=$THREADS XIAOTU_MOE_NSLICE_SMALL=0 XIAOTU_MOE_ASYNC=0 \
XIAOTU_MOE_SPIN_IDLE_US=$SPIN XIAOTU_MOE_GPU_RESIDENT_LAYERS=$RESIDENT \
XIAOTU_MOE_RESIDENT_BUDGET_GB=0 XIAOTU_RELEASE_SOURCE=$RELEASE_SOURCE \
${WCOPY:+XIAOTU_GPUPREFILL_WCOPY=$WCOPY} XIAOTU_MOE_RANK_SPLIT=$RANK_SPLIT VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GP_MIN \
FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ENGINE_READY_TIMEOUT_S=3600"

RUN_ENV="$RUN_ENV" READY_TIMEOUT="${READY_TIMEOUT:-2400}" \
  bash "$ROOT/report/tuning/probes/mem_footprint.sh" \
  numactl --interleave=all "$ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name dsv41-xtu --host 0.0.0.0 --port "$PORT" \
  --load-format auto --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" \
  $( [ -n "$MBT" ] && echo --max-num-batched-tokens "$MBT" ) \
  --trust-remote-code --enable-prefix-caching --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kernel-config '{"enable_jit_warmup": false}' \
  $( [ "$SPEC" = "1" ] && echo --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' ) \
