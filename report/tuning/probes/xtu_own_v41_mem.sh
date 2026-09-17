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
SPIN="${SPIN:-300}"
GP_MIN="${GP_MIN:-0}"      # VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS:>0 时 qlen>=该值 的层走 GPU 预填充
RESIDENT="${RESIDENT:-}"  # XIAOTU_MOE_GPU_RESIDENT_LAYERS,如 0-3,20-22
RESIDENT_BUDGET_GB="${RESIDENT_BUDGET_GB:-}"   # 常驻额度(§509:必须在 KV 之后预算,否则会挤掉 1M 上下文)
SPEC="${SPEC:-0}"
COMPILE="${COMPILE:-0}"
EAGER="${EAGER:-0}"      # 1 = --enforce-eager(关 CUDA graph)。【§600】我们实测 CUDA graph
                          #     **没有收益**(§584:43.19→44.73ms),但它会给每个形状建**持久池**
                          #     (实测首请求 +7.7 GiB 常驻)⇒ 大 chunk 的 GPU 预填充应该关掉它换显存。
SHARD_BY_NODE="${SHARD_BY_NODE:-}"   # 非空 = 设 XIAOTU_MOE_SHARD_BY_NODE ⇒ **按 NUMA node 分片**(nshard=NPS 的 node 数)
                                     # 留空 = 代码默认 **按 socket 分片**(nshard=2)。
                                     # 【2026-09-17 实测】B=8 时 node 分片比 socket 分片快 **4.7×**
                                     # (1.76 vs 8.25 ms/层),B=1 快 1.54×(0.37 vs 0.57) —— 见 NOTES §518。
CD_TIMING="${CD_TIMING:-0}"       # 1 = 打开**引擎侧**每层分相计时([cd-timing] period/compute/ep/rest §515)
                                   #   比 LAYER_TIMING 更有用:它写在引擎回调里,所以 **cudagraph replay 时也会打**
                                   #   (Python 的 apply() 在 replay 时不执行 ⇒ LAYER_TIMING 量不到稳态解码)。
LAYER_TIMING="${LAYER_TIMING:-0}"   # 1 = 打开每层分相计时(§515)      # 1 = 参考脚本的 --compilation-config {"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY"}          # 1 = 开 DSpark 投机(draft=mtp,默认常驻 GPU)
ITERDETAIL="${ITERDETAIL:-0}"  # 1 = --enable-logging-iteration-details(每步分解,§587)
GPF_V2="${GPF_V2:-0}"        # 1 = XIAOTU_GPF_V2=1:staging 分批版(DMA 全发完再组装/转置,§595)
GPF_STAGE="${GPF_STAGE:-0}"  # 1 = XIAOTU_GPF_STAGE=1:staging **子相**计时(§594 dma/asm/tr)
GP_SPLIT="${GP_SPLIT:-0}"    # 1 = XIAOTU_GP_SPLIT=1:每层打印 GPU 预填充的 **staging(asm)**
                             #     与 **kernels** 分相耗时(§594;用来定位 11.6s/chunk 的构成)
KVSHARE="${KVSHARE:-0}"    # 1 = 传 --kv-sharing-fast-prefill(§574/§575 的 ③ 实验)
PREFIX_CACHE="${PREFIX_CACHE:-1}"   # 0 = --no-enable-prefix-caching(用于"预填充可复现性"对照实验,§572)
KV_CACHE_BYTES="${KV_CACHE_BYTES:-}"  # 非空 = 显式 `--kv-cache-memory`(**封顶 KV 池**)。
                                      # 【§592 必需】vLLM 默认会把 KV 池填到 `--gpu-memory-utilization` 为止,
                                      # 启动后净空只剩 (1-util)×39.49 GiB ⇒ util=0.90 时只有 3.95 GiB,
                                      # 而 GPU 预填充的 preflight 要 8.4 GiB ⇒ **逐层被拒、静默退回 CPU**。
                                      # 显式封顶后,余量才留给预填充/投机。取什么值见
                                      # `python -m vllm_xiaotu_moe.vram_policy --maxlen M --max-num-seqs N`。
PROFILE_DIR="${PROFILE_DIR:-}"   # 非空 = 开 torch profiler;
                                 #   服务起来后 `curl -X POST localhost:$PORT/start_profile` / `/stop_profile`,
                                 #   产物是 chrome trace(默认 /tmp/<dir>/<...>.pt.trace.json.gz)。
                                 #   用途:把一步解码拆到 kernel 粒度(engram 查表 / MoE / 编排空档),见 §564。
                                 #   ⚠️ 本版 vLLM 的光有 `VLLM_TORCH_PROFILER_DIR` **不够**:HTTP 路由只在
                                 #   `profiler_config.profiler is not None` 时挂载(serve/profile/api_router.py:36),
                                 #   所以必须同时传 `--profiler-config '{"profiler":"torch",...}'`(否则 /start_profile 404)。
MBT="${MBT:-}"            # --max-num-batched-tokens(留空 = 不传)   # serve_v41.sh 的默认;0 = 完全不自旋(serve_mainline.sh 的默认,§355 证明 5000 是正反馈灾难)
RANK_SPLIT="${RANK_SPLIT:-}"   # **留空 = 交给引擎的自适应判据**(§519:node 数/world>=2 ⇒ node 分片 + 核按 node 子集切)。
                              # 【为什么改掉默认的 2】脚本原来硬写 `=2`(§505 的 socket+CCD 交错),那会**覆盖**掉引擎的
                              # 自适应默认,把服务钉死在 socket 模式 —— 实测 TP=2 上比 node 模式慢 **+29%**
                              # (14.17 → 18.22 t/s,§518)。要复现旧行为才显式给 `RANK_SPLIT=2`。

# ---- JIT 固定缓存目录(用户 2026-09-17 要求;原理见 scripts/lib_jitcache.sh)--------
# COMPILE=1 时 vLLM 会把 TRITON_CACHE_DIR 重定向进按 hash 算出的目录,而那个 hash 把
# **每个 VLLM_* 环境变量**和我们插件源码都算进去了 ⇒ 换旗标/改一行代码就要逐形状重新 JIT。
# 这里把目录钉死(名里带模型/TP/模式/我们源码指纹/vLLM commit ⇒ 该换时会自动换)。
JITCACHE_MODEL="$(basename "$CKPT")"
JITCACHE_TP="$TP"
JITCACHE_MODE="$([ "$COMPILE" = "1" ] && echo vllm_compile || echo none)"
JITCACHE_COMPILING="$COMPILE"
. "$ROOT/scripts/lib_jitcache.sh"
JIT_ENV=""
if [ -n "${TRITON_CACHE_DIR:-}" ]; then
  JIT_ENV="TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
  [ -n "${TILELANG_CACHE_DIR:-}" ] && JIT_ENV="$JIT_ENV TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR"
fi

# per-TAG env 桥(§497:不许再让全局 /tmp/xiaotu_env 污染)
XV="$ROOT/report/tuning/logs/$TAG.envfile"
{
  echo "XIAOTU_MOE_THREADS=$THREADS"
  echo "XIAOTU_MOE_NSLICE_SMALL=0"
  echo "XIAOTU_MOE_ASYNC=0"
  echo "XIAOTU_MOE_SPIN_IDLE_US=$SPIN"
  echo "XIAOTU_MOE_RESIDENT_BUDGET_GB=0"
  echo "XIAOTU_RELEASE_SOURCE=$RELEASE_SOURCE"
  echo "XIAOTU_ENGRAM_LAST=${XIAOTU_ENGRAM_LAST:-1}"
  echo "${WCOPY:+XIAOTU_GPUPREFILL_WCOPY=$WCOPY}"
  [ -n "$RANK_SPLIT" ] && echo "XIAOTU_MOE_RANK_SPLIT=$RANK_SPLIT"
  echo "XIAOTU_MOE_GPU_RESIDENT_LAYERS=$RESIDENT"
  echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GP_MIN"
  [ "$LAYER_TIMING" = "1" ] && echo "XIAOTU_LAYER_TIMING=1"
  [ -n "$SHARD_BY_NODE" ] && echo "XIAOTU_MOE_SHARD_BY_NODE=$SHARD_BY_NODE"
  [ "$GPF_V2" = "1" ] && echo "XIAOTU_GPF_V2=1"
  [ "$GPF_STAGE" = "1" ] && echo "XIAOTU_GPF_STAGE=1"
  [ "$GP_SPLIT" = "1" ] && echo "XIAOTU_GP_SPLIT=1"
  [ "$CD_TIMING" = "1" ] && echo "XIAOTU_CD_TIMING=1"
  [ "$CD_TIMING" = "1" ] && echo "XIAOTU_CD_TIMING_EVERY=40"
  [ -n "$RESIDENT_BUDGET_GB" ] && echo "XIAOTU_MOE_RESIDENT_BUDGET_GB=$RESIDENT_BUDGET_GB"
} > "$XV"
export TAG PORT

RUN_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS \
XIAOTU_ENV_FILE=$XV VLLM_EXPERTS_LOAD_DEVICE=cpu OMP_NUM_THREADS=1 \
XIAOTU_MOE_THREADS=$THREADS XIAOTU_MOE_NSLICE_SMALL=0 XIAOTU_MOE_ASYNC=0 \
XIAOTU_MOE_SPIN_IDLE_US=$SPIN XIAOTU_MOE_GPU_RESIDENT_LAYERS=$RESIDENT \
XIAOTU_MOE_RESIDENT_BUDGET_GB=0 XIAOTU_RELEASE_SOURCE=$RELEASE_SOURCE \
XIAOTU_ENGRAM_LAST=${XIAOTU_ENGRAM_LAST:-1} \
${WCOPY:+XIAOTU_GPUPREFILL_WCOPY=$WCOPY} ${RANK_SPLIT:+XIAOTU_MOE_RANK_SPLIT=$RANK_SPLIT} VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GP_MIN \
XIAOTU_LAYER_TIMING=$LAYER_TIMING XIAOTU_LAYER_TIMING_EVERY=40 \
${GPF_V2:+XIAOTU_GPF_V2=$GPF_V2} ${GPF_STAGE:+XIAOTU_GPF_STAGE=$GPF_STAGE} ${GP_SPLIT:+XIAOTU_GP_SPLIT=$GP_SPLIT} \
${CD_TIMING:+XIAOTU_CD_TIMING=$CD_TIMING} ${CD_TIMING:+XIAOTU_CD_TIMING_EVERY=40} \
${SHARD_BY_NODE:+XIAOTU_MOE_SHARD_BY_NODE=$SHARD_BY_NODE} \
${RESIDENT_BUDGET_GB:+XIAOTU_MOE_RESIDENT_BUDGET_GB=$RESIDENT_BUDGET_GB} \
${PROFILE_DIR:+VLLM_TORCH_PROFILER_DIR=$PROFILE_DIR} \
$JIT_ENV \
FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ENGINE_READY_TIMEOUT_S=3600"

RUN_ENV="$RUN_ENV" READY_TIMEOUT="${READY_TIMEOUT:-2400}" \
  bash "$ROOT/report/tuning/probes/mem_footprint.sh" \
  numactl --interleave=all "$ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name dsv41-xtu --host 0.0.0.0 --port "$PORT" \
  --load-format auto --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" \
  $( [ -n "$MBT" ] && echo --max-num-batched-tokens "$MBT" ) \
  --trust-remote-code --enable-chunked-prefill \
  $( [ "$PREFIX_CACHE" = "1" ] && echo --enable-prefix-caching || echo --no-enable-prefix-caching ) \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --kernel-config '{"enable_jit_warmup": false}' \
  $( [ -n "$KV_CACHE_BYTES" ] && printf -- '--kv-cache-memory %s' "$KV_CACHE_BYTES" ) \
  $( [ "$KVSHARE" = "1" ] && echo --kv-sharing-fast-prefill ) \
  $( [ "$ITERDETAIL" = "1" ] && echo --enable-logging-iteration-details ) \
  $( [ -n "$PROFILE_DIR" ] && printf -- "--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"%s\",\"torch_profiler_with_stack\":false}" "$PROFILE_DIR" ) \
  $( [ "$EAGER" = "1" ] && echo --enforce-eager ) \
  $( [ "$COMPILE" = "1" ] && echo --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"mode\":\"VLLM_COMPILE\"${JITCACHE_CC_EXTRA}}" ) \
  $( [ "$SPEC" = "1" ] && echo --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' ) \
