#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 3xA100-40GB (SM80) — bring-up / smoke runner.
#
# V4.1 is a 475 GiB model on 120 GB of VRAM, so nothing fits without offload.
# The split vLLM already supports (verified 2026-09-14, NOTES §369):
#   * routed experts (269 GiB)  -> host, computed by the xiaotu CPU engine
#                                  (VLLM_EXPERTS_LOAD_DEVICE=cpu + this plugin)
#   * Engram tables (188.8 GiB) -> pinned host memory (engram_config.cpu_offload,
#                                  on by default)
#   * everything else (~23 GiB)  -> GPU
#
# XIAOTU_RELEASE_SOURCE=1 (default here) implements 切分一层/释放一层: once a
# layer's engine owns its NUMA-sharded copy, the vLLM-side host source tensor is
# dropped, so the expert bytes are not held twice (IRON_RULES R9). Tune with
# XIAOTU_RELEASE_SOURCE=0 to A/B it.
# The one thing that does NOT work out of the box on A100 is attention: V4.1
# ships only FlashMLA (SM90+) and FlashInfer (SM100/SM120) paths. This script is
# the end-to-end acceptance test for the SM80 fallback.
#
# Usage:
#   bash scripts/serve_v41.sh                 # dummy weights (fast bring-up)
#   LOAD=auto bash scripts/serve_v41.sh       # real weights (~long load)
#
# Env: TAG PORT GPUS TP MAXLEN LOAD GPU_UTIL EXTRA_ENV MAXSEQS
#
# MBT(默认 **0 = 不传**,用 vLLM 默认 chunk 大小;如 8192):
#   **这是 GPU prefill 的真正前提**。vLLM 按 max_num_batched_tokens 把 prompt 切成
#   chunk,而"某层看到多少 token"= **chunk 大小**,不是 prompt 总长。所以
#   MBT=2048 时任何层都不可能超过 2048 ⇒ `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`
#   最多只能在 2048 这一档触发,而该档上**非重叠**的 GPU 流式(~400 ms/层)
#   仍慢于 CPU(实测 ~257 ms/层)⇒ 只会更慢(NOTES §463)。
#   要用 GPU prefill 必须同时把 MBT 提到 4096-8192(并相应留出 activation 显存)。
#   注意:vLLM 会读 `--max-num-batched-tokens` 反推 `max_num_seqs` 等预算,调大前先确认显存。
#
# SPEC(默认 **0 = 不启用**):开启 **DSpark** 投机解码(目标项 3)。
#   实测依据:参考实现 LvLLM-v2.5 在 V4.1 上报告 dspark 使 decode **+~48%**
#   (2x RTX3090 TP2:27 -> 26-40 t/s),而它用的就是**标准 vLLM 旗标**:
#     --speculative-config '{"method":"dspark","num_speculative_tokens":5,
#                            "draft_sample_method":"probabilistic"}'
#   我们的主线已内置 dspark(`vllm/config/speculative.py:68 DSparkModelTypes`,
#   以及 dspark_target_layer_ids / dspark_block_size 等校验),
#   且 V4.1 的 config 里 `dspark_block_size=5`、`dspark_target_layer_ids=[37,38,39]`
#   ⇒ **num_speculative_tokens 必须等于 5**。无需单独 draft model(旗标里没有 model 字段)。
#   可用 SPEC_CONFIG 覆盖;注意 draft 层要显存,配合 GPU_UTIL 一起调。见 NOTES §478。
SPEC_CONFIG="${SPEC_CONFIG:-{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\"}}"

# PREFIX_CACHE(默认 1 = 保持 vLLM 默认开启):设为 0 会加 --no-enable-prefix-caching。
#   **测量长上下文 prefill 时必须设 0** —— 否则不同 prompt 共享前缀会被整段命中,
#   量出"prefill 几千甚至上万 tok/s"的假象(NOTES §420 连续两次踩到)。
#
# 【性能默认已调】SPIN_IDLE_US 默认 5000(原 0):实测把 perf 里 ~42% 的 futex
#   (lll_lock_wait/wake)降到 ~0%,compute 0.43->0.37 ms/层,单流 25.7->27.5 tok/s。
#   见 NOTES §412。可用 SPIN 环境变量覆盖。
#
# EAGER(默认 **0** = 启用 CUDA graph):是否加 --enforce-eager。
#   **实测 +80%**:单流 14.31 -> 25.71 tok/s;每层 period 1.84 -> 0.95 ms,
#   其中 rest(GPU 侧算子)1.27 -> 0.52 ms —— 关掉 eager 省下的正是 batch=1 时
#   被 43 层放大的 kernel 启动开销。已在真实权重下验证启动完整、输出正确。
#   若遇到 CUDA graph 捕获问题(插件历史 TRIED_AND_REVERTED R4 提到
#   捕获期 cudaHostRegister 会作废捕获),用 EAGER=1 回退。见 NOTES §406。
#   实测(NOTES §405)每层 1.84 ms 里我们的 CPU MoE 只占 0.56 ms(31%),
#   69% 是 "rest"(GPU 侧算子 + 层间交接)。batch=1 时 GPU 算子极小,
#   kernel 启动开销被 43 层放大 ⇒ 关掉 eager(启用 CUDA graph)是头号候选优化。
#   默认仍为 1 以保持既有行为;测试用 EAGER=0。
#
# MAXSEQS(默认 1):并发批大小上限。**实测关键**:=1(原值)时 vLLM 同时只调度一个
#   序列,并发请求被完全串行化 —— 聚合吞吐恒定 ~13 tok/s、与并发无关,而单请求延迟
#   线性变差(C=8 时 64 token 要 21 s)。见 report/tuning/NOTES.md §400。
#   默认仍为 1 以保持既有行为不变;做吞吐测试时用 MAXSEQS=8/16 覆盖。
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
TAG="${TAG:-v41}"
PORT="${PORT:-8077}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
MAXLEN="${MAXLEN:-2048}"
LOAD="${LOAD:-dummy}"          # dummy = no disk read, exercises the kernels
GPU_UTIL="${GPU_UTIL:-0.85}"
EXTRA_ENV="${EXTRA_ENV:-}"

OUTDIR="$ROOT/report/tuning/logs"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"

if [ ! -f "$CKPT/config.json" ]; then
  echo "[v41] checkpoint not found: $CKPT" >&2; exit 2
fi

{
  echo "tag=$TAG port=$PORT gpus=$GPUS tp=$TP maxlen=$MAXLEN load=$LOAD util=$GPU_UTIL"
  echo "ckpt=$CKPT"; date -Is
} > "$OUTDIR/$TAG.env"

echo "[v41] starting (load=$LOAD maxlen=$MAXLEN tp=$TP gpus=$GPUS) log=$LOG"

# Optional memory sampler. V4.1 peaks near ~1.4 TB of host memory on a box whose
# 8 NUMA nodes are 193 GB each, so the failure mode is *one node* exhausting
# (kernel: "constraint=CONSTRAINT_MEMORY_POLICY, nodemask=N"), not total RAM.
# MEMTRACE=1 records per-node free memory plus the fattest process's
# Rss/Shmem/Pss so the next run says exactly which component grows where.
if [ "${MEMTRACE:-0}" = "1" ]; then
  MEMLOG="$OUTDIR/$TAG.mem"
  (
    while true; do
      echo "### $(date -Is)" >> "$MEMLOG"
      numactl --hardware 2>/dev/null | grep -E "^node [0-9]+ free" >> "$MEMLOG"
      fat=$(ps -eo pid,rss --sort=-rss --no-headers 2>/dev/null | head -1 | awk '{print $1}')
      if [ -n "$fat" ] && [ -r "/proc/$fat/smaps_rollup" ]; then
        echo "-- fattest pid=$fat" >> "$MEMLOG"
        grep -E "^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Dirty|Anonymous):" \
          "/proc/$fat/smaps_rollup" >> "$MEMLOG" 2>/dev/null
      fi
      sleep "${MEMTRACE_INTERVAL:-60}"
    done
  ) &
  echo $! > "$OUTDIR/$TAG.mem.pid"
  echo "[v41] memtrace -> $MEMLOG"
fi

# 【不要 cd /tmp】任何留在 /tmp 的 `*.py` 都会遮蔽同名模块 —— 2026-09-15 一个
# 遗留的 /tmp/attr.py 就命中了 aiohttp 的 `import attr`,让整个 cellA 起不来
# (`FileNotFoundError: '/proc/--model/status'`)。改用专用的空目录。
RUN_CWD="$OUTDIR/run"; mkdir -p "$RUN_CWD"; cd "$RUN_CWD"
export CUDA_VISIBLE_DEVICES="$GPUS"
# 【性能提示】下面四项是**为"先跑通"钉的保守值**,并不是引擎默认:
#   ASYNC        引擎默认开启,注释自带实测收益:每 token 6.53→1.80 ms(3.6×)、
#                C=4 聚合 71.8→107.3 t/s,且已验证"与 host-func 逐位相同"、
#                "greedy 文本 5/5 相同" ⇒ 关掉它等于白丢一个已验证的大收益;
#   SPIN_IDLE_US 插件默认 300(hybrid_model.py setdefault),这里钉成 0;
#   NSLICE_SMALL 引擎默认开启;=0 会**强制走 legacy 路径**;
#   THREADS      未设时插件按 n_ccd×5 自动调优(本机 24 CCD ⇒ 120),这里钉成 **192**
#                (实测数据见下面 XIAOTU_MOE_THREADS 那一段;旧值 60 是错的)。
# 一律写成 "${VAR:-<现值>}":**默认行为逐字不变**,但允许从外部覆盖做 A/B。
# 调参时配合 report/tuning/NOTES.md §389。
# ---- 环境变量文件桥 ----------------------------------------------------------
# 实测:vLLM spawn EngineCore 时会**静默丢弃**部分 XIAOTU_* (THREADS/SPIN_IDLE_US/
# GPU_RESIDENT_LAYERS/RELEASE_SOURCE 都丢过),于是"设了开关却没生效、且无日志"。
# 插件在 __init__.py 里会从该文件补齐缺失项(真实环境优先)。见 NOTES §414。
# XIAOTU_MOE_THREADS(默认 **184** = 物理核 192 留 8 个余量(实测拐点);原来是 60,中途曾是 192):
#   【为什么留余量】SMT 关掉后,192 = **占满全部物理核**。单 token 解码的 B=1 延迟在
#   192 下是**双峰**的(静默机器上连测:0.58 / 3.81 / 11.17 ms),而 176/184 稳定 ~0.5 ms
#   —— 只要有一个 worker 被抢占(OS/vLLM/CUDA 线程),barrier 就要等它,而没有任何余核可去。
#   prefill 侧 176 只比 192 慢 ~4-5%(B=4096: 355 vs 341 ms),用 4% 换解码尾延迟的稳定很划算。
#   (SMT 开着时 192 曾是安全的,因为还有 192 个逻辑核当余量;现机器 SMT=off。)

#   **实测线程数是 prefill 的头号旋钮**(NOTES §424,真权重单层微基准,B=1400):
#     60  -> 386.8 ms/层(B=2048)   ← 旧默认,两头都亏
#     120 -> 219.2                  ← 引擎自身默认(为"解码带宽"调的,解码最优)
#     192 -> 170.9   ⇒ 比 60 快 **2.26×**,比 120 快 1.28×
#     384 -> 176.0   但**解码**从 0.45 掉到 8.88 ms(**17×**,SMT 在作业少时纯抢 L3/功耗)
#   解码在 192 下**没有退化**(真权重服务实测:串行 27.53/27.78 tok/s,
#   与历史最佳 27.5-28.5 一致;C=8 聚合 85.31→**92.34** tok/s,历史最好)。
#   ⇒ 60 既不是引擎默认也不是任何一档的最优,是当初"先跑通"钉下的保守值,故改 192。
#   引擎自身的默认(120)**没有动** —— 那是别的模型的解码最优点(NOTES §424)。
XTU_ENV_FILE="${XIAOTU_ENV_FILE:-/tmp/xiaotu_env}"
# ---- 【§504 修:内存+性能,都在我们自己的环境里实测出来的】-------------------
# TP=2 时 **每个 rank 的线程数**必须按"本 rank 的 CCD 数 × 4-5"取,而不是按整机核数:
#   184(/rank)× 2 rank = 368 线程挤 192 物理核(用户规则:至少给每个 worker 留 2 核,
#   且绝不用满)⇒ 实测 **TPOT 436.7 ms / C=1 1.93 tok/s**;
#   60/rank(= 12 CCD × 5)⇒ **TPOT 68.1 ms / C=1 9.62 tok/s**(解码快 6.4×)。
# 184 是**单进程**基准(world=1)的拐点,不能直接搬到 TP=2 服务上 —— 这是本轮踩到的坑。
THREADS_DEFAULT="${THREADS_DEFAULT:-60}"
# 自旋 5000 µs 是 §355 记录过的正反馈灾难(48 层 × 3 相 × 5 ms ⇒ 池几乎永不停转),
# 与上面的超订叠在一起会互相放大 ⇒ 默认 0(完全不自旋,worker 直接 futex 睡)。
SPIN_DEFAULT="${SPIN_DEFAULT:-0}"
XTU_ENV_FILE="${XIAOTU_ENV_FILE:-/tmp/xiaotu_env}"
{
  echo "XIAOTU_MOE_THREADS=${XIAOTU_MOE_THREADS:-$THREADS_DEFAULT}"
  echo "XIAOTU_MOE_NSLICE_SMALL=${XIAOTU_MOE_NSLICE_SMALL:-0}"
  echo "XIAOTU_MOE_ASYNC=${XIAOTU_MOE_ASYNC:-0}"
  echo "XIAOTU_MOE_SPIN_IDLE_US=${XIAOTU_MOE_SPIN_IDLE_US:-$SPIN_DEFAULT}"
  echo "XIAOTU_MOE_GPU_RESIDENT_LAYERS=${XIAOTU_MOE_GPU_RESIDENT_LAYERS:-}"
  echo "XIAOTU_MOE_RESIDENT_BUDGET_GB=${XIAOTU_MOE_RESIDENT_BUDGET_GB:-0}"
  echo "XIAOTU_RELEASE_SOURCE=${XIAOTU_RELEASE_SOURCE:-1}"
  # 【§504】**多 rank 同机必须用 CCD 交错切分**(RANK_SPLIT=2)。原因(改自 moe_v2.hpp 的设计):
  #   nshard_ = max(1, numa_node_count()/world)。NPS1 本机 numa=2、TP=2 ⇒ **nshard_=1**
  #   ⇒ 分片路径整个失效,退回"每个 socket 一份副本"⇒ **权重存 2 份**;
  #   而 RANK_SPLIT=2 让**每个 rank 覆盖全部 node**(CCD 交错),于是 nshard_=2 生效
  #   ⇒ 每个 node 只存自己那 1/2 行 = **整机 1 份权重**(与 lk_moe 的"每 node 一份分片"同构)。
  #   实测(TP=2/V4.1 真权重,服务进程树总 RSS):峰值 1121 → 989.5 GiB;稳态 1121 → 859 GiB。
  echo "XIAOTU_MOE_RANK_SPLIT=${XIAOTU_MOE_RANK_SPLIT:-2}"
  for kv in $EXTRA_ENV; do case "$kv" in *=*) echo "$kv";; esac; done
} > "$XTU_ENV_FILE"
export XIAOTU_ENV_FILE

nohup env \
  HF_HUB_OFFLINE=1 \
  VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  VLLM_HANDSHAKE_TIMEOUT_MINS=120 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_EXPERTS_LOAD_DEVICE=cpu \
  XIAOTU_RELEASE_SOURCE="${XIAOTU_RELEASE_SOURCE:-1}" \
  XIAOTU_MOE_THREADS="${XIAOTU_MOE_THREADS:-$THREADS_DEFAULT}" \
  XIAOTU_MOE_NSLICE_SMALL="${XIAOTU_MOE_NSLICE_SMALL:-0}" \
  XIAOTU_MOE_ASYNC="${XIAOTU_MOE_ASYNC:-0}" \
  XIAOTU_MOE_SPIN_IDLE_US="${XIAOTU_MOE_SPIN_IDLE_US:-$SPIN_DEFAULT}" \
  XIAOTU_MOE_RANK_SPLIT="${XIAOTU_MOE_RANK_SPLIT:-2}" \
  OMP_NUM_THREADS=1 \
  $EXTRA_ENV \
  numactl --interleave=all "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name dsv41 \
    --load-format "$LOAD" \
    --max-model-len "$MAXLEN" --tensor-parallel-size "$TP" --max-num-seqs "${MAXSEQS:-1}" \
    $( [ "${MBT:-0}" -gt 0 ] 2>/dev/null && echo --max-num-batched-tokens "$MBT" ) \
    --gpu-memory-utilization "$GPU_UTIL" $( [ "${EAGER:-0}" = "1" ] && echo --enforce-eager ) \
    $( [ "${SPEC:-0}" = "1" ] && echo --speculative-config "$SPEC_CONFIG" ) \
    $( [ "${PREFIX_CACHE:-1}" = "1" ] || echo --no-enable-prefix-caching ) --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --kernel-config '{"enable_jit_warmup": false}' \
    --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[v41] pid=$(cat "$OUTDIR/$TAG.pid")"

# Wait for readiness. Model construction alone took ~23 min with dummy weights.
DEADLINE=$(( SECONDS + ${READY_TIMEOUT:-3600} ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "[v41] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[v41] server exited early; tail:"; tail -25 "$LOG"; exit 1
  fi
  sleep 10
done
echo "[v41] TIMEOUT waiting for readiness; tail:"; tail -25 "$LOG"; exit 1
