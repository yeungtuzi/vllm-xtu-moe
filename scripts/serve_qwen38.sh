#!/usr/bin/env bash
# Qwen3.8-Flash-Next(官方 FP8)在单卡 SM80 上的调试服务 —— **默认 GPU2 / TP=1**。
#
# 纪律(AGENTS.md「自服务环境纪律」,用户 2026-09-29):
#   * 8070 是本 agent 自身推理后端 ⇒ **本脚本默认不碰 GPU0/1、不用 8070**;
#   * 调试需要 GPU 时用 **GPU2 + TP=1**;要动 GPU0/1 必须显式获得用户许可。
#   本脚本默认 `GPUS=2 TP=1 PORT=8140 MAXLEN=8192` —— 全部是"单卡调试"口径。
#
# 为什么官方 FP8 可行(依据 dev-docs/patches/probe_official_fp8_gate.py 的实测):
#   * 检查点在本地 HF 缓存(173 GiB,131 分片),`quant_method: fp8` ⇒ vLLM 原生 `Fp8Config`;
#     本插件按 `FusedMoEMethodBase` **泛化打 shim** ⇒ `Fp8MoEMethod` 自动走 CPU 专家;
#   * 权重分布(按张量头部精确统计):路由专家 **114.9 GiB**(留 CPU)、
#     **PLE/ngram 表 47.8 GiB**(必须 `XIAOTU_PLE_CPU=1` 走 UVA,否则单卡必 OOM,
#     见 vllm_xiaotu_moe/ple_offload.py 头部注释)、其余非专家仅 ~10 GiB;
#   * ⇒ PLE offload 后 GPU 侧非专家 ~10 GiB,40GB 单卡放得下。
#
# 用法:
#   bash scripts/serve_qwen38.sh                       # dummy? 不,默认真权重
#   LOAD=dummy bash scripts/serve_qwen38.sh            # 只验架构/显存,不读 173GiB(首选第一步)
#   MAXLEN=4096 GPUS=2 TP=1 bash scripts/serve_qwen38.sh
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
# 官方 FP8 检查点(本地 HF 缓存快照;可用 CKPT= 覆盖)
CKPT="${CKPT:-$(ls -d /home/user/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/*/ 2>/dev/null | head -1)}"

TAG="${TAG:-qwen38_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8140}"        # ⚠️ 不能是 8070(生产)
GPUS="${GPUS:-2}"           # 【纪律】调试只用 GPU2
TP="${TP:-1}"               # 【纪律】TP=1
MAXLEN="${MAXLEN:-8192}"    # 【纪律】上下文尽量短
MBT="${MBT:-2048}"
SEQS="${SEQS:-1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
THREADS="${THREADS:-48}"    # ⭐⭐【2026-10-07 用户明令】48 = n_ccd×2;原 n_ccd×5(=120)【已作废】✗
LOAD="${LOAD:-auto}"        # auto=读真权重;dummy=不读盘,只验架构与显存
EAGER="${EAGER:-1}"         # 单卡调试先从全 eager 起,稳了再上图
PLE_CPU="${XIAOTU_PLE_CPU:-1}"   # 必须有:PLE 47.8GiB 走主机内存(UVA)
GPU_PREFILL_MIN="${GPU_PREFILL_MIN:-0}"  # 单卡调试先关 GPU 预填
KV_DTYPE="${KV_DTYPE:-auto}"
TOOL_PARSER="${TOOL_PARSER:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
OUTDIR="${OUTDIR:-$ROOT/dev-docs/report/tuning/logs}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"
# env 桥写【仓库内】,绝不覆盖生产用的 /tmp/xiaotu_env
XTU_ENV_FILE="${XTU_ENV_FILE_OVERRIDE:-$OUTDIR/$TAG.envfile}"

if [ -z "$CKPT" ] || [ ! -f "$CKPT/config.json" ]; then
  echo "[qwen38] 检查点未找到: '$CKPT'(可显式 CKPT=<snapshot 目录>)" >&2; exit 2
fi
command -v numactl >/dev/null 2>&1 && NCTL=(numactl --interleave=all) || NCTL=()

{
  echo "XIAOTU_MOE_THREADS=$THREADS"
  echo "XIAOTU_MOE_SPIN_IDLE_US=${SPIN_IDLE_US:-300}"
  echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GPU_PREFILL_MIN"
  echo "XIAOTU_PLE_CPU=$PLE_CPU"
  echo "XIAOTU_RELEASE_SOURCE=${XIAOTU_RELEASE_SOURCE:-1}"
} > "$XTU_ENV_FILE"
export XIAOTU_ENV_FILE

ARGS=(
  --model "$CKPT"
  --served-model-name Qwen3.8-Flash-Next
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP"
  --dtype bfloat16
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "$MBT"
  --max-num-seqs "$SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --load-format "$LOAD"
  --trust-remote-code
  --enable-chunked-prefill
  --disable-custom-all-reduce
)
[ "$KV_DTYPE" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
[ -n "$TOOL_PARSER" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[ "${PROMPT_TOKENS_DETAILS:-1}" = "1" ] && ARGS+=(--enable-prompt-tokens-details)
# ⭐ 2026-10-04:MXFP4 变体的 quantization_config 用的是 compressed-tensors 的 config_groups,
#   其 group_2 正则写的是【检查点侧】的 `in_proj_qkv` / `in_proj_z`,而新版模型代码把两者
#   融合成 `in_proj_qkvz` ⇒ 没有规则命中 ⇒ EngineCore 抛
#   "No compressed-tensors compatible scheme was found for … in_proj_qkvz" ✗
#   ⇒ 用 --hf-overrides 把融合名补进 targets(不改 vLLM 代码)。见 EXPERIMENTS B307。
[ -n "${HF_OVERRIDES:-}" ] && ARGS+=(--hf-overrides "$HF_OVERRIDES")
# 通用透传(测投机/图模式等一次性实验用;生产口径请写进本脚本而不是靠这个)
# ⭐ 2026-10-04【显式 KV 预算,照 DS-4.1 的做法】:给了 --kv-cache-memory 之后 vLLM 会
#   【跳过显存剖析】⇒ --gpu-memory-utilization 失效(R24 第 8 条)⇒ 剩下的显存留给
#   GPU-prefill 的 staging。QFN 实测 ≈72,368 B/token ⇒ 预算 = 目标 token 数 × 该值。
[ -n "${KV_CACHE_BYTES:-}" ] && ARGS+=(--kv-cache-memory "$KV_CACHE_BYTES")
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

echo "[qwen38] tag=$TAG port=$PORT gpus=$GPUS tp=$TP maxlen=$MAXLEN mbt=$MBT load=$LOAD"
echo "[qwen38] PLE_CPU=$PLE_CPU(必须=1,否则单卡 OOM)  eager=$EAGER  gpu_prefill_min=$GPU_PREFILL_MIN"
echo "[qwen38] ckpt=$CKPT"
echo "[qwen38] log=$LOG   envfile=$XTU_ENV_FILE"

if [ "${DRYRUN:-0}" = "1" ]; then
  printf '[qwen38] DRYRUN gpus=%s tp=%s: ' "$GPUS" "$TP"
  printf '%q ' env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="${XTU_TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}" \
    VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_ENV_FILE="$XTU_ENV_FILE" "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
  printf '\n'; exit 0
fi

cd "$OUTDIR"; mkdir -p run; cd run
export CUDA_VISIBLE_DEVICES="$GPUS"

# ⭐ 2026-10-04【同机多实例纪律,AGENTS.md §4】起第二个实例前的四项自查:
#   ① 结构相似的替代模型(QFN ✓,不是巨模型)  ② MemAvailable 够
#   ③ oom_score_adj=+800(让内核【先杀自己】,别去挑生产)  ④ THREADS=n_ccd×5(=120)+ taskset(见下方 2026-10-06 段 ✓)
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-800}"
# ⭐ 2026-10-06【qfn 服务默认值,用户授权】线程数默认改为 **n_ccd × 5**(本机 24 CCD ⇒ 120)。
#   依据(实测):
#     · 引擎自身默认就是 n_ccd×5 = 120(`serve_v41.sh` 注释:*"未设时插件按 n_ccd×5 自动调优"*);
#       本脚本此前钉成 60 是"先跑通"的保守值,`serve_v41.sh` 明确写着 **60 两头都亏**:
#       60 → 386.8 ms/层,120 → 219.2,192 → 170.9 ⇒ 120 比 60 快 **1.76×**。
#     · 本任务 qfn 端到端实测(全量 200 题):`THREADS=120` 比 60 快 **1.62×**。
#   ⚠️ 代价(用户已知并接受):生产 8070 自身也吃 CPU;用户说明**当前引擎实现的核心利用率
#      本就吃不满**,故按 120 设定。若生产转为满载,应把 THREADS_MAX 调回 64 ✗。
#   ⚠️ 本项与 GPU 预填无关 —— 本次任务不关注 GPU 预填性能,只要求行为正确 ✓。
# ⛔⛔【2026-10-07 用户明令 · 上面 120 那条【已作废】】✗✗
#   用户原话:"直接设置 qfn 的线程数为 **48**，**不准手工绑定核心**" ✓
#   ⇒ 默认改为 **48**(= n_ccd×2),并**取消手工核绑定**(见下方 TASKSET 段)✓
#   ⇒ 上面那段"120 = n_ccd×5"的实测依据**仍然成立**(那是"120 比 60 快 1.62×"的历史事实),
#     但它**不再是本脚本的默认**;要复现旧口径请显式传 `THREADS=120` ✓
THREADS_MAX="${THREADS_MAX:-192}"
THREADS="$(awk -v t="$THREADS" -v m="$THREADS_MAX" 'BEGIN{print (t+0>m)?m:(t+0)}')"
# ⛔ 手工绑核【默认不做】(用户 2026-10-07 明令:"不准手工绑定核心")✓
#   ⚠️ 语义(务必分清,否则会悄悄改变别人的实例 ✗):
#     · 取值 **`0-191`**(默认)= 全部 192 个在线核 ⇒ **等价于不绑**,既有调用方行为**不变** ✓
#     · 显式传【空字符串】`TASKSET=` ⇒ 连 `taskset` 包装都不加 ✓
#       ⭐ 核分配交给引擎自身(它按 CCD/NUMA 拓扑自己 pin `cores_`,见 numa_pool.hpp)✓
#   ⚠️ 故这里用 `${TASKSET-0-191}`【不带冒号】—— 带冒号会把"显式空"当未设 ⇒ 退回 0-191 ✗
TASKSET="${TASKSET-0-191}"
TS_PREFIX=(); [ -n "$TASKSET" ] && TS_PREFIX=(taskset -c "$TASKSET")
MEM_GATE_GIB="${MEM_GATE_GIB:-250}"  # QFN 实测宿主 ~163 GiB(FP8)/~107 GiB(MXFP4)
_avail="$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
echo "[qwen38] 资源纪律: THREADS=$THREADS taskset=${TASKSET:-<不加taskset>} nice=19 oom_score_adj=$OOM_SCORE_ADJ MemAvailable=${_avail}GiB"
if [ "${ALLOW_LOW_MEM:-0}" != "1" ] && [ "$_avail" -lt "$MEM_GATE_GIB" ]; then
  echo "[qwen38] ✗ 拒绝启动:MemAvailable=${_avail}GiB < MEM_GATE_GIB=${MEM_GATE_GIB}GiB" >&2
  echo "[qwen38]   同机很可能已有生产在跑 ⇒ 确需放行请显式 ALLOW_LOW_MEM=1(自担风险)" >&2
  exit 3
fi

# ⭐ 日志按【真 PID】命名(与 serve_v41.sh 同规):旧的 `> "$LOG"` 固定名会互相截断 ✗
(
  LOGFILE="$OUTDIR/$TAG.$BASHPID.log"
  ln -sfn "$(basename "$LOGFILE")" "$LOG" 2>/dev/null || true
  echo "$BASHPID" > "$OUTDIR/$TAG.pid"
  exec > "$LOGFILE" 2>&1
  echo "[qwen38] service log=$LOGFILE pid=$BASHPID"
  exec env \
    PYTHONPATH="${XTU_TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}" \
    HF_HUB_OFFLINE=1 \
    VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}" \
    VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-120}" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_EXPERTS_LOAD_DEVICE=cpu \
    XIAOTU_ENV_FILE="$XTU_ENV_FILE" \
    XIAOTU_PLE_CPU="$PLE_CPU" \
    VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$GPU_PREFILL_MIN" \
    XIAOTU_MOE_THREADS="$THREADS" \
    OMP_NUM_THREADS=1 \
    "${NCTL[@]}" "${TS_PREFIX[@]}" nice -n 19 \
    "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
) &
SVC_PID="$(cat "$OUTDIR/$TAG.pid" 2>/dev/null)"
if [ -n "$SVC_PID" ] && echo "$OOM_SCORE_ADJ" > "/proc/$SVC_PID/oom_score_adj" 2>/dev/null; then
  echo "[qwen38] oom_score_adj=$OOM_SCORE_ADJ 已注入 pid=$SVC_PID ✓(OOM 时内核先杀本实例)"
else
  echo "[qwen38] ⚠️ oom_score_adj 注入失败(pid=$SVC_PID)⇒ 内存告急时【生产有被误杀的风险】"
fi
echo "[qwen38] pid=$SVC_PID"
echo "[qwen38] 日志(按 PID 命名)= $OUTDIR/$TAG.$SVC_PID.log;稳定符号链接 $LOG → $TAG.$SVC_PID.log"

DEADLINE=$(( SECONDS + ${READY_TIMEOUT:-3600} ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "[qwen38] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[qwen38] server exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  sleep 10
done
echo "[qwen38] TIMEOUT; tail:"; tail -40 "$LOG"; exit 1
