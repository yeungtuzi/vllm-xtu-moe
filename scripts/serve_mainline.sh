#!/usr/bin/env bash
# 主线 vLLM + `vllm-xtu-moe` 插件(**零核心补丁**路径)。
#
# 与 `serve_lk_port.sh` 的区别:
#   * `serve_lk_port.sh` 走的是 **lk 编排 fork**(Lvllmds4-x + 我们的移植提交),
#     性能数字最好,但要跟着 fork 走;
#   * 本脚本走 **mainline vLLM + 插件**(`vllm.general_plugins` 入口),
#     不需要改任何主线源码 —— 见 `docs/UPSTREAM_DRIFT.md` 的"形态 0"。
#
# 用法:
#   ENV=/path/to/mainline-venv TAG=mymain PORT=8071 bash scripts/serve_mainline.sh
#   XIAOTU_OOT_OVERRIDE=0 ...   # 换成"主线 RoutedExperts + CPU backend"路径
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"

TAG="${TAG:-mainline_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8071}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAXLEN="${MAXLEN:-8192}"
SEQS="${SEQS:-8}"
# ---- GPU 预填充三开关(见 NOTES §340;缺一个就失效) ----
# 便捷预设(必须放在 MBT 解析之前):PREFILL=1 ⇒ 对齐参考 fork 的生产预填充协议
#   (serve_lk_port.sh: MBT=8192 / MINBATCH=1024 / 只捕获解码尺寸)。
#   实测端到端:2048 token TTFT 5.71 s / 366 t/s(CPU 侧要 92.6 s ⇒ 16.2×);
#   TP=2 16000 token ⇒ 1149 t/s(report/curve_thr.jsonl, report/curve_tp2.jsonl)。
if [ "${PREFILL:-0}" = "1" ]; then
  MBT="${MBT:-8192}"
  GP_MIN="${GP_MIN:-1024}"
  CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-1 2 4 8}"
fi
MBT="${MBT:-256}"                    # 与 fork 协议对齐(实测:不设时主线默认很大)
# GPU 预填充阈值(fork 里叫 LVLLM_GPU_PREFILL_MIN_BATCH_SIZE,生产值 1024)。
#   0 = 关(默认,保持解码基准协议不变);N>0 时 qlen>=N 的 MoE 层走 GPU
#   (hybrid_model.py:982 的 _gp_min 分支)。**必须同时满足 MBT >= N**,
#   否则 batch 永远到不了阈值 —— 与 fork 把 MINBATCH 夹到 MBT 的语义一致。
GP_MIN="${GP_MIN:-0}"
# 只给这些 batch size 捕获 CUDA 图(**空格分隔**,对应 --cudagraph-capture-sizes 的 nargs='+')。**这是让 GPU 预填充生效的关键**:
#   预填充形状若不在捕获集里就不走图、直接 eager ⇒ hybrid_model 的 GPU 分支才会被选中;
#   同时解码(size<=SEQS)仍然享有 CUDA 图。留空 = 用主线默认(PIECEWISE 会把预填充
#   形状也捕获掉,重放永远走捕获时的 CPU 分支 ⇒ GPU 预填充形同虚设)。见 NOTES §340(c)。
CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-}"
# NUMA 交错(**默认开,不要关**)。
# 为什么必须:vLLM 加载权重时每个 rank 装**全部 256 个专家**(≈138 GB/worker,
# EP 不在加载期切分存储),这部分是**未绑定**的 first-touch 分配,会顺着加载线程
# 所在的 node 堆下去。实测(2026-09-14,已加载 38/43 层时):
#     node 0 free:  7220 MB   ← 185 GB 已用      node 4 free: 168037 MB
#     node 2 free:  7951 MB   ← 185 GB 已用      node 7 free: 169973 MB
# **总空闲还有 863 GB,但 node 0/2 已 96% 满** ⇒ 下一次落上去的分配就:
#     oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=0,
#               task=VLLM::Worker_TP, anon-rss:236289496kB
# 进程**静默死亡**(无 Traceback,日志"只加载到一半就没了")。
# 引擎自身的权重分片已由 mbind 正确铺开(见 moe_v2.hpp shard_region),但管不到
# vLLM 那 138 GB,所以必须在进程级交错。INTERLEAVE=0 仅在你自己已绑核/绑节点时用。
INTERLEAVE="${INTERLEAVE:-1}"
if [ "$GP_MIN" -gt 0 ] && [ "$GP_MIN" -gt "$MBT" ]; then
  echo "[mainline] GP_MIN=$GP_MIN > MBT=$MBT ⇒ 夹到 MBT(否则预填充永远够不到阈值;NOTES §319c)"
  GP_MIN="$MBT"
fi
# chunked prefill:**建议保持主线默认(开)**。
#   * 关掉它时,主线要求 MBT >= max_model_len(否则直接报错),而且预填充会整段进
#     我们的 CPU 引擎(qlen=MBT)⇒ 一次 256 token 的预填充就要 26 ms/层 × 43 ≈ 1.1 s;
#   * 开着时,解码步里可能混进预填充块(qlen=257)⇒ 解码也会付这份钱。
# ⇒ 两种都不理想,根因是**形态 B 没有"预填充专用路径"**(fork 里有 `_cpu_prefill` /
#   `_gpu_prefill` 的分流)。见 NOTES §334s。默认先取主线默认(开)。
# **补充(2026-09-14,NOTES §340)**:形态 A(OOT=1,本脚本默认)的 `hybrid_model.py`
#   里 **GPU 预填充路径早就写好了**(gpu_prefill.py:KV-Major 锁页缓存 + PrefetchSlot
#   重叠 + Triton MXFP4 分组 GEMM),只是本脚本一直**没导出阈值 + MBT=256 够不到 +
#   预填充形状被 CUDA 图捕获**,三个开关全缺 ⇒ 形同虚设。用 `PREFILL=1` 一次打开。
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
# 加载策略:实测插件路径读分片 2.5-4.8 s/片(fork 路径 0.6 s/片);
# 主线日志明确建议 EXT4 上用 prefetch 强制预取。
LOAD_STRATEGY="${LOAD_STRATEGY:-prefetch}"
# A100(SM80)没有 fp8e4nv:Triton 的 PackSeq 等内核在 JIT warmup 时会报
# "type fp8e4nv not supported in this architecture" ⇒ 直接关掉预热(实测 ml08modeB)
KERNEL_WARMUP="${KERNEL_WARMUP:-0}"
# EAGER=1 ⇒ --enforce-eager(关 CUDA 图)。诊断用:区分"图捕获契约"与"调度"问题。
EAGER="${EAGER:-0}"
# 引擎线程:插件侧旋钮(每 rank 12 CCD ⇒ 5 核/CCD = 60)
THREADS="${THREADS:-60}"
# 额外常驻 GPU 的 MoE 层(与 lk 的 LVLLM_GPU_RESIDENT_MOE_LAYERS 同义)
RESIDENT="${RESIDENT-0-11}"
# 1 = OOT 模型覆盖(默认,零补丁);0 = 主线 RoutedExperts + 我们的 CPU 后端
OOT="${XIAOTU_OOT_OVERRIDE:-1}"
EXTRA_ENV="${EXTRA_ENV:-}"

OUTDIR="$ROOT/report/tuning/logs"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"
rm -f /dev/shm/xiaotu_ep_*.bin 2>/dev/null || true

ARGS=(
  --model "$CKPT"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --trust-remote-code
  --served-model-name DeepSeek-V4-Flash-xiaotu
)
if [ -n "$MBT" ]; then ARGS+=(--max-num-batched-tokens "$MBT"); fi
if [ "$CHUNKED_PREFILL" = "0" ]; then ARGS+=(--no-enable-chunked-prefill); fi
if [ "$CHUNKED_PREFILL" = "1" ]; then ARGS+=(--enable-chunked-prefill); fi
if [ "$EAGER" = "1" ]; then ARGS+=(--enforce-eager); fi
# 注意:`--cudagraph-capture-sizes` 是 nargs='+'(**空格分隔**,不是逗号)
if [ -n "$CUDAGRAPH_SIZES" ]; then ARGS+=(--cudagraph-capture-sizes $CUDAGRAPH_SIZES); fi
if [ -n "$LOAD_STRATEGY" ]; then ARGS+=(--safetensors-load-strategy "$LOAD_STRATEGY"); fi
if [ "$KERNEL_WARMUP" = "0" ]; then
  ARGS+=(--kernel-config '{"enable_jit_warmup": false}')
fi

{
  echo "tag=$TAG port=$PORT tp=$TP gpus=$GPUS maxlen=$MAXLEN seqs=$SEQS gpu_util=$GPU_UTIL"
  echo "mbt='$MBT' chunked='$CHUNKED_PREFILL' threads=$THREADS resident='$RESIDENT' oot=$OOT load_strategy='$LOAD_STRATEGY' extra_env='$EXTRA_ENV'"
echo "gp_min=$GP_MIN cudagraph_sizes='$CUDAGRAPH_SIZES' prefill_preset='${PREFILL:-0}' interleave='$INTERLEAVE'"
  echo "env=$ENV"; echo "ckpt=$CKPT"; date -Is
} > "$OUTDIR/$TAG.env"

# 等显存真正释放
for _g in $(echo "$GPUS" | tr ',' ' '); do
  for _i in $(seq 1 60); do
    _used=$(nvidia-smi --id="$_g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$_used" ] && [ "$_used" -lt 1024 ] && break
    sleep 5
  done
done

cd /tmp
export PATH="$ENV/bin:$PATH"
NCTL=()
if [ "$INTERLEAVE" = "1" ] && command -v numactl >/dev/null 2>&1; then
  NCTL=(numactl --interleave=all)
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
nohup env \
  HF_HUB_OFFLINE=1 \
  VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
  VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-60}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}" \
  VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="${VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS:-30}" \
  VLLM_EXPERTS_LOAD_DEVICE=cpu \
  XIAOTU_OOT_OVERRIDE="$OOT" \
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$GP_MIN" \
  XIAOTU_MOE_THREADS="$THREADS" \
  XIAOTU_MOE_GPU_RESIDENT_LAYERS="$RESIDENT" \
  OMP_NUM_THREADS=1 \
  $EXTRA_ENV \
  "${NCTL[@]}" "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[mainline] tag=$TAG pid=$(cat "$OUTDIR/$TAG.pid") log=$LOG"

for _ in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[mainline] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[mainline] server exited early; tail:"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[mainline] timeout; tail:"; tail -30 "$LOG"; exit 1
