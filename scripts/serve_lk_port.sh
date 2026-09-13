#!/usr/bin/env bash
# serve_lk_port.sh — 用 lk 的全套编排链 + 我们的计算引擎(xiaotu_moe)启服务。
#
# 这不是新功能,而是把上游现成的东西拼起来:
#   * vLLM 侧 = `Lvllmds4-x`(guqiong96/Lvllmds4-x,SM80 分支)的 lk 编排,
#     只有两个文件与上游不同(`fused_moe/routed_experts.py` + `runner/moe_runner.py`),
#     且只做了 `lk_moe` -> `xiaotu_moe` 的机械改名(见该仓库 commit "port: rebind ...");
#   * 引擎   = 本仓的 `xiaotu_moe`(Apache-2.0),以 `pip install` 装进该 env;
#   * 启动参数 = lk 生产脚本 `process_data/scripts/dsv4.sh` 逐条照搬,
#     只把 max_model_len / max_num_seqs / gpu_util 改成我们本机测试用的值。
#
# 环境:conda env `lkxtu`(= `lvllmds4-x` 的干净克隆 + 上述两个文件 + 我们的引擎)。
# 注意:`lkxtu/bin/vllm` 是 cp -a 留下的**指向旧 env 的符号链接**,所以必须用
# `python -m vllm.entrypoints.openai.api_server` 启动,不能调 `vllm`。
#
# 用法:
#   bash scripts/serve_lk_port.sh                 # **默认:不投机**(两卡实测最快)
#   SPEC=auto bash scripts/serve_lk_port.sh       # 自动识别 ckpt 形态(本 ckpt=dspark)并开投机
#   SPEC=1 bash scripts/serve_lk_port.sh          # 强制开(投机参数取 SPEC_JSON,默认=作者 3/greedy)
#   SPEC=auto SPEC_JSON='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' ...
#   TAG=xxx PORT=8070 THREADS=48 ... bash scripts/serve_lk_port.sh
#
# License: Apache-2.0
set -euo pipefail

ENV="${ENV:-/home/user/anaconda3/envs/lkxtu}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"

TAG="${TAG:-lkport_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8070}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
MAXLEN="${MAXLEN:-262144}"
SEQS="${SEQS:-8}"
MBT="${MBT:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
THREADS="${THREADS:-48}"          # lk 生产是每卡 48(LK_THREADS)
MINBATCH="${MINBATCH:-1024}"      # lk 生产同值:开 GPU prefill
SPEC="${SPEC:-0}"      # **默认 0 = 不投机**(实测最快:TP=2 图 C=1 20.06 vs 投机 11.60 t/s,见 NOTES §298)
                       # auto = 从 ckpt config/张量自动识别 dspark 并把草稿钉在 GPU;1 = 强制开
EAGER="${EAGER:-0}"   # 1 = 加 --enforce-eager(避开 lk 的 _cpu_prefill 在捕获期同步的问题)
PREFETCH="${PREFETCH:-1}"        # GPU 预取窗口;**作者推荐值 1**(README:一般预取 1~2 层)
RESIDENT="${RESIDENT:-}"         # 额外常驻 GPU 的 MoE 层,如 "0-9";这里只填开关,不改代码
DRAFT_RESIDENT="${DRAFT_RESIDENT:-1}"  # 1=草稿层强制常驻 GPU(用户硬约束);0=完全复刻作者配方(不设常驻)
FORCE_DRAFT="${FORCE_DRAFT:-0}"        # 1=跳过显存护栏(明知偏紧仍要开草稿),会打印警告
EXTRA_ENV="${EXTRA_ENV:-}"             # 额外环境变量透传给服务进程,如 EXTRA_ENV="XIAOTU_CD_TIMING=1"
# 投机解码参数:**默认取作者推荐命令里的值**(num 5 / probabilistic);可用 SPEC_JSON 覆盖
DEFAULT_SPEC_JSON='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
SPEC_JSON="${SPEC_JSON:-$DEFAULT_SPEC_JSON}"   # 注意:不要把 JSON 直接写进 ${VAR:-...},bash 会多吐一个 }
MODEL_EST_GIB="${MODEL_EST_GIB:-auto}"  # 目标模型 GPU 占用估值(auto: 按实测 TP=1→11 / TP=2→7)
LK_BUF_GIB="${LK_BUF_GIB:-6}"           # lk 引擎在 GPU 的缓冲 + decode/gpu_prefill 暂存(实测≈5.3)
WARMUP_GIB="${WARMUP_GIB:-3}"           # 预热/首次分配的余量(实测 0.90 时因一笔 2 GiB 分配 OOM)
KV_MIN_GIB="${KV_MIN_GIB:-6}"           # 必须留给 KV cache + 激活的最低显存

OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/report/tuning/logs"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"
rm -f /dev/shm/xiaotu_ep_*.bin 2>/dev/null || true

ARGS=(
  --model "$CKPT"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAXLEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --trust-remote-code
  --served-model-name DeepSeek-V4-Flash-xiaotu
  --compilation_config.cudagraph_mode FULL_DECODE_ONLY
  --enable-prefix-caching
  --enable-chunked-prefill
  --max-num-batched-tokens "$MBT"
  --dtype bfloat16
  --max-num-seqs "$SEQS"
  --enable-auto-tool-choice
  --kv-cache-dtype fp8_ds_mla
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --reasoning-parser deepseek_v4
  --default-chat-template-kwargs '{"enable_thinking": true}'
  --disable-custom-all-reduce
)
if [ "$EAGER" = "1" ]; then
  ARGS+=(--enforce-eager)
fi

# ---------------------------------------------------------------------------
# 【自动匹配】从 ckpt **自带的** config/张量识别投机解码形态,并把 draft 钉在 GPU。
# 判据全部来自官方权重(层数规则与上游 `dspark.py:108` 完全一致),不写死层号:
#   * `dspark_target_layer_ids` 存在 **且** 有 `mtp.*` 张量 ⇒ 该 ckpt 是 DSpark 形态;
#   * 草稿层号 = [num_hidden_layers, num_hidden_layers + (n_mtp_layers or 3))。
# 为什么必须显式指定常驻:draft 是**独立模型**(`dspark.py:132/316`:prefix="" -> "model"
#   -> 层名 `model.layers.{43+i}.ffn.experts`),lk 的 `is_lk_moe_mtp_layer()`
#   (`envs.py:2262`,只认 `mtp.` 前缀)对 DS-V4 草稿**永不命中**(已核对最新官方
#   patch `Lvllm/patches/01_lk_moe__3116c5d.patch`,逻辑相同)。用 lk 自己的开关
#   `LVLLM_GPU_RESIDENT_MOE_LAYERS`(`envs.py:2272`)标常驻 ⇒ 该层在
#   `quantization/mxfp4.py:548` 拿 **CUDA** 权重,走 vLLM 原生 GPU MoE
#   (`forward_monolithic`),既不进 CPU 也不进 `_gpu_prefill`。
# ---------------------------------------------------------------------------
DRAFT_IDS="" DRAFT_GIB=0 NLAYERS=0 NMTP=0 IS_DSPARK=0
if [ -f "$CKPT/config.json" ]; then
  eval "$(python3 - "$CKPT" "$TP" <<'PY'
import json, os, struct, sys
ckpt, tp = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(os.path.join(ckpt, "config.json")))
n = int(cfg["num_hidden_layers"])
m = int(cfg.get("n_mtp_layers") or 3)          # 与上游 dspark.py:108 同规则
idx = os.path.join(ckpt, "model.safetensors.index.json")
wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {}
has_mtp = any(k.startswith("mtp.") for k in wm)
is_dspark = int(bool(cfg.get("dspark_target_layer_ids")) and has_mtp)
ids = f"{n}-{n + m - 1}" if m > 0 else ""
expert = other = 0
for i in range(m):
    pre = f"mtp.{i}."
    for f in sorted({v for k, v in wm.items() if k.startswith(pre)}):
        with open(os.path.join(ckpt, f), "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        for name, meta in hdr.items():
            if name == "__metadata__" or not name.startswith(pre):
                continue
            nb = meta["data_offsets"][1] - meta["data_offsets"][0]
            if ".experts." in name:      # 专家权重按 TP 切分
                expert += nb
            else:                        # attn/norm/dense 每 rank 各一份
                other += nb
print(f'DRAFT_IDS="{ids}" NLAYERS={n} NMTP={m} IS_DSPARK={is_dspark} '
      f'DRAFT_GIB={((expert / max(tp, 1) + other) / 2**30):.2f}')
PY
)"
fi

case "$SPEC" in
  1) SPEC_ON=1 ;;
  0) SPEC_ON=0 ;;
  *) SPEC_ON="$IS_DSPARK" ;;
esac
if [ "$SPEC" = "auto" ] || [ "$SPEC" = "AUTO" ]; then
  echo "[lk_port] 自动识别: dspark=$IS_DSPARK(依据 ckpt 的 dspark_target_layer_ids + mtp.* 张量),草稿 $NMTP 层(层号 $DRAFT_IDS,≈${DRAFT_GIB} GiB/rank) ⇒ SPEC_ON=$SPEC_ON"
fi

if [ "$SPEC_ON" = "1" ]; then
  DRAFT_ARGS=(--speculative-config "$SPEC_JSON")
  if [ "$DRAFT_RESIDENT" = "0" ]; then
    # 完全复刻作者配方:不设常驻 ⇒ 草稿是"GPU 预填充层":预填充 _gpu_prefill(GPU),
    # 解码落 _cpu_prefill/_cpu_decode(CPU)。仅用于 A/B 对照(见 NOTES §291)。
    echo "[lk_port] DRAFT_RESIDENT=0 ⇒ 按作者配方不加常驻;草稿解码将在 CPU(仅供对照)"
    ARGS+=("${DRAFT_ARGS[@]}")
  else
    TOTAL_MIB=$(for _g in $(echo "$GPUS" | tr ',' ' '); do
        nvidia-smi --id="$_g" --query-gpu=memory.total --format=csv,noheader,nounits
      done | sort -n | head -1)
    BUDGET_GIB=$(python3 -c "print(f'{$TOTAL_MIB/1024*$GPU_UTIL:.2f}')")
    M_EST="$MODEL_EST_GIB"
    [ "$M_EST" = "auto" ] && M_EST=$( [ "$TP" -ge 2 ] && echo 7 || echo 11 )
    NEED_GIB=$(python3 -c "print(f'{$M_EST+$DRAFT_GIB+$LK_BUF_GIB+$WARMUP_GIB+$KV_MIN_GIB:.2f}')")
    echo "[lk_port] 估算: 模型 ${M_EST} + 草稿 ${DRAFT_GIB} + lk缓冲 ${LK_BUF_GIB} + 预热 ${WARMUP_GIB} + KV最低 ${KV_MIN_GIB} = ${NEED_GIB} GiB"
    echo "[lk_port] draft=$NMTP 层(层号 $DRAFT_IDS) 需常驻 GPU ≈${DRAFT_GIB} GiB/rank; 预算 ${BUDGET_GIB} GiB, 需要 ≥${NEED_GIB} GiB"
    if [ "$FORCE_DRAFT" = "1" ]; then
      echo "[lk_port] WARNING: FORCE_DRAFT=1 ⇒ 跳过显存护栏(预算 ${BUDGET_GIB} < 估算 ${NEED_GIB});" \
           "若预热 OOM 请回调 GPU_UTIL 或改 TP=2(实测 TP=1 草稿常驻至少要 util≈0.85 且 KV 会被压到 ~5 GiB)"
    fi
    if [ "$FORCE_DRAFT" = "1" ] || python3 -c "import sys; sys.exit(0 if $BUDGET_GIB >= $NEED_GIB else 1)"; then
      RESIDENT="${RESIDENT:+$RESIDENT,}$DRAFT_IDS"
      ARGS+=("${DRAFT_ARGS[@]}")
    else
      echo "[lk_port] WARNING: 显存放不下 draft(需 ≈${DRAFT_GIB} GiB/rank,预算仅 ${BUDGET_GIB} GiB," \
           "需 ≥${NEED_GIB} GiB)/ 无法保证 draft 常驻 GPU -> 按策略**禁用 draft model**," \
           "忽略 SPEC=$SPEC 参数(如需强制请调大 GPU_UTIL 或 TP,或降低 KV_MIN_GIB;" \
           "如接受草稿解码在 CPU 可用 DRAFT_RESIDENT=0 复刻作者配方)"
      SPEC_ON=0
    fi
  fi
fi
RESIDENT="${RESIDENT# }"

{
  echo "tag=$TAG port=$PORT tp=$TP gpus=$GPUS maxlen=$MAXLEN seqs=$SEQS mbt=$MBT"
  echo "gpu_util=$GPU_UTIL threads=$THREADS minbatch=$MINBATCH spec=$SPEC_ON eager=$EAGER"
  echo "extra_env='$EXTRA_ENV' spec_json=$SPEC_JSON"
  echo "spec_mode=$SPEC prefetch=$PREFETCH resident='$RESIDENT' draft_ids=$DRAFT_IDS draft_gib=$DRAFT_GIB nlayers=$NLAYERS nmtp=$NMTP is_dspark=$IS_DSPARK"
  echo "env=$ENV"
  date -Is
} > "$OUTDIR/$TAG.env"

# 等显存真正释放:上一轮被 kill 的 worker 还会占着显存几秒~几十秒,不等就会撞上
# "Free memory ... less than desired GPU memory utilization"(M9,已踩过两次)。
for _g in $(echo "$GPUS" | tr ',' ' '); do
  for _i in $(seq 1 60); do
    _used=$(nvidia-smi --id="$_g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$_used" ] && [ "$_used" -lt 1024 ] && break
    sleep 5
  done
done

cd /tmp   # neutral CWD: see comment above
export PATH="$ENV/bin:$PATH"   # ninja/flashinfer JIT need to be visible to workers
export CUDA_VISIBLE_DEVICES="$GPUS"
nohup env \
  LVLLM_MOE_NUMA_ENABLED=1 \
  LK_THREADS="$THREADS" \
  OMP_NUM_THREADS=1 \
  LK_THREAD_BINDING=CPU_CORE \
  LVLLM_GPU_PREFETCH_WINDOW="$PREFETCH" \
  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE="$MINBATCH" \
  LVLLM_GPU_RESIDENT_MOE_LAYERS="$RESIDENT" \
  LK_POWER_SAVING=1 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  HF_HUB_OFFLINE=1 \
  $EXTRA_ENV \
  "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[lk_port] tag=$TAG pid=$(cat "$OUTDIR/$TAG.pid") log=$LOG"

for _ in $(seq 1 900); do           # 最长等 75 分钟
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[lk_port] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[lk_port] server exited early; tail:"; tail -25 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[lk_port] timeout; tail:"; tail -25 "$LOG"; exit 1
