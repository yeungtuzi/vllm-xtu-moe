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
#   bash scripts/serve_lk_port.sh                 # 无投机(干净基线)
#   SPEC=1 bash scripts/serve_lk_port.sh          # 带 dspark5(lk 生产同款)
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
SPEC_ON="${SPEC:-0}"
EAGER="${EAGER:-0}"   # 1 = 加 --enforce-eager(避开 lk 的 _cpu_prefill 在捕获期同步的问题)
PREFETCH="${PREFETCH:-3}"        # GPU 预取窗口,lk 上游默认 3(vllm/envs.py:1860)
RESIDENT="${RESIDENT:-}"         # 额外常驻 GPU 的 MoE 层,如 "0-9";这里只填开关,不改代码
MODEL_EST_GIB="${MODEL_EST_GIB:-12}"  # 目标模型 GPU 占用估值(实测 TP=1 为 10.44 GiB)
KV_MIN_GIB="${KV_MIN_GIB:-8}"         # 必须留给 KV cache + 激活的最低显存

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
# draft 永远在 GPU(用户硬约束),且这里**不加任何 vLLM 代码**:
#   vLLM 对 DS-V4 的专门处理是把 draft 当**独立模型**加载(`dspark.py:132/316`:
#   prefix="" -> "model" -> 层名 `model.layers.{num_hidden_layers+i}.ffn.experts`,
#   `n_mtp_layers` 未设 -> 3 层),权重来自目标 ckpt 的 `mtp.{0,1,2}.*`
#   (`model.py:1250` 的 mapper + `dspark.py:502` 的正则)。而 lk 判 GPU 驻留只用
#   **层名**规则(`envs.py:2262` `layer_name.startswith("mtp.")`),这条规则对
#   独立模型的 `model.layers.43...` **不会命中**,所以必须用 lk 自己的开关
#   `LVLLM_GPU_RESIDENT_MOE_LAYERS`(envs.py:2272)把草稿层号显式标成常驻 ——
#   常驻层在 `quantization/mxfp4.py:548` 得到 **CUDA** 权重,走 vLLM 原生 GPU MoE
#   (`forward_monolithic`),既不进 CPU 也不进 _gpu_prefill。
# ---------------------------------------------------------------------------
DRAFT_IDS="" DRAFT_GIB=0 NLAYERS=0 NMTP=0
if [ -f "$CKPT/config.json" ]; then
  eval "$(python3 - "$CKPT" "$TP" <<'PY'
import json, os, struct, sys
ckpt, tp = sys.argv[1], int(sys.argv[2])
cfg = json.load(open(os.path.join(ckpt, "config.json")))
n = int(cfg["num_hidden_layers"])
m = int(cfg.get("n_mtp_layers") or 3)          # 与 dspark.py:108 同规则
ids = f"{n}-{n + m - 1}" if m > 0 else ""
idx = os.path.join(ckpt, "model.safetensors.index.json")
expert = other = 0
if os.path.exists(idx):
    wm = json.load(open(idx))["weight_map"]
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
print(f'DRAFT_IDS="{ids}" NLAYERS={n} NMTP={m} '
      f'DRAFT_GIB={((expert / max(tp, 1) + other) / 2**30):.2f}')
PY
)"
fi

if [ "$SPEC_ON" = "1" ]; then
  TOTAL_MIB=$(for _g in $(echo "$GPUS" | tr ',' ' '); do
      nvidia-smi --id="$_g" --query-gpu=memory.total --format=csv,noheader,nounits
    done | sort -n | head -1)
  BUDGET_GIB=$(python3 -c "print(f'{$TOTAL_MIB/1024*$GPU_UTIL:.2f}')")
  NEED_GIB=$(python3 -c "print(f'{$MODEL_EST_GIB+$DRAFT_GIB+$KV_MIN_GIB:.2f}')")
  echo "[lk_port] draft=$NMTP 层(层号 $DRAFT_IDS) 需常驻 GPU ≈${DRAFT_GIB} GiB/rank; 预算 ${BUDGET_GIB} GiB, 需要 ≥${NEED_GIB} GiB"
  if python3 -c "import sys; sys.exit(0 if $BUDGET_GIB >= $NEED_GIB else 1)"; then
    RESIDENT="${RESIDENT:+$RESIDENT,}$DRAFT_IDS"
    ARGS+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}')
  else
    echo "[lk_port] WARNING: 显存放不下 draft(需 ≈${DRAFT_GIB} GiB/rank,预算仅 ${BUDGET_GIB} GiB," \
         "需 ≥${NEED_GIB} GiB)/ 无法保证 draft 常驻 GPU -> 按策略**禁用 draft model**," \
         "忽略 SPEC=$SPEC_ON 参数(如需强制请调大 GPU_UTIL 或 TP,或提高 DRAFT_GIB_MIN 相关阈值)"
    SPEC_ON=0
  fi
fi
RESIDENT="${RESIDENT# }"

{
  echo "tag=$TAG port=$PORT tp=$TP gpus=$GPUS maxlen=$MAXLEN seqs=$SEQS mbt=$MBT"
  echo "gpu_util=$GPU_UTIL threads=$THREADS minbatch=$MINBATCH spec=$SPEC_ON eager=$EAGER"
  echo "prefetch=$PREFETCH resident='$RESIDENT' draft_ids=$DRAFT_IDS draft_gib=$DRAFT_GIB nlayers=$NLAYERS nmtp=$NMTP"
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
