#!/usr/bin/env bash
# MiMo-V2.6-Flash-RL 启动脚本(把已验证的配方固化;事实来源 docs/EXPERIMENTS.md B92–B104)。
#
# 已验证的事实(别再重新推):
#   * 专家是 MXFP4(e2m0 block-32)⇒ 走 V4.1 那条 MOE_MXFP4 引擎路径;
#   * 不需要 --hf-overrides:vLLM 自己把纯文本 architectures 解析成 MiMoV2OmniForCausalLM;
#   * 语言模型在 SM80 上走 TRITON_ATTN_DIFFKV(hybrid SWA 9 GA + 39 SWA + sink 均可用);
#   * 不加 --language-model-only 时:视觉塔/MM encoder 走 FLASH_ATTN,与语言模型后端不冲突;
#   * 投机只用 MTP k=1(k=3 每步仅多 4%,多深度补丁已回退 —— 见 B98/B100,不要再追);
#   * KV 账:GA 9 层 × 4 KV 头 × 320B × 2 = 23,040 B/token(全机),TP=2 每 rank ≈11.25 KiB/token;
#     SWA 39 层封顶在窗口(每序列常数 ~25.6MB);⚠️ 引擎补齐 padding 层会浪费 7–15% KV,算预算要扣掉。
#
# 用法:
#   GPUS=0,1 TP=2 MAXLEN=1048576 MBT=4096 SEQS=64 LOAD=auto scripts/serve_mimo26.sh
#   GPUS=0 MAXLEN=65536 MM=1 LOAD=dummy scripts/serve_mimo26.sh      # 单卡 dummy 冒烟
# 关键旋钮:
#   GPUS    可见 GPU(决定 TP 上限)                默认 0,1
#   TP      tensor parallel                        默认 2
#   MAXLEN  --max-model-len                        默认 1048576(用户口径)
#   MBT     --max-num-batched-tokens               默认 4096(用户口径)
#   SEQS    --max-num-seqs                         默认 4(并发必须 ≥2;见 B124)
#   UTIL    --gpu-memory-utilization               默认 0.85
#   LOAD    --load-format                          默认 auto(dummy = 骨架冒烟)
#   SPEC_K  MTP num_speculative_tokens             默认 1(0 = 关掉投机)
#   MM      1 = 开多模态(不加 --language-model-only)默认 0(纯文本,省显存且放开后端选择)
#   VERIFY  1 = 打开逐层数值自校验(经 env 文件传给子进程)默认 0
#   PORT    API 端口                               默认 8130
#   PROMPTS 多模态每 prompt 允许的图片数           默认 1
#   MM_LIMITS 完整覆盖 --limit-mm-per-prompt(默认 {"image":$PROMPTS,"video":0,"audio":0};
#             要测音视频就传 '{"image":1,"video":1,"audio":1}')
#   RESIDENT_LAYERS 常驻 GPU 的专家层(逗号+区间,如 "20-21")默认空=不常驻
#   GPU_PREFILL_MIN GPU 预填充门槛(低于它走 CPU)  默认由 env 桥决定
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/MiMo-V2.6-Flash-RL}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
MAXLEN="${MAXLEN:-1048576}"
MBT="${MBT:-4096}"
SEQS="${SEQS:-4}"   # 用户 2026-09-22 定:**所有模型默认 seqs=4**(seqs=1 会让 C≥2 退化成串行,见 EXPERIMENTS B124)
UTIL="${UTIL:-0.85}"
LOAD="${LOAD:-auto}"
SPEC_K="${SPEC_K:-1}"
MM="${MM:-0}"
VERIFY="${VERIFY:-0}"
PORT="${PORT:-8130}"
PROMPTS="${PROMPTS:-1}"
VLLM_TREE="${VLLM_TREE:-/home/user/lvllm/vllm-consolidated}"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
ENVF="${ENVF:-/tmp/mimo26.env}"

[ -d "$CKPT" ] || { echo "检查点不存在: $CKPT" >&2; exit 2; }

# XIAOTU_* 必须经 env 文件传给子进程:EngineCore 会重建环境,直接 export 会丢(B97)。
mkdir -p "$(dirname "$ENVF")"
{ [ "$VERIFY" = "1" ] && echo "XIAOTU_VERIFY_LAYER=1"
  echo "XIAOTU_MOE_THREADS=${XIAOTU_MOE_THREADS:-60}"
  echo "XIAOTU_MOE_SPIN_IDLE_US=${XIAOTU_MOE_SPIN_IDLE_US:-300}"
  [ -n "${RESIDENT_LAYERS:-}" ] && echo "XIAOTU_MOE_GPU_RESIDENT_LAYERS=${RESIDENT_LAYERS}"
  [ -n "${GPU_PREFILL_MIN:-}" ] && echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=${GPU_PREFILL_MIN}"; } > "$ENVF"

# 【DSH 兼容】工具调用 + 推理(思考强度):vLLM 注册了 mimo 工具/推理解析器;置空可关。
TOOL_PARSER="${TOOL_PARSER:-mimo}"
REASONING_PARSER="${REASONING_PARSER:-mimo}"
ARGS=(--model "$CKPT" --served-model-name mimo26
      --host 127.0.0.1 --port "$PORT"
      --tensor-parallel-size "$TP" --dtype bfloat16 --kv-cache-dtype bfloat16
      --max-model-len "$MAXLEN" --max-num-batched-tokens "$MBT" --max-num-seqs "$SEQS"
      --gpu-memory-utilization "$UTIL" --trust-remote-code
      --kernel-config '{"enable_jit_warmup": false}'
      --load-format "$LOAD")
[ -n "$TOOL_PARSER" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[ "${PROMPT_TOKENS_DETAILS:-1}" = "1" ] && ARGS+=(--enable-prompt-tokens-details)
[ "$MM" = "1" ] || ARGS+=(--language-model-only)
if [ "$MM" = "1" ]; then
  # 注意:不要把 JSON 默认值直接写进 ${VAR:-...},花括号会和展开的 } 冲突(会多出一个 })。
  if [ -n "${MM_LIMITS:-}" ]; then MM_LIM="$MM_LIMITS"; else MM_LIM="{\"image\":$PROMPTS,\"video\":0,\"audio\":0}"; fi
  ARGS+=(--limit-mm-per-prompt "$MM_LIM")
fi
[ "$SPEC_K" != "0" ] && ARGS+=(--speculative-config "{\"method\":\"mtp\",\"model\":\"$CKPT\",\"num_speculative_tokens\":$SPEC_K}")

LOGF="${LOGF:-/tmp/serve_mimo26_${PORT}.log}"
echo "[serve_mimo26] GPUS=$GPUS TP=$TP MAXLEN=$MAXLEN MBT=$MBT SEQS=$SEQS LOAD=$LOAD SPEC_K=$SPEC_K MM=$MM VERIFY=$VERIFY"
echo "[serve_mimo26] 日志 -> $LOGF"
setsid env PYTHONPATH="$VLLM_TREE" \
  XIAOTU_ENV_FILE="$ENVF" VLLM_EXPERTS_LOAD_DEVICE=cpu CUDA_VISIBLE_DEVICES="$GPUS" \
  VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
  "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$LOGF" 2>&1 < /dev/null &
echo "[serve_mimo26] pid=$! 已启动;等 READY:"
echo "  for i in \$(seq 1 90); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && { echo READY; break; }; sleep 20; done"
echo "  ⚠️ 加载约 40–60 分钟(161 GiB,磁盘瓶颈);日志可能因块缓冲看起来冻结 —— "
echo "     判活要看 worker 的 CPU ticks,不要只看日志(B103)。"
