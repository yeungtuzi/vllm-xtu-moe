#!/usr/bin/env bash
# 【§555】启动后**形状预热**:修 §513/§555 量到的"首个长上下文请求付 237 s"问题。
#
# 实测(同机同日,TP=2/1M/常驻2层):
#   同形状第 1 次 TTFT = 237,405 ms(逐形状 Triton/TileLang JIT + graph 捕获)
#   同形状第 2 次 TTFT =   1,150 ms  ⇒ **稳态预填充 ~28,500 t/s,快 206×**
# ⇒ 用户可见的"长上下文卡 4 分钟"完全是**冷启动**,必须由启动时的预热吸收掉。
#
# 用法(serve_v41.sh 已内建:WARMUP=1 时在 READY 之后自动调用):
#   PORT=8077 LENS="8192 32768" bash scripts/warmup_shapes.sh
# 每个长度发 1 条请求(--random-output-len 1,只付预填充),失败不致命(打告警继续启动)。
set -uo pipefail
PORT="${PORT:-8077}"
LENS="${LENS:-8192 32768}"
MODEL="${MODEL:-dsv41}"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
export PATH="${BENCH_ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}/bin:$PATH"
export HF_HUB_OFFLINE=1
[ "${WARMUP:-1}" = "1" ] || { echo "[warmup] 跳过(WARMUP=0)"; exit 0; }
for L in $LENS; do
  echo "[warmup] 预热形状: input=$L ..."
  t0=$(date +%s)
  out="$(vllm bench serve --backend openai --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
        --dataset-name random --random-input-len "$L" --random-output-len 1 \
        --num-prompts 1 --max-concurrency 1 --tokenizer "$CKPT" 2>&1 | grep -E 'Mean TTFT|Successful' | tr '\n' ' ')"
  t1=$(date +%s)
  echo "[warmup] input=$L 完成($((t1-t0))s): $out"
done
echo "[warmup] 全部形状预热完成 —— 真实请求不再为首个长上下文付 JIT 成本"
