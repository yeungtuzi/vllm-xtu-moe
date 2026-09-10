#!/usr/bin/env bash
# ShareGPT 压测客户端(vllm bench serve 包装)+ 结果归集
#
# 用法:
#   TAG=dsv4_tp1_c8_out4k PORT=8081 MODEL=DeepSeek-V4-Flash-xiaotu C=8 N=32 OUT=4096 \
#     scripts/tune_client.sh
#
# 可选:RATE=inf(默认,一次性全发)/ 数值(按 RPS 发);SERVER_TAG=<服务端 tag,用于归集>;
#       IN_LEN=N(合成输入长度,用于长上下文点,会改用 random 数据集)。
#
# 结果:
#   report/tuning/raw/<TAG>.json     vLLM 原生结果(含逐请求 ttft/itl)
#   report/tuning/raw/<TAG>.log      客户端完整输出
#   report/tuning/summary.jsonl      一行一条汇总(含 server_tag / 并发 / 输出长度)
#
# License: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/report/tuning/raw"
mkdir -p "$RAW"

PORT="${PORT:-8081}"
MODEL="${MODEL:-DeepSeek-V4-Flash-xiaotu}"
C="${C:-8}"
N="${N:-32}"
OUT="${OUT:-4096}"
RATE="${RATE:-inf}"
TAG="${TAG:-${MODEL}_c${C}_n${N}_out${OUT}_$(date +%H%M%S)}"
SERVER_TAG="${SERVER_TAG:-unknown}"
IN_LEN="${IN_LEN:-}"
DATASET="${DATASET:-/home/user/lvllm/ShareGPT_V3_unfiltered_cleaned_split.json}"
TOKENIZER="${TOKENIZER:-}"
export HF_HUB_OFFLINE=1
export PATH=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin:$PATH

ARGS=(
  --backend openai --host 127.0.0.1 --port "$PORT" --model "$MODEL"
  --num-prompts "$N" --max-concurrency "$C" --request-rate "$RATE"
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99
  --ignore-eos
  --save-result --result-dir "$RAW" --result-filename "$TAG.json"
)
if [ -n "$IN_LEN" ]; then
  ARGS+=(--dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT")
else
  ARGS+=(--dataset-name sharegpt --dataset-path "$DATASET" --sharegpt-output-len "$OUT")
fi
[ -n "$TOKENIZER" ] && ARGS+=(--tokenizer "$TOKENIZER")

{
  echo "### $(date -Is) tag=$TAG server_tag=$SERVER_TAG C=$C N=$N OUT=$OUT RATE=$RATE IN_LEN=${IN_LEN:-sharegpt}"
  uptime
  vllm bench serve "${ARGS[@]}"
} > "$RAW/$TAG.log" 2>&1 || { echo "[tune_client] FAILED tag=$TAG"; tail -20 "$RAW/$TAG.log"; exit 1; }

python3 - "$RAW/$TAG.json" "$TAG" "$SERVER_TAG" "$C" "$N" "$OUT" "$RATE" "${IN_LEN:-sharegpt}" \
        "$ROOT/report/tuning/summary.jsonl" <<'PY'
import json, sys, os, datetime
src, tag, server_tag, C, N, out, rate, inlen, dst = sys.argv[1:]
d = json.load(open(src))
rec = {
    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    "tag": tag, "server_tag": server_tag,
    "concurrency": int(C), "num_prompts": int(N), "output_len": int(out),
    "input_len": inlen, "request_rate": rate,
    "completed": d.get("completed"), "failed": d.get("failed"),
    "duration_s": round(d.get("duration", 0), 3),
    "req_per_s": round(d.get("request_throughput", 0), 3),
    "out_tok_per_s": round(d.get("output_throughput", 0), 2),
    "total_tok_per_s": round(d.get("total_token_throughput", 0), 2),
    "mean_ttft_ms": round(d.get("mean_ttft_ms", 0), 1),
    "p99_ttft_ms": round(d.get("p99_ttft_ms", 0), 1),
    "mean_tpot_ms": round(d.get("mean_tpot_ms", 0), 2),
    "p99_tpot_ms": round(d.get("p99_tpot_ms", 0), 2),
    "mean_e2el_ms": round(d.get("mean_e2el_ms", 0), 1),
    "total_input_tokens": d.get("total_input_tokens"),
    "total_output_tokens": d.get("total_output_tokens"),
}
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "a") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("[tune_client] " + json.dumps(rec, ensure_ascii=False))
PY
echo "[tune_client] done tag=$TAG -> $RAW/$TAG.json"
