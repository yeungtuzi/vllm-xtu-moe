#!/usr/bin/env bash
# 复现"用户测 lk-moe 生产环境"的那条基准(process_data/scripts/bench.sh),
# 以便和本插件做**同形对比**:C=4、50 prompts、ShareGPT 自带输出长度、不忽略 EOS。
#
# 生产脚本原文:
#   export LVLLM_MOE_NUMA_ENABLED=1
#   vllm bench serve --backend openai --base-url http://localhost:8070 \
#     --model DeepSeek-V4-Flash-0731 --dataset-name sharegpt --dataset-path <ShareGPT> \
#     --num-prompts 50 --max-concurrency 4 --tokenizer <0731 快照>
#
# 用法:TAG=dsv4_tp1_lkshape PORT=8090 scripts/bench_lk_shape.sh
# 结果:dev-docs/report/tuning/raw/<TAG>.{json,log} + 一行 summary.jsonl(标记 lk_shape=1)
#
# License: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/dev-docs/report/tuning/raw"
mkdir -p "$RAW"

PORT="${PORT:-8090}"
MODEL="${MODEL:-DeepSeek-V4-Flash-xiaotu}"
C="${C:-4}"
N="${N:-50}"
TAG="${TAG:-lk_shape_c${C}_n${N}_$(date +%H%M%S)}"
SERVER_TAG="${SERVER_TAG:-unknown}"
DATASET="${DATASET:-/home/user/lvllm/ShareGPT_V3_unfiltered_cleaned_split.json}"
TOKENIZER="${TOKENIZER:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
export HF_HUB_OFFLINE=1
export PATH=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin:$PATH

# 注意:**不加** --ignore-eos、**不指定** --sharegpt-output-len(与生产脚本一致)
{
  echo "### $(date -Is) tag=$TAG server_tag=$SERVER_TAG lk-shape C=$C N=$N (sharegpt default out, EOS honored)"
  uptime
  vllm bench serve \
    --backend openai --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --dataset-name sharegpt --dataset-path "$DATASET" \
    --num-prompts "$N" --max-concurrency "$C" \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
    --tokenizer "$TOKENIZER" \
    --save-result --result-dir "$RAW" --result-filename "$TAG.json"
} > "$RAW/$TAG.log" 2>&1 || { echo "[bench_lk_shape] FAILED tag=$TAG"; tail -25 "$RAW/$TAG.log"; exit 1; }

python3 - "$RAW/$TAG.json" "$TAG" "$SERVER_TAG" "$C" "$N" "$ROOT/dev-docs/report/tuning/summary.jsonl" <<'PY'
import json, sys, datetime
src, tag, server_tag, C, N, dst = sys.argv[1:]
d = json.load(open(src))
rec = {
    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    "tag": tag, "server_tag": server_tag, "lk_shape": 1,
    "concurrency": int(C), "num_prompts": int(N),
    "output_len": "sharegpt-default(eos)",
    "input_len": "sharegpt", "request_rate": "inf",
    "completed": d.get("completed"), "failed": d.get("failed"),
    "duration_s": round(d.get("duration", 0), 2),
    "req_per_s": round(d.get("request_throughput", 0), 4),
    "out_tok_per_s": round(d.get("output_throughput", 0), 2),
    "total_tok_per_s": round(d.get("total_token_throughput", 0), 2),
    "mean_ttft_ms": round(d.get("mean_ttft_ms", 0), 1),
    "p99_ttft_ms": round(d.get("p99_ttft_ms", 0), 1),
    "mean_tpot_ms": round(d.get("mean_tpot_ms", 0), 2),
    "p99_tpot_ms": round(d.get("p99_tpot_ms", 0), 2),
    "mean_e2el_ms": round(d.get("mean_e2el_ms", 0), 1),
}
open(dst, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[bench_lk_shape] {tag}: {rec['out_tok_per_s']} out tok/s, "
      f"ttft {rec['mean_ttft_ms']} ms, tpot {rec['mean_tpot_ms']} ms, "
      f"completed {rec['completed']}/{N}")
PY
