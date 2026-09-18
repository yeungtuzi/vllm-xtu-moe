#!/usr/bin/env bash
# 自然文本 + 精确上下文长度的单/双路基准(目标场景:每 token 延迟)。
#
# 背景(§36):sharegpt 默认采样只给我们 ~21 token/条 的短 prompt,而
# `--dataset-name random` 的随机 token 会把 draft 接受率打到 0.2(位置0),
# 两者都不能代表目标场景。本脚本用 scripts/make_nat_dataset.py 生成的
# nat<LEN>.jsonl(真实文本、精确 LEN 个 token)。
#
# 用法:L=512 C=1 PORT=8070 scripts/bench_nat.sh
#      L=512 C=2 N=8 OUT=128 TAG=xxx scripts/bench_nat.sh
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/dev-docs/report/tuning/raw"; mkdir -p "$RAW"
PORT="${PORT:-8070}"
MODEL="${MODEL:-DeepSeek-V4-Flash-xiaotu}"
L="${L:-512}"
C="${C:-1}"
N="${N:-8}"
OUT="${OUT:-128}"
SERVER_TAG="${SERVER_TAG:-unknown}"
TAG="${TAG:-nat${L}_c${C}_n${N}_out${OUT}}"
DS="$ROOT/dev-docs/report/tuning/datasets/nat${L}.jsonl"
TOKENIZER="${TOKENIZER:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
export HF_HUB_OFFLINE=1
export PATH=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin:$PATH

{
  echo "### $(date -Is) tag=$TAG server_tag=$SERVER_TAG ctx=$L C=$C N=$N out=$OUT (natural text, ignore-eos)"
  uptime
  vllm bench serve \
    --backend openai --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --dataset-name custom --dataset-path "$DS" --custom-output-len "$OUT" \
    --num-prompts "$N" --max-concurrency "$C" --ignore-eos \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
    --tokenizer "$TOKENIZER" \
    --save-result --result-dir "$RAW" --result-filename "$TAG.json"
} > "$RAW/$TAG.log" 2>&1 || { echo "[bench_nat] FAILED tag=$TAG"; tail -20 "$RAW/$TAG.log"; exit 1; }

python3 - "$RAW/$TAG.json" "$TAG" "$SERVER_TAG" "$C" "$N" "$OUT" "$L" "$ROOT/dev-docs/report/tuning/summary.jsonl" <<'PY'
import json, sys, datetime
src, tag, server_tag, C, N, out, ctx, dst = sys.argv[1:]
d = json.load(open(src))
rec = {
    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    "tag": tag, "server_tag": server_tag, "natural_text": 1,
    "concurrency": int(C), "num_prompts": int(N), "output_len": int(out),
    "input_len": f"nat{ctx}", "request_rate": "inf",
    "completed": d.get("completed"), "failed": d.get("failed"),
    "duration_s": round(d.get("duration", 0), 2),
    "out_tok_per_s": round(d.get("output_throughput", 0), 2),
    "per_stream_tok_per_s": round(d.get("output_throughput", 0) / int(C), 2),
    "total_tok_per_s": round(d.get("total_token_throughput", 0), 2),
    "mean_ttft_ms": round(d.get("mean_ttft_ms", 0), 1),
    "p99_ttft_ms": round(d.get("p99_ttft_ms", 0), 1),
    "mean_tpot_ms": round(d.get("mean_tpot_ms", 0), 2),
    "p99_tpot_ms": round(d.get("p99_tpot_ms", 0), 2),
}
open(dst, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[bench_nat] {tag}: {rec['out_tok_per_s']} tok/s ({rec['per_stream_tok_per_s']}/stream) "
      f"ttft {rec['mean_ttft_ms']} ms tpot {rec['mean_tpot_ms']} ms")
PY
