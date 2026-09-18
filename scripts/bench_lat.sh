#!/usr/bin/env bash
# 固定协议的延迟/吞吐基准(不依赖 tokenizer 的 chat_template)。
#
# 背景:mainline 的 `vllm bench serve --dataset-name custom` 会调用
# `tokenizer.apply_chat_template`,而 DeepSeek-V4-Flash 的快照里
# `tokenizer_config.json` **没有** chat_template ⇒ 直接报
# "Cannot use chat template functions..."。本脚本改用 `--dataset-name random`
# (token 级、可复现,服务端不需要 chat 模板),协议固定:
#
#   输入 512 token / 输出 128 token / num-prompts 8 / C=1,2,4
#
# 用法:
#   PORT=8070 TAG=mytag C=1 bash scripts/bench_lat.sh
#   PORT=8070 TAG=mytag CS="1 2 4" bash scripts/bench_lat.sh
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/dev-docs/report/tuning/raw"; mkdir -p "$RAW"
PORT="${PORT:-8070}"
MODEL="${MODEL:-DeepSeek-V4-Flash-xiaotu}"
L="${L:-512}"
OUT="${OUT:-128}"
N="${N:-8}"
CS="${CS:-${C:-1}}"
SERVER_TAG="${SERVER_TAG:-unknown}"
TOKENIZER="${TOKENIZER:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
export HF_HUB_OFFLINE=1
# 基准**客户端**用哪个 env 的 `vllm bench serve`。默认保持不变(我们的 env);
# 同机 A/B(scripts/ab_lvllm_vs_xiaotu.sh)会把两边都指向被测服务所在的 env。
export PATH="${BENCH_ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}/bin:$PATH"

BASE_TAG="${TAG:-lat}"
for C in $CS; do
  TAG="${BASE_TAG}_c${C}"
  {
    echo "### $(date -Is) tag=$TAG server_tag=$SERVER_TAG L=$L out=$OUT C=$C N=$N (random tokens)"
    vllm bench serve \
      --backend openai --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
      --dataset-name random --random-input-len "$L" --random-output-len "$OUT" \
      --num-prompts "$N" --max-concurrency "$C" \
      --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
      --tokenizer "$TOKENIZER" \
      --save-result --result-dir "$RAW" --result-filename "$TAG.json"
  } > "$RAW/$TAG.log" 2>&1 || { echo "[bench_lat] FAILED tag=$TAG"; tail -20 "$RAW/$TAG.log"; continue; }

  python3 - "$RAW/$TAG.json" "$TAG" "$SERVER_TAG" "$C" "$N" "$OUT" "$L" <<'PY'
import json, sys
src, tag, server_tag, C, N, out, L = sys.argv[1:]
d = json.load(open(src))
print(f"[bench_lat] {tag} (server={server_tag}) C={C}: "
      f"agg {d.get('output_throughput',0):.2f} tok/s "
      f"({d.get('output_throughput',0)/int(C):.2f}/stream) "
      f"TPOT {d.get('mean_tpot_ms',0):.2f} ms p99 {d.get('p99_tpot_ms',0):.2f} "
      f"TTFT {d.get('mean_ttft_ms',0):.0f} ms completed={d.get('completed')}/{N}"
      f" | rec={{'tag':'{tag}','server_tag':'{server_tag}','C':{C},'L':{L},'out':{out},"
      f"'agg':{d.get('output_throughput',0):.2f},'tpot':{d.get('mean_tpot_ms',0):.2f},"
      f"'ttft':{d.get('mean_ttft_ms',0):.1f}}}")
PY
done
