#!/usr/bin/env bash
# ShareGPT(真实对话)基准 —— 与 `bench_lat.sh`(随机 token)互补。
#
# 为什么需要它(用户 2026-09-16 提醒:本机已下载 ShareGPT):
#   * `--dataset-name random` 是**随机 token**,对**投机解码是最坏情况**(可预测性≈0、
#     接受率≈0 ⇒ draft 的 forward 是纯开销),用它评价 DSpark 会得出错误结论;
#   * ShareGPT 是真实人类对话:长度分布真实、语言有可预测性 ⇒ 才是投机解码该被考核的场景;
#   * 也顺带给出"真实长度分布"下的 TTFT/吞吐(而不是我们挑的 512/2048/8192)。
#
# 注意:vLLM 的 ShareGPT 加载器**取 `conversations[0].value` 当 prompt、`[1].value` 当
# completion 的参考长度**,**不套 chat template** ⇒ 对"快照里没有 chat_template"的
# DeepSeek-V4.x 也能用(这正是 `bench_lat.sh` 当初绕开 chat template 的原因)。
#
# 用法:
#   PORT=8260 TAG=sg1 CS="1 4" OUT=128 N=16 bash scripts/bench_sharegpt.sh
#   DATASET=/path/to/sharegpt.json ...    # 默认用本机已下载的那份
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/report/tuning/raw"; mkdir -p "$RAW"
PORT="${PORT:-8260}"
MODEL="${MODEL:-dsv41-xtu}"
DATASET="${DATASET:-/home/user/lvllm/ShareGPT_V3_unfiltered_cleaned_split.json}"
OUT="${OUT:-128}"
N="${N:-16}"
CS="${CS:-${C:-1}}"
SERVER_TAG="${SERVER_TAG:-unknown}"
TOKENIZER="${TOKENIZER:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
export HF_HUB_OFFLINE=1
export PATH="${BENCH_ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}/bin:$PATH"

[ -f "$DATASET" ] || { echo "[bench_sg] 数据集不存在: $DATASET"; exit 2; }

BASE_TAG="${TAG:-sg}"
for C in $CS; do
  TAG="${BASE_TAG}_c${C}"
  {
    echo "### $(date -Is) tag=$TAG server=$SERVER_TAG dataset=sharegpt C=$C N=$N out=$OUT"
    vllm bench serve \
      --backend openai --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
      --dataset-name sharegpt --dataset-path "$DATASET" \
      --sharegpt-output-len "$OUT" \
      --num-prompts "$N" --max-concurrency "$C" \
      --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
      --tokenizer "$TOKENIZER" \
      --save-result --result-dir "$RAW" --result-filename "$TAG.json"
  } > "$RAW/$TAG.log" 2>&1 || { echo "[bench_sg] FAILED tag=$TAG"; tail -20 "$RAW/$TAG.log"; continue; }

  python3 - "$RAW/$TAG.json" "$TAG" "$SERVER_TAG" "$C" "$N" "$OUT" <<'PY'
import json, sys
p, tag, server, c, n, out = sys.argv[1:7]
d = json.load(open(p))
def g(*ks):
    for k in ks:
        if k in d and d[k] is not None: return d[k]
    return 0
print(f"[bench_sg] {tag} (server={server}) C={c}: "
      f"agg {g('output_throughput'):.2f} tok/s  TPOT {g('mean_tpot_ms'):.2f} ms  "
      f"TTFT {g('mean_ttft_ms'):.0f} ms  "
      f"prompt_len~{g('mean_input_len','mean_prompt_len'):.0f}  completed={d.get('completed')}/{n} out={out}")
PY
done
