#!/usr/bin/env bash
# 投机解码 draft 长度(k)扫描 —— 单卡 256K 已知好配置,只变 num_speculative_tokens。
#
# 动机(report/tuning/NOTES.md §35):实测逐位置接受率为
#   p = [0.667, 0.393, 0.179, 0.048, 0.036]
# 而我们的 MoE 成本 ∝ (1+k)·C(每个 (row,expert) 对都要重新流一遍 12.6 MB 专家权重),
# 于是第 4/5 个 draft token 的期望收益(0.048/0.036)远低于其代价 ⇒ 预测 k=2 最优:
#   k=1 17.9 / k=2 19.4 / k=3 18.7 / k=5 15.9 tok/s(模型)
# 本脚本做真实 A/B:对每个 k 起一次服务,跑 C=1 与 C=2 的定长解码,并从服务端日志
# 抓 "Mean acceptance length" 与逐位置接受率(验证 p 是否随 k 变)。
#
# 用法:K_LIST="2 3 5" scripts/sweep_spec_k.sh
#
# License: Apache-2.0
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/report/tuning/logs"
PORT="${PORT:-8070}"
K_LIST="${K_LIST:-2 3 5}"
MODEL_DIR="${MODEL_DIR:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
DRAFT="${DRAFT:-probabilistic}"
N1="${N1:-4}"; N2="${N2:-8}"; OUT="${OUT:-200}"; IN_LEN="${IN_LEN:-1024}"

kill_port() {
  local pid
  pid="$(cat "$LOGS/$1.pid" 2>/dev/null || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -KILL "$pid" 2>/dev/null || true
    # 残留的 EngineCore 子进程(front-end 退出后可能活着)
    [ -n "$pid" ] && pkill -9 -P "$pid" 2>/dev/null || true
  fi
}

# 先把上一轮的实验服务(grp_c)关掉,避免抢 8070 端口/抢 CPU 线程
kill_port grp_c
kill_port serve_8070
sleep 5

for k in $K_LIST; do
  SRV="spec${k}"
  echo "=================== $(date -Is) k=$k tag=$SRV"
  kill_port "$SRV"
  SPEC="{\"method\":\"dspark\",\"num_speculative_tokens\":${k},\"draft_sample_method\":\"${DRAFT}\",\"model\":\"${MODEL_DIR}\"}"
  MODE=dsv4 TAG="$SRV" PORT="$PORT" TP=1 MAXLEN=262144 SEQS=128 MAX_NBT=8192 \
    GPU_UTIL=0.85 KV_DTYPE=fp8_ds_mla KV_MEM_BYTES=8589934592 \
    THREADS=192 OMP=96 EAGER=0 EP=0 GPUS=2 SPEC="$SPEC" \
    scripts/tune_serve.sh || { echo "[sweep] start failed k=$k"; continue; }

  TAG="sweep_spec${k}_c1" PORT="$PORT" C=1 N="$N1" OUT="$OUT" IN_LEN="$IN_LEN" \
    SERVER_TAG="$SRV" scripts/tune_client.sh || echo "[sweep] bench c1 failed k=$k"
  TAG="sweep_spec${k}_c2" PORT="$PORT" C=2 N="$N2" OUT="$OUT" IN_LEN="$IN_LEN" \
    SERVER_TAG="$SRV" scripts/tune_client.sh || echo "[sweep] bench c2 failed k=$k"

  {
    echo "### k=$k ($(date -Is))"
    grep -h "Mean acceptance length" "$LOGS/$SRV.log" 2>/dev/null | tail -12
  } > "$LOGS/$SRV.acceptance"
  echo "--- k=$k acceptance:"; tail -4 "$LOGS/$SRV.acceptance"

  kill_port "$SRV"
  sleep 10
done
echo "=================== $(date -Is) sweep done"
