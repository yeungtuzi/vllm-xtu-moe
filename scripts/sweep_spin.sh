#!/usr/bin/env bash
# 自旋窗口扫描:引擎 worker 在层间会自旋 spin_idle_us 等下一次调用,192 个线程
# 全速自旋会把整机烧穿(实测解码期间进程占 166–259 核,而真正的 MoE 算力只需要
# ~21 核),同时抢走驱动 GPU 的主线程 ⇒ `rest`(每层非 MoE 时间)被推高。
#
# 本机是共享主机,必须**同窗口**比:每个取值重启一次服务,跑同一基准,记录
# tok/s、每层 compute/rest、以及引擎进程在基准期间实际占用的核数。
#
# 用法:SPINS="5000 300 1000" scripts/sweep_spin.sh
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/dev-docs/report/tuning/logs"
OUT="$ROOT/dev-docs/report/tuning/spin_sweep.txt"
PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
SPINS="${SPINS:-5000 300 1000}"
SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","model":"/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master"}'

stop_server() {
  local pid; pid="$(cat "$LOGS/$1.pid" 2>/dev/null || true)"
  [ -n "${pid:-}" ] || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -KILL "$pid" 2>/dev/null || true
  pkill -9 -P "$pid" 2>/dev/null || true
}

cpu_cores() {  # $1=pid $2=秒 → 该窗口内平均占用核数
  local pid=$1 secs=$2 a b
  a=$(awk '{print $14+$15}' /proc/$pid/stat)
  sleep "$secs"
  b=$(awk '{print $14+$15}' /proc/$pid/stat)
  python3 -c "print(round(($b-$a)/$(getconf CLK_TCK)/$secs,1))"
}

: > "$OUT"
prev=""
for spin in $SPINS; do
  tag="spin${spin}"
  for t in "$tag" "$prev" aba_new ilp23 blk23 prof23 nat_k5; do [ -n "$t" ] && stop_server "$t"; done; sleep 8
  MODE=dsv4 TAG="$tag" PORT=8070 TP=1 MAXLEN=262144 SEQS=128 MAX_NBT=8192 \
    GPU_UTIL=0.85 KV_DTYPE=fp8_ds_mla KV_MEM_BYTES=8589934592 THREADS=192 OMP=96 \
    EAGER=0 EP=0 GPUS=2 ENV_EXTRA="XIAOTU_CD_TIMING=1 XIAOTU_MOE_SPIN_IDLE_US=$spin" \
    SPEC="$SPEC" scripts/tune_serve.sh >/dev/null 2>&1 || { echo "serve failed spin=$spin" | tee -a "$OUT"; continue; }
  pid=$(cat "$LOGS/$tag.pid")
  load=$(awk '{print $1}' /proc/loadavg)
  (L=128 C=1 N=2 OUT=200 TAG="spin${spin}_c1" SERVER_TAG="$tag" "$PY" scripts/bench_nat_client.py \
     | sed "s/^\[nat-client\] /spin=$spin load=$load /;s/ per-req.*//" | tee -a "$OUT") &
  bpid=$!
  cores=$(cpu_cores "$pid" 12)
  wait $bpid
  echo "spin=$spin 引擎解码期间占用 ≈ ${cores} 核" | tee -a "$OUT"
  grep -h "cd-timing" "$LOGS/$tag.log" | tail -2 | sed "s/^/spin=$spin /" | tee -a "$OUT"
  prev="$tag"
done
echo "[spin-sweep] done; 最后运行的 tag=$prev" | tee -a "$OUT"
