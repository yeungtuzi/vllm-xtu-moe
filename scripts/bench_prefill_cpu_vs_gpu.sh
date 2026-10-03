#!/usr/bin/env bash
# CPU prefill vs GPU prefill —— prompt 长度 × MBT 的 tok/s 对比矩阵(§目标 2026-10-03)
#
# 设计(为什么这样写):
#   * 每个 (臂, MBT) **独立重启服务** —— 不靠运行时切阈值,因为阈值切到 0 会让引擎
#     释放源权重,再切回来就没有可流式的源 ⇒ 会给"CPU 臂"引入"源权重双份常驻"的混淆。
#   * PREFIX_CACHE=0 + LMCACHE=0 + UNIQUE=1 —— 否则量到的是缓存命中,不是真预填充
#     (probe_ttft.py 头部的 §591 尺子教训)。
#   * 每个配置先跑一遍**丢弃的 warmup pass**,再跑测量 pass ⇒ 报告的才是 warmup 状态。
#   * TP=2 一律 GPU1,2(IRON_RULES R19:GPU0 只有 PCIe x8)。
#   * 失败即记录原因(例如 MBT=16384 可能因激活显存不可用),不静默跳过。
#
# 用法: bash scripts/bench_prefill_cpu_vs_gpu.sh
# 输出: /tmp/bench_prefill/ 下的 jsonl + summary.tsv

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"

MBTS="${MBTS:-4096 8192 16384}"
LENS="${LENS:-128 1024 4096 16384}"
# ⚠️ probe_ttft.py 的 LENS 是**逗号**分隔(split(",")),不是空格 ⇒ 必须转换
LENS_CSV="${LENS// /,}"
REP="${REP:-3}"
PORT="${PORT:-8091}"
MAXLEN="${MAXLEN:-131072}"
GPU_UTIL="${GPU_UTIL:-0.70}"
# MBT=16384 时激活工作区 ∝ chunk 会 OOM(实测 aten::new_empty 失败,CPU 臂在 16k 请求上同样中招)
# ⇒ 可用 KVB 压缩 KV 池、用 GPU_UTIL 腾出激活空间来跑那一档。
KVB="${KVB:-3221225472}"
GPUS="${GPUS:-1,2}"
TP="${TP:-2}"
# ⚠️【2026-10-03 用户提醒】serve_v41.sh 的线程默认是 **60**,那是 **decode 口径**
#   (R8:4 核/CCD 即饱和)。用它测 prefill 会**系统性压低 CPU 臂**:
#   实测 MBT=8192,CPU 60 → 188 线程:1k 274→303(+10.5%)、4k 285→333(+16.9%)、
#   16k 303→361(+19.4%)。⇒ **两臂必须用同一个 prefill 口径的线程数**,
#   取"全核 −4"(用户口径,lk-moe 也是这么留核的)。
THREADS="${THREADS:-188}"

OUTDIR="${OUTDIR:-/tmp/bench_prefill}"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
FAILED="$OUTDIR/failed.tsv"
PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
READY_TIMEOUT="${READY_TIMEOUT:-900}"

: > "$SUMMARY"; : > "$FAILED"
printf 'arm\tmbt\tlen_target\tprompt_tokens\tcached_tokens\tttft_s\tprefill_tok_per_s\tlabel\n' >> "$SUMMARY"
printf 'arm\tmbt\treason\n' >> "$FAILED"

log(){ echo "[bench $(date '+%H:%M:%S')] $*"; }

wait_ready(){  # $1=tag
  # ⚠️ 不能写成 `local a="$1" b="${a}"` —— bash 把整条 local 一起展开,set -u 下会报 unbound
  local tag="$1"
  local lf="dev-docs/report/tuning/logs/${tag}.log"
  local i
  for i in $(seq 1 $((READY_TIMEOUT/10))); do
    if grep -q "Application startup complete" "$lf" 2>/dev/null; then return 0; fi
    if grep -qE "Traceback|OutOfMemoryError|CUDA out of memory|Engine core initialization failed|ValueError" "$lf" 2>/dev/null; then
      return 2
    fi
    if ! kill -0 "$(cat "dev-docs/report/tuning/logs/${tag}.pid" 2>/dev/null)" 2>/dev/null; then
      return 3
    fi
    sleep 10
  done
  return 4
}

run_one(){  # $1=mbt $2=arm $3=threshold
  local mbt="$1"
  local arm="$2"
  local thr="$3"
  local tag="pf_${arm}_${mbt}"
  local lf="dev-docs/report/tuning/logs/${tag}.log"
  local rc
  log "START arm=$arm mbt=$mbt thr=$thr tag=$tag"
  bash scripts/proc.sh spawn "$tag" env \
    LMCACHE=0 MAXLEN="$MAXLEN" MBT="$mbt" MAXSEQS=1 GPUS="$GPUS" TP="$TP" GPU_UTIL="$GPU_UTIL" \
    COMPILE=1 EAGER=0 SPEC=0 KV_DTYPE=fp8_ds_mla KV_CACHE_BYTES="$KVB" \
    PREFIX_CACHE=0 WARMUP=0 LOAD=auto PORT="$PORT" TAG="$tag" \
    VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$thr" XIAOTU_MOE_THREADS="$THREADS" \
    bash scripts/serve_v41.sh >/dev/null

  wait_ready "$tag"; rc=$?
  if [ "$rc" -ne 0 ]; then
    log "FAIL arm=$arm mbt=$mbt rc=$rc"
    printf '%s\t%s\t%s\n' "$arm" "$mbt" "startup rc=$rc: $(grep -m1 -E 'Traceback|OutOfMemoryError|CUDA out of memory|ValueError|Error' "$lf" 2>/dev/null | cut -c1-180)" >> "$FAILED"
    bash scripts/proc.sh stop "$tag" >/dev/null 2>&1
    return 1
  fi
  log "READY arm=$arm mbt=$mbt"

  # warmup pass(丢弃)
  PORT="$PORT" LENS="$LENS_CSV" REP=1 UNIQUE=1 LABEL="${tag}_warm" OUT="$OUTDIR/${tag}_warm.jsonl" \
    "$PY" scripts/probe_ttft.py > "$OUTDIR/${tag}_warm.txt" 2>&1
  log "WARMUP done arm=$arm mbt=$mbt"

  # 测量 pass
  PORT="$PORT" LENS="$LENS_CSV" REP="$REP" UNIQUE=1 LABEL="${tag}" OUT="$OUTDIR/${tag}.jsonl" \
    "$PY" scripts/probe_ttft.py > "$OUTDIR/${tag}.txt" 2>&1

  "$PY" - "$OUTDIR/${tag}.jsonl" "$SUMMARY" "$arm" "$mbt" <<'PYEOF' >> "$OUTDIR/parse.err" 2>&1
import json,sys
src,summary,arm,mbt = sys.argv[1:5]
rows=[json.loads(l) for l in open(src) if l.strip().startswith("{")]
with open(summary,"a") as f:
    for r in rows:
        f.write(f"{arm}\t{mbt}\t{r.get('target_len')}\t{r.get('prompt_tokens')}\t"
                f"{r.get('cached_tokens')}\t{r.get('ttft_s')}\t{r.get('prefill_tok_per_s')}\t{r.get('label')}\n")
PYEOF
  log "DONE arm=$arm mbt=$mbt"
  bash scripts/proc.sh stop "$tag" >/dev/null 2>&1
  sleep 5
}

log "matrix: MBTS=[$MBTS] LENS=[$LENS] REP=$REP TP=$TP GPUS=$GPUS util=$GPU_UTIL THREADS=$THREADS"
for mbt in $MBTS; do
  run_one "$mbt" cpu 0
  run_one "$mbt" gpu 1
done
log "ALL DONE summary=$SUMMARY failed=$FAILED"
