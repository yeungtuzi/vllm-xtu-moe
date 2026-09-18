#!/usr/bin/env bash
# v0.2 同机 A/B 回归基线:**fork + 我们的引擎** vs **主线 + 我们的插件**。
#
# 为什么必须做这个 A/B(而不是跟文档里的数字比):
#   同一个 `xiaotu_moe.so`、同一份 `bench_lat.sh`、同一台机器,
#   fork 编排 **26.81 ms** TPOT,主线曾 **1225 ms**(47×)。
#   差异**不在代码 diff 里**,只有把同一份引擎放进两种编排对跑才会现形
#   —— 这正是"支持主线"唯一可验证的定义(见 dev-docs/PLUGIN_INTERFACE.md)。
#
# 用法:
#   bash scripts/ab_mainline_vs_fork.sh                 # 两边都起,跑 C=1
#   CS="1 2 4" bash scripts/ab_mainline_vs_fork.sh      # 跑 C=1/2/4
#   SKIP_MAINLINE=1 bash scripts/ab_mainline_vs_fork.sh # 复用已在跑的 8071
#   MAINLINE_PORT=8071 FORK_PORT=8070 ...
#
# 产物:dev-docs/report/tuning/raw/ab_{mainline,fork}_c<N>.json + 终端对照表。
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAINLINE_PORT="${MAINLINE_PORT:-8071}"
FORK_PORT="${FORK_PORT:-8070}"
CS="${CS:-1}"
TP="${TP:-2}"
GPUS="${GPUS:-0,1}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAXLEN="${MAXLEN:-8192}"
SEQS="${SEQS:-8}"
THREADS="${THREADS:-60}"
RESIDENT="${RESIDENT:-0-11}"
MBT="${MBT:-256}"

MAIN_ENV="${MAIN_ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
FORK_ENV="${FORK_ENV:-/home/user/anaconda3/envs/lkxtu}"

READY_TIMEOUT_S="${READY_TIMEOUT_S:-900}"

note() { printf '\033[36m[ab]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[ab]\033[0m %s\n' "$*"; }

# 起一个服务并等到 ready;失败返回 1。参数:标签 端口 日志 启动命令...
launch_and_wait() {
  local tag="$1" port="$2" log="$3"; shift 3
  rm -f "$log"
  setsid "$@" > "$log" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  # 【坑】"Application startup complete" 是 vLLM 写进**服务自己的日志**
  # (dev-docs/report/tuning/logs/$tag.log)的;我们捕获的 stdout 里只有 serve 脚本打印的
  # "[mainline] READY tag=..." / "[lk_port] READY tag=..."。两个文件都要看。
  local slog="$ROOT/dev-docs/report/tuning/logs/$tag.log"
  local waited=0
  while [ "$waited" -lt "$READY_TIMEOUT_S" ]; do
    if grep -qa "READY tag=" "$log" 2>/dev/null \
       || grep -qa "Application startup complete" "$slog" 2>/dev/null; then
      note "$tag READY(${waited}s) on :$port"; return 0
    fi
    if [ "$waited" -ge 60 ] && ! pgrep -f "VLLM::Worke[r]" >/dev/null 2>&1; then
      warn "$tag 进程退出;日志尾部:"; tail -8 "$slog" 2>/dev/null || tail -8 "$log"; return 1
    fi
    sleep 15; waited=$((waited+15))
  done
  warn "$tag 等待超时(${READY_TIMEOUT_S}s)"; tail -8 "$slog" 2>/dev/null || tail -8 "$log"; return 1
}

stop_tag() {
  local tag="$1"
  local pf="$ROOT/dev-docs/report/tuning/logs/$tag.pid"
  [ -f "$pf" ] && { kill -TERM "$(cat "$pf")" 2>/dev/null || true; }
  sleep 10
  local p
  for p in $(ps -eo pid,args 2>/dev/null | grep "VLLM::Worke[r]" | awk '{print $1}'); do
    kill -TERM "$p" 2>/dev/null || true
  done
  sleep 6
  for p in $(ps -eo pid,args 2>/dev/null | grep "VLLM::Worke[r]" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
  sleep 4
}

# 用 bench_lat.sh 跑一个配置,回显 "agg tpot ttft"
run_bench() {
  local port="$1" tag="$2" c="$3" model="$4"
  PORT="$port" TAG="$tag" MODEL="$model" SERVER_TAG="$tag" CS="$c" \
    timeout 1800 bash scripts/bench_lat.sh >/dev/null 2>&1 || true
  local j="$ROOT/dev-docs/report/tuning/raw/${tag}_c${c}.json"
  [ -f "$j" ] || { echo "NA NA NA"; return; }
  python3 -c "
import json;d=json.load(open('$j'))
print('%.2f %.2f %.1f'%(d.get('output_throughput',0),
      d.get('mean_tpot_ms') or 0, d.get('mean_ttft_ms') or 0))"
}

# ---------------------------- 1) 主线 ---------------------------------------
MAIN_MODEL="DeepSeek-V4-Flash-xiaotu"
if [ "${SKIP_MAINLINE:-0}" = "1" ]; then
  note "复用已在跑的主线 :$MAINLINE_PORT"
else
  note "启动主线(我们的插件)…"
  launch_and_wait ab_mainline "$MAINLINE_PORT" /tmp/ab_mainline.log \
    env ENV="$MAIN_ENV" TAG=ab_mainline PORT="$MAINLINE_PORT" TP="$TP" GPUS="$GPUS" \
        GPU_UTIL="$GPU_UTIL" MAXLEN="$MAXLEN" SEQS="$SEQS" MBT="$MBT" GP_MIN="${GP_MIN:-0}" \
        THREADS="$THREADS" RESIDENT="$RESIDENT" EAGER="${EAGER:-0}" \
        bash scripts/serve_mainline.sh \
    || { warn "主线起不来,终止"; exit 1; }
fi

# ---------------------------- 2) fork + 我们的引擎 ---------------------------
if [ "${SKIP_FORK:-0}" = "1" ]; then
  note "复用已在跑的 fork :$FORK_PORT"
else
  note "启动 fork + 我们的引擎…"
  launch_and_wait ab_fork "$FORK_PORT" /tmp/ab_fork.log \
    env ENV="$FORK_ENV" TAG=ab_fork PORT="$FORK_PORT" TP="$TP" GPUS="$GPUS" \
        GPU_UTIL="$GPU_UTIL" MAXLEN="$MAXLEN" SEQS="$SEQS" MBT="$MBT" MINBATCH=0 \
        PREFETCH=1 EAGER=0 THREADS="$THREADS" SPEC=0 RESIDENT="$RESIDENT" \
        bash scripts/serve_lk_port.sh \
    || { warn "fork 起不来;只报主线"; SKIP_FORK=0; }
fi

# fork 的 served-model-name 也是这个(两边的 --served-model-name 一致)
FORK_MODEL="$("$FORK_ENV/bin/python" -c "
import json,urllib.request
try: print(json.loads(urllib.request.urlopen('http://127.0.0.1:$FORK_PORT/v1/models',timeout=30).read())['data'][0]['id'])
except Exception: print('$MAIN_MODEL')" 2>/dev/null | tail -1)"

# ---------------------------- 3) 跑基准 + 对照表 -----------------------------
printf '\n'
printf '%-12s %10s %12s %12s %12s\n' "并发" "配置" "agg t/s" "TPOT ms" "TTFT ms"
printf -- '--------------------------------------------------------------\n'
declare -A RES
for c in $CS; do
  m=$(run_bench "$MAINLINE_PORT" "ab_mainline" "$c" "$MAIN_MODEL")
  printf '%-12s %10s %12s %12s %12s\n' "C=$c" "mainline" $m
  RES["mainline_c$c"]="$m"
  if [ "${SKIP_FORK:-0}" != "1" ]; then
    f=$(run_bench "$FORK_PORT" "ab_fork" "$c" "$FORK_MODEL")
    printf '%-12s %10s %12s %12s %12s\n' "" "fork" $f
    RES["fork_c$c"]="$f"
    # 比值(用 TPOT:它是排除预填充的 steady-state 每 token 时间,唯一的可比口径)
    python3 - "$m" "$f" "$c" <<'PY' || true
import sys
m=[float(x) for x in sys.argv[1].split()]
f=[float(x) for x in sys.argv[2].split()]
c=sys.argv[3]
if m[1]>0 and f[1]>0:
    print(f"             ⇒ C={c} 主线/fork TPOT 比 = {m[1]/f[1]:.2f}×"
          + ("  ✅" if m[1]/f[1] < 1.5 else "  ⚠️ 退化"))
PY
  fi
done
printf -- '--------------------------------------------------------------\n'
note "口径提醒:TPOT 已排除预填充;不要用 wall/out_tokens(会把预填充算进去,见 PLUGIN_INTERFACE §3)"
note "raw: dev-docs/report/tuning/raw/ab_{mainline,fork}_c*.json"

if [ "${KEEP_SERVERS:-0}" != "1" ]; then
  note "清理服务(KEEP_SERVERS=1 可保留)…"
  [ "${SKIP_MAINLINE:-0}" = "1" ] || stop_tag ab_mainline
  [ "${SKIP_FORK:-0}" = "1" ] || stop_tag ab_fork
fi
