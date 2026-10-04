#!/usr/bin/env bash
# 【2026-10-05】卡死看门狗 —— 目标:当场抓住"逃逸内核",而不是靠"日志停了"猜 ✗
#
# 为什么不能只看心跳:实测 130k 预填时心跳会停 45/95/125/130 秒而请求**一直在推进** ✓
# ⇒ "心跳停 = 卡死"必然误报 ✗(我据此已至少误判一次 ✗)
#
# 本脚本用**两条**判据:
#   ★ A(铁证,立即告警):num_requests_running=0 且 waiting=0,而 GPU 利用率 >50% 且功耗 >150W
#       ⇒ 零请求还满载烧电 = 逃逸内核 ✓✓ 无需等待、无需推断 ✓
#   ☆ B(可疑,告警):有请求在跑,但 prompt+generation 计数 **连续 >MAXSTALL 分钟不变**,且 GPU 满载
#       ⇒ 可能是极慢的 step,也可能卡死 ⇒ 告警并留档,由人判断 ✓
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
L="$HERE/../dev-docs/report/tuning/logs"
INTERVAL="${INTERVAL:-30}"; MAXSTALL="${MAXSTALL:-600}"     # 秒
OUT="$L/watch_hang.log"
sum(){ curl -s --noproxy 127.0.0.1 --max-time 5 http://127.0.0.1:8070/metrics 2>/dev/null; }
num(){ echo "$1" | grep -E "^vllm:$2\{" | sed 's/.*} //' | head -1; }
gpu(){ nvidia-smi --query-gpu=index,utilization.gpu,power.draw --format=csv,noheader 2>/dev/null; }
# 生产 vLLM 进程的 CPU 时间合计(ticks)⇒ 与上次采样相减 = 实际占用的核数 ✓
# 用途(用户 2026-10-05):与 GPU 一起判断【当前是谁在干 prefill】✓
cpu_ticks(){
  local tot=0 p cl
  for p in /proc/[0-9]*; do
    cl=$(cat "$p/cmdline" 2>/dev/null | tr '\0' ' ')
    case "$cl" in *VLLM::*|*vllm.entrypoints*)
      set -- $(awk '{print $14+$15}' "$p/stat" 2>/dev/null)   # utime+stime
      [ -n "${1:-}" ] && tot=$((tot+$1)) ;;
    esac
  done
  echo "$tot"
}
HZ=$(getconf CLK_TCK 2>/dev/null || echo 100)
log(){ printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" >>"$OUT"; }
log "=== watchdog 启动(INTERVAL=${INTERVAL}s MAXSTALL=${MAXSTALL}s)==="
lastsum=-1; lastchg=$(date +%s); alerted=0
while :; do
  M=$(sum)
  R=$(num "$M" num_requests_running); W=$(num "$M" num_requests_waiting)
  P=$(num "$M" prompt_tokens_total); G=$(num "$M" generation_tokens_total)
  R=${R:-?}; W=${W:-?}; P=${P:-0}; G=${G:-0}
  U=$(gpu | awk -F, '{gsub(/ /,"",$2); if($2+0>mx) mx=$2+0} END{print mx+0}')
  PW=$(gpu | awk -F, '{gsub(/[^0-9.]/,"",$3); if($3+0>mx) mx=$3+0} END{print int(mx)}')
  # CPU 核数(本次采样与前次的差)⇒ 判断谁在干 ✓
  T=$(cpu_ticks); NOWT=$(date +%s)
  if [ "${lastT:-0}" -gt 0 ]; then
    DT=$(( NOWT - lastTs )); [ "$DT" -le 0 ] && DT=1
    CORES=$(awk -v a="$T" -v b="$lastT" -v hz="$HZ" -v dt="$DT" 'BEGIN{printf "%.0f",(a-b)/hz/dt}')
  else CORES="?"; fi
  lastT=$T; lastTs=$NOWT
  # 谁在干活(四象限)✓
  if   [ "${U:-0}" -ge 50 ] && [ "${CORES:-0}" -lt 40 ]; then WHO="GPU 在干 ✓"
  elif [ "${CORES:-0}" -ge 40 ] && [ "${U:-0}" -lt 50 ]; then WHO="CPU 在干 ✓"
  elif [ "${U:-0}" -ge 50 ] && [ "${CORES:-0}" -ge 40 ]; then WHO="GPU+CPU 并行 ✓"
  elif [ "$R" != "0" ]; then WHO="⚠️ 两者都不忙却有请求 ⇒ 可疑"
  else WHO="空闲"; fi
  cur="$P/$G"
  [ "$cur" != "$lastsum" ] && { lastchg=$(date +%s); lastsum="$cur"; }
  stall=$(( $(date +%s) - lastchg ))
  log "running=$R waiting=$W tok=$cur stall=${stall}s | gpu=${U}%/${PW}W cpu=${CORES}cores | $WHO"
  # ★ A 铁证
  if [ "$R" = "0" ] && [ "$W" = "0" ] && [ "${U:-0}" -gt 50 ] && [ "${PW:-0}" -gt 150 ]; then
    log "★★★ 铁证:零请求 + GPU ${U}%/${PW}W ⇒ 逃逸内核!开始留档"
    bash "$HERE/diagnose_hang.sh" >>"$OUT" 2>&1 || true
    log "★★★ 已留档到 $L/hang_*"; alerted=1
  # ☆ B 可疑
  elif [ "$R" != "0" ] && [ "$stall" -gt "$MAXSTALL" ] && [ "${U:-0}" -gt 80 ]; then
    log "☆☆ 可疑:有请求但 ${stall}s 无进展,且 GPU ${U}% ⇒ 可能是超慢 step,也可能卡死 ⇒ 留档"
    bash "$HERE/diagnose_hang.sh" >>"$OUT" 2>&1 || true
  fi
  [ "$alerted" = "1" ] && { log "（已告警,继续观察;重启后请停掉本 watchdog）"; alerted=0; }
  sleep "$INTERVAL"
done
