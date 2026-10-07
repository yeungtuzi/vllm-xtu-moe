#!/usr/bin/env bash
# 【2026-10-05】密集采样 —— 用户触发真实负载时同步抓现场(1 秒级)
#
# 为什么需要它:看门狗 30 秒采样 ✗ 会漏掉 prefill 决定性序列;而 LMCache 日志**每次启动被覆盖** ✗
# ⇒ 必须在【故障发生的那一刻】把这几样都记下来 ✓
#   ① 引擎进度心跳 [xtu-pf-progress](带 device/layer/qlen)⇒ 卡死时看**最后一条停在哪一层** ✓✓
#   ② 指标 running/waiting/tokens ⇒ 是否推进 ✓
#   ③ GPU 利用率/功耗 + 引擎进程 CPU 核数 ⇒ "谁在干活" ✓
#   ④ LMCache 的 store/retrieve/错误行 ✓
#   ⑤ 一旦命中铁证(running=0 且 waiting=0 且 GPU>50% 且功耗>150W)⇒ 立即调 diagnose_hang.sh 全量留档 ✓
#
# 用法: INTERVAL=1 DURATION=2400 bash scripts/dense_sample.sh      # 1 秒一次,最多 40 分钟
#       bash scripts/proc.sh spawn dense_sample env INTERVAL=1 DURATION=2400 bash scripts/dense_sample.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; L="$HERE/../dev-docs/report/tuning/logs"
INTERVAL="${INTERVAL:-1}"; DURATION="${DURATION:-2400}"
TS=$(date +%Y%m%d-%H%M%S); OUT="$L/dense_$TS"; mkdir -p "$OUT"
CSV="$OUT/samples.csv"; PFLOG="$OUT/pf_progress.txt"
echo "t,epoch,running,waiting,prompt_tok,gen_tok,gpu_max,power_max,cpu_cores,verdict,kv_usage_pct,avail_gib,lmc_stored" >"$CSV"
SLOG=$(ls -t "$L"/v41_8070.*.log 2>/dev/null | head -1)
echo "=== dense sample start $TS  interval=${INTERVAL}s duration=${DURATION}s" | tee -a "$OUT/run.log"
echo "    服务日志: $(basename "${SLOG:-无}")" | tee -a "$OUT/run.log"
HZ=$(getconf CLK_TCK 2>/dev/null || echo 100)
cpu_ticks(){ local tot=0 p cl; for p in /proc/[0-9]*; do
    cl=$(cat "$p/cmdline" 2>/dev/null | tr '\0' ' '); [ -n "$cl" ] || continue
    case "$cl" in *VLLM::*|*vllm.entrypoints*) set -- $(awk '{print $14+$15}' "$p/stat" 2>/dev/null)
      [ -n "${1:-}" ] && tot=$((tot+$1));; esac; done; echo "$tot"; }
lastT=0; lastTs=0; pfseen=0; end=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$end" ]; do
  M=$(curl -s --noproxy 127.0.0.1 --max-time 3 http://127.0.0.1:8070/metrics 2>/dev/null)
  g(){ echo "$M" | grep -E "^vllm:$1\{" | sed 's/.*} //' | head -1; }
  R=$(g num_requests_running); W=$(g num_requests_waiting)
  R=${R:-}; W=${W:-}; P=$(g prompt_tokens_total); G=$(g generation_tokens_total)
  Rn=${R%%.*}; Wn=${W%%.*}; [ -z "$Rn" ] && Rn=-1; [ -z "$Wn" ] && Wn=-1
  GPU=$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader 2>/dev/null)
  U=$(echo "$GPU" | awk -F, '{gsub(/ /,"",$1); if($1+0>m)m=$1+0} END{print m+0}')
  PW=$(echo "$GPU" | awk -F, '{gsub(/[^0-9.]/,"",$2); if($2+0>m)m=$2+0} END{print int(m)}')
  T=$(cpu_ticks); NOW=$(date +%s)
  if [ "$lastT" -gt 0 ]; then DT=$((NOW-lastTs)); [ $DT -le 0 ] && DT=1
    CORES=$(awk -v a="$T" -v b="$lastT" -v hz="$HZ" -v dt="$DT" 'BEGIN{printf "%.0f",(a-b)/hz/dt}')
  else CORES=0; fi
  lastT=$T; lastTs=$NOW
  if   [ "${U:-0}" -ge 50 ] && [ "${CORES:-0}" -lt 40 ]; then V="GPU"
  elif [ "${CORES:-0}" -ge 40 ] && [ "${U:-0}" -lt 50 ]; then V="CPU"
  elif [ "${U:-0}" -ge 50 ] && [ "${CORES:-0}" -ge 40 ]; then V="GPU+CPU"
  elif [ "${Rn:-0}" -gt 0 ] && [ "${U:-0}" -lt 15 ] && [ "${CORES:-0}" -lt 10 ]; then V="!!STALL"
  elif [ "${Rn:-0}" -gt 0 ]; then V="busy"
  else V="idle"; fi
  # 抓新的 [xtu-pf-progress](它们带 device/layer/qlen ⇒ 卡死时最关键 ✓)
  if [ -n "$SLOG" ]; then
    cur=$(grep -c "xtu-pf-progress" "$SLOG" 2>/dev/null || echo 0)
    if [ "$cur" -gt "$pfseen" ]; then
      grep "xtu-pf-progress" "$SLOG" 2>/dev/null | tail -n $((cur-pfseen)) \
        | sed 's/\x1b\[[0-9;]*m//g' >>"$PFLOG"
      pfseen=$cur
    fi
  fi
  KV=$(echo "$M" | grep -E "^vllm:gpu_cache_usage_perc" | sed 's/.*} //' | head -1)
  AV=$(awk '/^MemAvailable:/{printf "%.0f",$2/1048576}' /proc/meminfo)
  LS=$(grep -ac "Stored" "$L/lmcache_server.log" 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S),$NOW,$R,$W,${P:-},${G:-},${U:-0},${PW:-0},${CORES:-0},$V,${KV:-},${AV:-},${LS:-}" >>"$CSV"
  # ★ 铁证 ⇒ 立刻全量留档
  if [ "${Rn:-0}" -eq 0 ] && [ "${Wn:-0}" -eq 0 ] && [ "${U:-0}" -gt 50 ] && [ "${PW:-0}" -gt 150 ]; then
    echo "★★★ $(date +%H:%M:%S) 铁证:零请求但 GPU ${U}%/${PW}W ⇒ 留档" | tee -a "$OUT/run.log"
    cp -f "$CSV" "$OUT/samples_at_alert.csv" 2>/dev/null
    bash "$HERE/diagnose_hang.sh" >>"$OUT/run.log" 2>&1 || true
    echo "★★★ 已留档;继续采样以便看恢复过程" | tee -a "$OUT/run.log"
  fi
  sleep "$INTERVAL"
done
echo "=== dense sample end $(date +%H:%M:%S) ⇒ $OUT ===" | tee -a "$OUT/run.log"
