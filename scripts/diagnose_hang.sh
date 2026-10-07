#!/usr/bin/env bash
# 【2026-10-05】生产卡死时的一键取证(只读,绝不重启任何东西)
# 用法: bash scripts/diagnose_hang.sh          # 输出到终端 + 存到 logs/hang_<时间>/
# 为什么需要它:2026-10-04 14:42:36 那次卡死,两个 worker 处于 State=R / wchan=0(用户态自旋),
# GPU 93–96% / 245–258 W 却 **零 token 产出** ⇒ 若当时有这一条命令,证据链就完整了 ✓
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; L="$HERE/../dev-docs/report/tuning/logs"
GAP="${GAP:-8}"
TS=$(date +%Y%m%d-%H%M%S); OUT="$L/hang_$TS"; mkdir -p "$OUT"
log(){ echo "$@" | tee -a "$OUT/summary.txt"; }

log "=== 卡死取证 $TS ==="
log ""
log "--- 1) 推理是否在推进(两次采样间隔 ${GAP}s,默认 8s)---"
log "    ⚠️ 判据:冻结 <2 分钟【不算卡死】(大预填一个 step 就要数秒,累计计数 step 完成才更新)✓"
for i in 1 2; do
  curl -s --noproxy 127.0.0.1 --max-time 8 http://127.0.0.1:8070/metrics 2>/dev/null \
    | grep -E "^vllm:(num_requests_(running|waiting)|prompt_tokens_total|generation_tokens_total)" \
    | sed 's/^/    /' | tee -a "$OUT/summary.txt"
  [ "$i" = 1 ] && { log "    ---(等待 ${GAP}s)---"; sleep "$GAP"; }
done

log ""
log "--- 1b) ★ 引擎心跳(每 10s 一行)⇒ 【多久没心跳 = 引擎卡了多久】,比 metrics 更可靠 ✓"
S0=$(ls -t "$L"/v41_8070.*.log 2>/dev/null | head -1)
if [ -n "$S0" ]; then
  hb=$(grep -E 'Engine 000: Avg prompt throughput' "$S0" 2>/dev/null | tail -1)
  hts=$(echo "$hb" | grep -oE '[0-9]{2}-[0-9]{2} [0-9:]{8}' | tail -1)
  if [ -n "$hts" ]; then
    hage=$(( $(date +%s) - $(date -d "$(date +%Y)-$hts" +%s 2>/dev/null || date +%s) ))
    log "    最后心跳: $hts   距今 ${hage}s   $([ "$hage" -gt 40 ] && echo '✗ 引擎循环卡住' || echo '✓ 正常')"
    log "    $hb" | cut -c1-170
  fi
fi
log ""
log "--- 2) GPU(满载却零产出 = 自旋的指纹)---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader \
  | sed 's/^/    /' | tee -a "$OUT/gpu.txt" | sed 's/^/  /'

log ""
log "--- 3) 进程状态(关键:wchan=0 + State=R ⇒ 用户态自旋)---"
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cl=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')   # 静默:进程可能刚退出 ✓
  [ -n "$cl" ] || continue
  case "$cl" in *VLLM::*|*vllm.entrypoints*)
    printf "    pid=%-9s %-18s %s\n" "$pid" "$(grep -m1 '^State:' /proc/$pid/status 2>/dev/null | cut -f2- | tr -d '\t')" \
      "$(echo "$cl" | grep -oE 'VLLM::[A-Za-z_0-9]+' | head -1)" ;;
  esac
done | tee -a "$OUT/summary.txt" | sed 's/^/  /'

log ""
log "--- 4) ★ 有没有【别的实例】在抢 GPU/显存(用户 2026-10-05 的怀疑点)---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
  | sed 's/^/    /' | tee -a "$OUT/compute_apps.txt" | sed 's/^/  /'
log "    (上面若出现非 8070 那 4 个 pid 的进程 ⇒ 就是【第二个实例在抢资源】✗)"
PROD=$(tr -dc '0-9' < "$L/v41_8070.pid" 2>/dev/null)
LMC=$(tr -dc '0-9' < "$L/lmcache_server.pid" 2>/dev/null)
# 沿 PPid 向上追溯,判断某 pid 是否属于【生产那棵树】或【LMCache】(两者都合法用 GPU ✓)
in_tree(){ local x="$1" hop=0; while [ -n "$x" ] && [ "$x" != "0" ] && [ "$x" != "1" ] && [ $hop -lt 12 ]; do
    [ "$x" = "$PROD" ] && return 0; [ "$x" = "$LMC" ] && return 0
    x=$(grep -m1 '^PPid:' /proc/$x/status 2>/dev/null | tr -dc '0-9'); hop=$((hop+1)); done; return 1; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | while read ap; do
  [ -z "$ap" ] && continue
  if ! in_tree "$ap"; then
    log "    ⚠️ 非生产进程 $ap 正占用 GPU ⇒ 竞争资源 ✗"
    log "       命令行: $(cat /proc/$ap/cmdline 2>/dev/null | tr '\0' ' ' | cut -c1-120)"
  fi
done
log ""
log "--- 5) 日志最后一次非 metrics 活动(= 卡死起点)---"
S=$(ls -t "$L"/v41_8070.*.log 2>/dev/null | head -1)
log "    $(basename "${S:-无}")  mtime=$(stat -c %y "$S" 2>/dev/null | cut -c1-19)"
[ -n "$S" ] && grep -vE '"GET /metrics HTTP/1.1" 200|"GET /v1/models HTTP/1.1" 200' "$S" 2>/dev/null \
  | tail -12 | sed 's/^/    /' | tee -a "$OUT/log_tail.txt" >/dev/null

log ""
log "--- 5b) ⭐ LMCache 现场(它的日志【每次启动被覆盖】⇒ 必须当场抓,否则证据永久丢失 ✗)"
LML="$L/lmcache_server.log"
if [ -f "$LML" ]; then
  log "    大小=$(du -h "$LML" | cut -f1) 最后写入=$(stat -c %y "$LML" | cut -c1-19)"
  log "    最近动作(去 ANSI):"
  grep -a "Stored\|Retrieved\|ERROR\|timeout" "$LML" 2>/dev/null | tail -8 \
    | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-150 | sed 's/^/      /' | tee -a "$OUT/lmcache.txt" >/dev/null
  log "    Stored 频次分布(多份=同一块反复搬运 ⇒ 传输层可能在重试 ✗):"
  grep -aoE "Stored [0-9]+ tokens" "$LML" 2>/dev/null | sort | uniq -c | sort -rn | head -5 \
    | sed 's/^/      /' | tee -a "$OUT/lmcache.txt" >/dev/null
else
  log "    ✗ 找不到 $LML"
fi
log ""
log "--- 6) Python 栈(若装了 py-spy 就能直接看到卡在哪一行)---"
if command -v py-spy >/dev/null 2>&1; then
  for pid in $(ls /proc | grep -E '^[0-9]+$'); do
    cl=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')   # 静默:进程可能刚退出 ✓
  [ -n "$cl" ] || continue
    case "$cl" in *VLLM::Worker*) log "    --- py-spy dump $pid ---"
      py-spy dump --pid "$pid" 2>&1 | head -40 | sed 's/^/      /' | tee -a "$OUT/pyspy_$pid.txt" >/dev/null ;;
    esac
  done
else
  log "    ✗ 未安装 py-spy ⇒ 只能看内核态(wchan);安装后可直接抓到 Python 行号:"
  log "      HTTPS_PROXY=<proxy> pip install py-spy    # 独立二进制,不影响已加载的模块"
fi
log ""
log "=== 取证完成 ⇒ $OUT ==="
