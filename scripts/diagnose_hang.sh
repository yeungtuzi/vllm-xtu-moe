#!/usr/bin/env bash
# 【2026-10-05】生产卡死时的一键取证(只读,绝不重启任何东西)
# 用法: bash scripts/diagnose_hang.sh          # 输出到终端 + 存到 logs/hang_<时间>/
# 为什么需要它:2026-10-04 14:42:36 那次卡死,两个 worker 处于 State=R / wchan=0(用户态自旋),
# GPU 93–96% / 245–258 W 却 **零 token 产出** ⇒ 若当时有这一条命令,证据链就完整了 ✓
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; L="$HERE/../dev-docs/report/tuning/logs"
TS=$(date +%Y%m%d-%H%M%S); OUT="$L/hang_$TS"; mkdir -p "$OUT"
log(){ echo "$@" | tee -a "$OUT/summary.txt"; }

log "=== 卡死取证 $TS ==="
log ""
log "--- 1) 推理是否在推进(两次采样,间隔 8s)---"
for i in 1 2; do
  curl -s --noproxy 127.0.0.1 --max-time 8 http://127.0.0.1:8070/metrics 2>/dev/null \
    | grep -E "^vllm:(num_requests_(running|waiting)|prompt_tokens_total|generation_tokens_total)" \
    | sed 's/^/    /' | tee -a "$OUT/summary.txt"
  [ "$i" = 1 ] && { log "    ---(8s)---"; sleep 8; }
done

log ""
log "--- 2) GPU(满载却零产出 = 自旋的指纹)---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader \
  | sed 's/^/    /' | tee -a "$OUT/gpu.txt" | sed 's/^/  /'

log ""
log "--- 3) 进程状态(关键:wchan=0 + State=R ⇒ 用户态自旋)---"
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  cl=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
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
for p in $(seq 8090 8099); do ss -ltn 2>/dev/null | grep -q ":$p " && log "    ⚠️ 端口 $p 有实例在监听"; done

log ""
log "--- 5) 日志最后一次非 metrics 活动(= 卡死起点)---"
S=$(ls -t "$L"/v41_8070.*.log 2>/dev/null | head -1)
log "    $(basename "${S:-无}")  mtime=$(stat -c %y "$S" 2>/dev/null | cut -c1-19)"
[ -n "$S" ] && grep -vE '"GET /metrics HTTP/1.1" 200|"GET /v1/models HTTP/1.1" 200' "$S" 2>/dev/null \
  | tail -12 | sed 's/^/    /' | tee -a "$OUT/log_tail.txt" >/dev/null

log ""
log "--- 6) Python 栈(若装了 py-spy 就能直接看到卡在哪一行)---"
if command -v py-spy >/dev/null 2>&1; then
  for pid in $(ls /proc | grep -E '^[0-9]+$'); do
    cl=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
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
