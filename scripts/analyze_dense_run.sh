#!/usr/bin/env bash
# 【2026-10-05】等采样结束后自动出分析报告(不依赖 agent 在线 ✓)
# 用法: bash scripts/analyze_dense_run.sh [采样目录] [额外等待秒数]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; L="$HERE/../dev-docs/report/tuning/logs"
# ⚠️ 必须【只匹配目录】:dense_* 会命中 dense_analysis.log / dense_sample.log ✗(已踩过 ✓)
if [ -n "${1:-}" ] && [ -d "${1:-}" ]; then D="$1"
else D="$(find "$L" -maxdepth 1 -type d -name 'dense_20*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d" " -f2-)"; fi
WAIT="${2:-120}"
[ -d "$D" ] || { echo "找不到采样目录 [$D]  (L=$L)"; ls -d "$L"/dense_20* 2>/dev/null | head -3; exit 1; }
# 1) 等采样器自己结束(DURATION 到点)+ 宽限
if [ "${SKIP_WAIT:-0}" = "1" ]; then
  echo "[analyze] SKIP_WAIT=1 ⇒ 跳过等待,直接分析现有数据(用于自测 ✓)"
else
  for i in $(seq 1 90); do
    # ⭐ 2026-10-07 审计修:原为 `status | grep -q "NOT RUNNING"` 子串匹配 ✗
    #   ("NOT RUNNING" 里含 "RUNNING",且 status 曾恒 exit 0 ⇒ 判据互相掩盖)
    #   ⇒ 改用【退出码】:0=RUNNING / 1=已死 / 3=无 pidfile ⇒ 非 0 即结束 ✓
    if ! bash "$HERE/proc.sh" status dense_sample >/dev/null 2>&1; then break; fi
    sleep 60
  done
fi
sleep "$WAIT"
python3 "$HERE/analyze_dense_run.py" "$D"
