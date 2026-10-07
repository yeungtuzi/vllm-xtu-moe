#!/usr/bin/env bash
# 对照组比较(只读 ✓):本地 ds41f 的会话 vs 官方 API 的会话,同口径 ✓
set -u
R=/home/user/lvllm/vllm-xiaotu-moe
printf "%-14s %-6s %-7s %-9s %-9s %-9s %-7s %s\n" 组 轮 调用 meta前 meta全 meta后 损坏 "损坏%"
for spec in "$@"; do
  sid="${spec%%=*}"; lab="${spec#*=}"
  [ "$lab" = "$spec" ] && lab="${sid:8:10}"
  row=$(python3 "$R/scripts/_sess_metrics.py" "$sid" "$lab")
  echo "$row" | awk -F'\t' '{printf "%-14s %-6s %-7s %-9s %-9s %-9s %-7s %s\n",$1,$2,$3,$4,$5,$6,$7,$8}'
done
