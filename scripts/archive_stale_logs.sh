#!/usr/bin/env bash
# 【规则·用户 2026-10-05 定】PID 对应的进程【已消失满 N 天】(默认 3)⇒ 打包归档其 .pid 与同批 .log
#
# 判据说明:"进程何时消失"无法直接观测 ⇒ 用**该实例日志的最后修改时间**作为代理
# (日志 mtime ≈ 它最后一次活动;不再更新的日志即"已死且无人再看")✓
# 永不触碰:①活着进程的 PID 文件与日志 ②_launch_audit.log ③archive/ 本身
#
# 用法:bash scripts/archive_stale_logs.sh          # 执行(按 3 天)
#       DAYS=7 bash scripts/archive_stale_logs.sh  # 改阈值
#       DRY=1 bash scripts/archive_stale_logs.sh   # 只列不删
set -uo pipefail
DAYS="${DAYS:-3}"
DRY="${DRY:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
L="$HERE/../dev-docs/report/tuning/logs"
cd "$L" || exit 0
[ -d archive ] || mkdir -p archive

# ---- 活着的一律保留(含其日志)----
KEEP=" _launch_audit.log "
for f in *.pid; do
  [ -f "$f" ] || continue
  p="$(tr -dc '0-9' <"$f" 2>/dev/null)"
  [ -n "$p" ] || continue
  if kill -0 "$p" 2>/dev/null; then
    n="${f%.pid}"; KEEP="$KEEP$f $n.log $n.mem.pid $n.mem.log "
  fi
done
for f in v41_8070.*.log; do            # 活实例的服务日志
  [ -f "$f" ] || continue
  p="$(echo "$f" | grep -oE '[0-9]{6,}' | head -1)"
  kill -0 "$p" 2>/dev/null && KEEP="$KEEP$f "
done

# ---- 候选:陈旧(按"日志 mtime 或 pid mtime 距今 > DAYS 天")且不活着 ----
: >/tmp/_asl_list.txt
for f in *.pid; do
  [ -f "$f" ] || continue
  case "$KEEP" in *" $f "*) continue;; esac
  n="${f%.pid}"
  newest=0
  for g in "$f" "$n.log" "$n.mem.log"; do
    [ -f "$g" ] || continue
    m=$(stat -c %Y "$g" 2>/dev/null || echo 0); [ "$m" -gt "$newest" ] && newest=$m
  done
  age=$(( ( $(date +%s) - newest ) / 86400 ))
  [ "$age" -ge "$DAYS" ] || continue
  echo "$f" >>/tmp/_asl_list.txt
  for g in "$n.log" "$n.mem.pid" "$n.mem.log" "$n.pid.bak"; do
    [ -f "$g" ] && echo "$g" >>/tmp/_asl_list.txt
  done
done
N=$(wc -l </tmp/_asl_list.txt)
echo "[archive_stale_logs] 阈值 ${DAYS} 天 ⇒ 候选 ${N} 个文件"
if [ "$N" -eq 0 ]; then echo "[archive_stale_logs] 无陈旧件,跳过"; exit 0; fi
if [ "$DRY" != "0" ]; then echo "[archive_stale_logs] DRY=1 ⇒ 仅列出:"; sed 's/^/  /' /tmp/_asl_list.txt; exit 0; fi

TS=$(date +%Y%m%d-%H%M%S); DEST="archive/logs_stale_${TS}.tar.gz"
{ echo "# archived $TS  by archive_stale_logs.sh (DAYS=$DAYS)";
  echo "# kept(live): $KEEP"; echo "# --- archived ---"; cat /tmp/_asl_list.txt; } >/tmp/_asl_manifest.txt
tar czf "$DEST" -T /tmp/_asl_list.txt -C /tmp _asl_manifest.txt 2>/dev/null
IN=$(tar tzf "$DEST" 2>/dev/null | wc -l)
if [ "$IN" -ne "$((N+1))" ] || [ ! -s "$DEST" ]; then
  echo "[archive_stale_logs] ✗ 包内 ${IN} / 期望 $((N+1)) ⇒ 不删除任何原件"; rm -f "$DEST"; exit 1
fi
while IFS= read -r f; do [ -f "$f" ] && rm -f "$f"; done </tmp/_asl_list.txt
echo "[archive_stale_logs] ✅ 归档 ${N} 个 → $DEST(校验 $((N+1)) 通过,原件已删)"
