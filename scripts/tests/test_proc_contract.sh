#!/usr/bin/env bash
# G4 —— `scripts/proc.sh` 的【行为契约】测试(2026-10-07)
#
# 为什么需要它(审计 §2 判定 proc.sh 有 6 处硬伤):
#   ① 门跑在 `kill -0` 之前 ⇒ 陈旧 PID 全 REFUSE(约 180 个无法清理)
#   ② 组杀假定 pid==pgid(实测生产 pgid≠pid)、且杀后不验尸仍打印 stopped
#   ③ `status` 恒 exit 0 ⇒ 消费端 `if ! status` 恒假
#   ④ `spawn` 无条件覆盖活着的同名 PID 文件 + `sleep 1` 竞态 ⇒ 制造孤儿
#   ⑤ `FORCE` 从环境继承 ⇒ 一句 export 静默关掉全部门
#   ⑥ 门只判单 pid、proc.sh 却整组击杀
#
# ⭐ 安全设计(本测试自身绝不能造成事故):
#   * 全程只用【临时 LOGDIR】;入口硬门:若 LOGDIR 落在真日志目录 ⇒ 直接拒绝运行 ✗
#   * 只会杀掉【本测试自己 spawn 的】进程;用一个"杀不死的僵尸"验证 exit 5
#   * 绝不按名字/模式选目标 ✓
#
# 用法: bash scripts/tests/test_proc_contract.sh
# 退出码:0 = 全绿 · 1 = 有用例不符 · 2 = 用法/内部错
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REAL_LOGDIR="$ROOT/dev-docs/report/tuning/logs"
# shellcheck source=../lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"

TMP="$(mktemp -d)" || exit 2
export LOGDIR="$TMP"
PROC=(bash "$ROOT/scripts/proc.sh")

# ⭐ 硬门:绝不允许在真日志目录上做写操作(会把生产的 PID 文件搅乱)✗
case "$(readlink -f "$TMP")/" in
  "$(readlink -f "$REAL_LOGDIR")"/*) echo "⛔ 测试拒绝在真 LOGDIR 上运行" >&2; exit 2 ;;
esac
[ "$TMP" != "$REAL_LOGDIR" ] || { echo "⛔ 测试拒绝在真 LOGDIR 上运行" >&2; exit 2; }

# ⭐ 2026-10-07 独立审计:旁路不得来自环境变量 ⇒ 只认显式 --allow-skip=<理由>
ALLOW_SKIP_REASON=""
for _a in "$@"; do case "$_a" in --allow-skip=*) ALLOW_SKIP_REASON="${_a#--allow-skip=}" ;; esac; done
PASS=0; FAIL=0
chk() {   # chk <描述> <期望> <实得>
  if [ "$2" = "$3" ]; then echo "  ✅ $1  (期望=$2 实得=$3)"; PASS=$((PASS+1))
  else echo "  ❌ $1  (期望=$2 实得=$3)"; FAIL=$((FAIL+1)); fi
}
note() { echo "  ── $*"; }

_cleanup() {
  # 只停本测试自己创建的实例(名字写死在本脚本里,不按模式匹配)✓
  for n in t_sleep t_sleep2 t_stale t_fast; do
    [ -f "$TMP/$n.pid" ] || continue
    # ⭐ G5 要求 stop 调用点必须"留下"退出码(收尾清理,rc 只作记录)
    _o="$("${PROC[@]}" stop "$n" --force --reason=test-cleanup 2>&1)"; : "$_o"
  done
  [ -n "${ZPY:-}" ] && kill -TERM "$ZPY" 2>/dev/null || true
  rm -rf "$TMP"
}
trap _cleanup EXIT

echo "G4 proc.sh 契约测试(临时 LOGDIR=$TMP)"
echo

# ── 1) status 的退出码契约:0=RUNNING / 1=有 pidfile 但已死 / 3=无 pidfile ──────
note "1) status 退出码"
"${PROC[@]}" status nosuch >/dev/null 2>&1; chk "无 pidfile ⇒ 3" 3 "$?"

# ── 2) spawn 正常路径 ────────────────────────────────────────────────────────
note "2) spawn / 重入 / status"
out="$("${PROC[@]}" spawn t_sleep sleep 300 2>&1)"; rc=$?
chk "spawn ⇒ 0" 0 "$rc"
p1="$(cat "$TMP/t_sleep.pid" 2>/dev/null || echo)"
if pi_valid_pid "$p1" && pi_alive "$p1"; then chk "pidfile 里的 pid 活着" 0 0; else chk "pidfile 里的 pid 活着" 0 1; fi

out="$("${PROC[@]}" spawn t_sleep sleep 300 2>&1)"; rc=$?
chk "spawn 重入(活着的同名)⇒ 4" 4 "$rc"
p2="$(cat "$TMP/t_sleep.pid" 2>/dev/null || echo)"
chk "重入时 PID 文件未被覆盖" "$p1" "$p2"

"${PROC[@]}" status t_sleep >/dev/null 2>&1; chk "status(RUNNING)⇒ 0" 0 "$?"

# ── 3) 归属门:非"显式 GPU0 的 vLLM"一律拒绝 ─────────────────────────────────
note "3) 归属门(默认不可杀)"
out="$("${PROC[@]}" stop t_sleep 2>&1)"; rc=$?
chk "stop 非白名单进程 ⇒ 4(拒绝)" 4 "$rc"
if pi_alive "$p1"; then chk "被拒后进程仍活着" 0 0; else chk "被拒后进程仍活着" 0 1; fi

FORCE=1 "${PROC[@]}" stop t_sleep >/dev/null 2>&1; rc=$?
chk "⭐ 环境变量 FORCE=1 不得生效 ⇒ 仍 4" 4 "$rc"
if pi_alive "$p1"; then chk "FORCE=1 被拒后进程仍活着" 0 0; else chk "FORCE=1 被拒后进程仍活着" 0 1; fi

"${PROC[@]}" stop t_sleep --force >/dev/null 2>&1; rc=$?
chk "--force 缺 --reason ⇒ 4" 4 "$rc"

# ── 4) 正常停止 + 验尸 ───────────────────────────────────────────────────────
note "4) --force --reason 正常停止 + 验尸"
out="$("${PROC[@]}" stop t_sleep --force --reason=g4-test 2>&1)"; rc=$?
chk "stop --force --reason ⇒ 0" 0 "$rc"
if pi_alive "$p1"; then chk "停止后进程确已死亡(验尸)" 0 1; else chk "停止后进程确已死亡(验尸)" 0 0; fi

out="$("${PROC[@]}" stop t_sleep 2>&1)"; rc=$?
chk "对死 pid 再 stop ⇒ 0(不再恒 REFUSE)" 0 "$rc"
case "$out" in *"not running"*) chk "死 pid 输出含 not running" 0 0 ;; *) chk "死 pid 输出含 not running" 0 1 ;; esac
"${PROC[@]}" status t_sleep >/dev/null 2>&1; chk "status(已死)⇒ 1" 1 "$?"

# ── 5) 陈旧 PID 文件:spawn 应能清理并重启 ───────────────────────────────────
note "5) 陈旧 PID 文件"
if kill -0 999999 2>/dev/null; then echo "  (跳过:999999 竟然活着)"; else
  echo 999999 > "$TMP/t_stale.pid"
  "${PROC[@]}" spawn t_stale sleep 300 >/dev/null 2>&1; chk "陈旧 pidfile ⇒ spawn 成功(0)" 0 "$?"
  "${PROC[@]}" stop t_stale --force --reason=g4 >/dev/null 2>&1; chk "停掉 t_stale" 0 "$?"
fi

# ── 6) 参数/名字校验 ─────────────────────────────────────────────────────────
note "6) 参数校验(拒绝路径穿越/非法名/非法 pid)"
"${PROC[@]}" spawn '../evil' true >/dev/null 2>&1; chk "非法实例名 '../evil' ⇒ 3" 3 "$?"
"${PROC[@]}" adopt t_x 999999 >/dev/null 2>&1; chk "adopt 不存在的 pid ⇒ 3" 3 "$?"
"${PROC[@]}" adopt t_x "$$" >/dev/null 2>&1; chk "adopt 自身/祖先链 ⇒ 3" 3 "$?"
"${PROC[@]}" status >/dev/null 2>&1; chk "缺参数 ⇒ 2" 2 "$?"
"${PROC[@]}" bogus x >/dev/null 2>&1; chk "未知动作 ⇒ 2" 2 "$?"

# ── 7) 杀后未死 ⇒ 5(用"僵尸"当杀不死的目标)──────────────────────────────────
#    僵尸:kill -0 成功(进程表里还在),但 TERM/KILL 都无效 ⇒ 正好测"验尸 ⇒ exit 5" ✓
note "7) 杀后未死 ⇒ 5(僵尸进程)"
python3 - "$TMP" <<'PY' &
import os, sys, time
d = sys.argv[1]
pid = os.fork()
if pid == 0:
    os._exit(0)                     # 子进程立刻变僵尸(父进程不 wait)
open(os.path.join(d, 'zchild'), 'w').write(str(pid))
open(os.path.join(d, 'zparent'), 'w').write(str(os.getpid()))
time.sleep(120)
PY
ZPY=$!
zpid=""
for _ in $(seq 1 50); do
  [ -s "$TMP/zchild" ] && { zpid="$(cat "$TMP/zchild")"; break; }
  sleep 0.1
done
if pi_valid_pid "$zpid" && [ "$(ps -o stat= -p "$zpid" 2>/dev/null | cut -c1)" = "Z" ]; then
  echo "$zpid" > "$TMP/zt.pid"
  out="$("${PROC[@]}" stop zt --force --reason=g4-zombie 2>&1)"; rc=$?
  chk "僵尸停不掉 ⇒ 5" 5 "$rc"
  case "$out" in *"仍有存活"*) chk "exit 5 时说明仍有存活" 0 0 ;; *) chk "exit 5 时说明仍有存活" 0 1 ;; esac
else
  echo "  ❌ 核心用例未能执行:未能造出僵尸(验尸 rc=5 未被验证)"
  # ⭐ 2026-10-07 独立审计:核心用例"跳过"不得等于通过(除非显式 ALLOW_SKIP=1)✗
  if [ -n "${ALLOW_SKIP_REASON:-}" ]; then :; else FAIL=$((FAIL+1)); fi
fi

# ── 8) 审计留痕 ─────────────────────────────────────────────────────────────
note "8) 审计日志动作名"
alog="$TMP/_launch_audit.log"
for act in SPAWN SPAWN-REFUSE REFUSE FORCE-STOP STOP DEAD; do
  if grep -q " $act " "$alog" 2>/dev/null; then chk "审计含 $act" 0 0; else chk "审计含 $act" 0 1; fi
done

echo
echo "G4 结果:通过 $PASS / 失败 $FAIL"
[ "$FAIL" -eq 0 ] && echo "G4 ✅ 全绿" || echo "G4 ⛔ 不通过"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
