#!/usr/bin/env bash
# G10 —— `proc.sh status` 的【消费端决策表】测试(2026-10-07)
#
# 为什么需要它(审计 §2 的结构性矛盾):
#   ⛔ `proc.sh status` 原先**恒 exit 0** ⇒ 三个消费端的 `if ! status` **恒假**
#   ⛔ 而消费端当时用 `status | grep -q "NOT RUNNING"` 这种**子串匹配**
#      ⇒ `"NOT RUNNING"` 里也含 `"RUNNING"` ⇒ 两类判据互相掩盖 ✗
#
# 本测试分两部分:
#   Part 1【契约】用临时 LOGDIR 实测 `status` 在三种状态下**必须**返回 0/1/3;
#   Part 2【消费端符合性】扫描全仓 `proc.sh status` 调用点,断言:
#           ① 不得把 `status` 与 `grep` 混进同一管道(子串匹配 ✗)
#           ② 不得用字面量 `NOT RUNNING` 做条件(去注释后)
#           ③ 必须靠【退出码】或【PID 文件 + kill -0】决策
#   Part 3 打印决策表(把"契约"写成人能读的东西)
#
# ⭐ 安全:Part 1 只用临时 LOGDIR 与自建进程;Part 2/3 纯静态 ✓
#
# 用法: bash scripts/tests/test_grep_consumers.sh
# 退出码:0 = 全绿 · 1 = 有用例不符 · 2 = 内部错
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REAL_LOGDIR="$ROOT/dev-docs/report/tuning/logs"
# shellcheck source=../lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"
PROC=(bash "$ROOT/scripts/proc.sh")

TMP="$(mktemp -d)"
export LOGDIR="$TMP"
case "$(readlink -f "$TMP")/" in
  "$(readlink -f "$REAL_LOGDIR")"/*) echo "⛔ 拒绝在真 LOGDIR 上运行" >&2; exit 2 ;;
esac
_cleanup() { _o="$("${PROC[@]}" stop g10inst --force --reason=g10 2>&1)"; : "$_o"; rm -rf "$TMP"; }
trap _cleanup EXIT

PASS=0; FAIL=0; INFO=0
chk() {
  if [ "$2" = "$3" ]; then echo "  ✅ $1  (期望=$2 实得=$3)"; PASS=$((PASS+1))
  else echo "  ❌ $1  (期望=$2 实得=$3)"; FAIL=$((FAIL+1)); fi
}

echo "G10 proc.sh status 消费端决策表"
echo
echo "── Part 1:status 的退出码契约(行为实测,临时 LOGDIR=$TMP)──"

"${PROC[@]}" status g10inst >/dev/null 2>&1; chk "无 PID 文件 ⇒ 3" 3 "$?"

"${PROC[@]}" spawn g10inst sleep 300 >/dev/null 2>&1 || true
GP="$(cat "$TMP/g10inst.pid" 2>/dev/null || true)"
if pi_valid_pid "$GP" && pi_alive "$GP"; then
  "${PROC[@]}" status g10inst >/dev/null 2>&1; chk "RUNNING ⇒ 0" 0 "$?"
  "${PROC[@]}" stop g10inst --force --reason=g10 >/dev/null 2>&1
  "${PROC[@]}" status g10inst >/dev/null 2>&1; chk "有 PID 文件但进程已死 ⇒ 1" 1 "$?"
else
  echo "  ❌ 无法起 g10inst 探针进程"; FAIL=$((FAIL+1))
fi

echo
echo "── Part 2:全仓 proc.sh status 调用点的符合性(静态)──"
# 收集调用点(排除 proc.sh 自身 = 定义处)
mapfile -t CALLSITES < <(grep -rlE 'proc\.sh"?[[:space:]]+status' --include=*.sh "$ROOT" 2>/dev/null \
                          | grep -v '/scripts/proc.sh$' | grep -v '/scripts/tests/' | sort)
[ "${#CALLSITES[@]}" -gt 0 ] || { echo "  ❌ 一个消费端都没找到(扫描有问题?)"; FAIL=$((FAIL+1)); }

for f in "${CALLSITES[@]}"; do
  rel="${f#"$ROOT"/}"
  # 其它任务线的文件:豁免(与 G5 的 allow 名单同源理由)✓
  case "$rel" in
  esac
  # 去注释后再判(注释里提到 grep "NOT RUNNING" 是允许的 ✓)
  code="$(grep -vE '^[[:space:]]*#' "$f" 2>/dev/null || true)"
  bad=0
  # ① status 与 grep 混管道(子串匹配 ✗)
  if printf '%s\n' "$code" | grep -qE 'proc\.sh"?[[:space:]]+status[^|]*\|[^|]*grep'; then
    echo "     ⛔ ① status 与 grep 混管道"; bad=1
  fi
  # ② 用字面量 NOT RUNNING 做条件
  if printf '%s\n' "$code" | grep -q 'NOT RUNNING'; then
    echo "     ⛔ ② 用字面量 'NOT RUNNING' 做条件(子串匹配)"; bad=1
  fi
  # ③ 必须靠退出码 / PID 文件 + kill -0
  if ! printf '%s\n' "$code" | grep -qE '(^|[^!])!\s*.*proc\.sh"?[[:space:]]+status|status[^|]*>[[:space:]]*/dev/null[^|]*2>&1|kill -0|pi_alive'; then
    echo "     ⛔ ③ 没看到「靠退出码」或「PID 文件 + kill -0」的决策"; bad=1
  fi
  if [ "$bad" = "0" ]; then echo "  ✅ $rel"; PASS=$((PASS+1)); else echo "  ❌ $rel(见上)"; FAIL=$((FAIL+1)); fi
done

echo
echo "── Part 3:决策表(契约文档化)──"
cat <<'TABLE'
     status 退出码   含义                    消费端正确决策
     --------------  ----------------------  ------------------------------------------
     0               RUNNING                 继续等待/继续流程
     1               有 PID 文件但进程已死   判"启动失败"⇒ 打印日志尾部 + 退出
     3               没有 PID 文件           判"未受管/未启动"⇒ 不可当 RUNNING;要就绪判据
     其它(rc≠0)     异常                    一律按"未就绪/失败"处理,禁止放行
     禁止            任何形式的 `status | grep` / 字面量 "NOT RUNNING" ✗
TABLE

echo
echo "G10 结果:通过 $PASS / 失败 $FAIL / 豁免 $INFO"
[ "$FAIL" -eq 0 ] && echo "G10 ✅ 全绿" || echo "G10 ⛔ 不通过"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
