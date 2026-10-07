#!/usr/bin/env bash
# run_all.sh —— 【机械验收】一条命令跑完所有门禁与行为测试(2026-10-07)
#
# 对应交接文件的验收判据:
#   · G1..G13 全绿(至少 G1/G2/G3/G4/G5/G7)
#   · 行为测试 G3(门矩阵) G4(proc 契约) G9(wait_ready) G10(grep 消费端)
#   · `grep -rn 'scripts\.proc\.sh'` 为空
#   · 门在"生产/生产端口/生产 PID 文件/自身或祖先/读不到 CV"下一律 rc≠0,
#     仅在"显式 CUDA_VISIBLE_DEVICES=0 的调试 vLLM"下 rc=0
#
# ⭐ 所有门禁/测试都必须【只读或只动自己的临时目录】;本脚本自身不启动、不杀任何生产进程 ✓
#
# 用法: bash scripts/tests/run_all.sh [--list]
# 退出码:0 = 全绿 · 1 = 有红 · 2 = 内部错
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GATES=(
  "$ROOT/scripts/gates/check_no_pattern_kill.sh"        # G1
  "$ROOT/scripts/gates/check_gate_selfconsistency.sh"   # G2
  "$ROOT/scripts/gates/check_stop_callsites.sh"         # G5
  "$ROOT/scripts/gates/check_spelling_and_syntax.sh"    # G6
  "$ROOT/scripts/gates/check_instance_names.sh"         # G7
  "$ROOT/scripts/gates/check_no_bare_launch.sh"         # G8
  "$ROOT/scripts/gates/check_prod_identity.sh"          # G11
  "$ROOT/scripts/gates/check_force_audit.sh"            # G12
  "$ROOT/scripts/gates/check_no_status_grep.sh"         # G13
)
TESTS=(
  "$ROOT/scripts/tests/test_g1_pattern_kill.sh"
  "$ROOT/scripts/tests/test_whoami_gate.sh"
  "$ROOT/scripts/tests/test_proc_contract.sh"
  "$ROOT/scripts/tests/test_wait_ready.sh"
  "$ROOT/scripts/tests/test_grep_consumers.sh"
)

if [ "${1:-}" = "--list" ]; then
  printf 'GATES:\n'; printf '  %s\n' "${GATES[@]}"
  printf 'TESTS:\n'; printf '  %s\n' "${TESTS[@]}"
  exit 0
fi

# ⭐ 2026-10-07 独立审计:拒绝环境变量旁路(核心断言不得被 env 静默降级)✗
if [ -n "${ALLOW_SKIP:-}" ]; then
  echo "⛔ 拒绝运行:环境变量 ALLOW_SKIP 不被接受(旁路必须显式:给测试传 --allow-skip=<理由>)" >&2
  exit 2
fi
unset ALLOW_SKIP
PASS=0; FAIL=0; MISSING=0
declare -a RED=()

_run_one() {   # <kind> <path>
  local kind="$1" f="$2" rc
  local name; name="$(basename "$f")"
  if [ ! -f "$f" ]; then
    printf '  ⏭  %-10s %-38s (未实现)\n' "$kind" "$name"; MISSING=$((MISSING+1)); return
  fi
  # ⭐ 2026-10-07 独立审计:每个测试加超时,避免任一测试挂起拖死整条验收 ✓
  timeout "${RUNALL_TIMEOUT:-300}" bash "$f" >/tmp/runall."$name".out 2>&1; rc=$?
  if [ "$rc" -eq 0 ]; then
    printf '  ✅ %-10s %-38s rc=0\n' "$kind" "$name"; PASS=$((PASS+1))
  else
    printf '  ❌ %-10s %-38s rc=%s  (详见 /tmp/runall.%s.out)\n' "$kind" "$name" "$rc" "$name"
    FAIL=$((FAIL+1)); RED+=("$name rc=$rc")
  fi
}

echo "=== 静态门禁 ==="
for f in "${GATES[@]}"; do _run_one "GATE" "$f"; done
echo
echo "=== 行为测试 ==="
for f in "${TESTS[@]}"; do _run_one "TEST" "$f"; done
echo
echo "=== 附加机械判据 ==="
if grep -rn 'scripts\.proc\.sh' --include=*.sh "$ROOT" 2>/dev/null | grep -q .; then
  echo "  ❌ 仍存在「scripts+点号+proc.sh」式笔误"; FAIL=$((FAIL+1)); RED+=("scripts 与 proc.sh 之间误写点号")
else
  echo "  ✅ 无「scripts+点号+proc.sh」式笔误"; PASS=$((PASS+1))
fi

echo
echo "=== 汇总 ==="
echo "  通过=$PASS 失败=$FAIL 未实现=$MISSING"
if [ "${#RED[@]}" -gt 0 ]; then
  echo "  红的项:"; printf '    - %s\n' "${RED[@]}"
fi
if [ "$FAIL" -eq 0 ] && [ "$MISSING" -eq 0 ]; then
  echo "  ✅ 全绿(可以进入独立审计)"
  exit 0
fi
echo "  ⛔ 未全绿 ⇒ 未过验收"
exit 1
