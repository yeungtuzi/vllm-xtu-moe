#!/usr/bin/env bash
# G12 —— 【FORCE 旁路必须留痕】静态门禁(2026-10-07)
#
# 来由(proc.sh v4 审计 ⑤):`FORCE` 曾从环境继承 —— 一句 `export FORCE=1` 就
#   **静默关掉全部门**,而且不写审计 ⇒ 事后无法定案 ✗
#   修法:① 不再读环境变量 `FORCE`(旁路只认显式 `--force --reason=<理由>`)
#         ② 审计函数 `_launch_audit` 必须记录 `FORCE-STOP` 与 `REFUSE` 两个动作 ✓
#
# fail 条件:
#   F1 `scripts/proc.sh` 的【代码】里出现对 `FORCE` 的**变量读取** —— 判据是
#      `\$\{?FORCE\b`(覆盖 `$FORCE` / `${FORCE}` / `${FORCE:-}` / `[[ $FORCE ]]` /
#      `case "$FORCE" in` 等,**不再只认 `[ "${FORCE` 与 `case "${FORCE`**)
#      ⇒ 读 FORCE 做判断/赋值 = 静默旁路,必须改成 `--force --reason=<理由>` ✗
#      ⚠️ 唯一放行:proc.sh 里那条"明确忽略它"的告警行 —— 精确形态 =
#         同一行同时含 `忽略`(或 ignore)与 `--force`,且**不**把 FORCE 变成放行动作
#         (`_force=1`)。实测 proc.sh:154 正是这种告警行,必须放行 ✓
#         (审计 2026-10-07:原判据只认 `[ "${FORCE`/`case "${FORCE` ⇒ 不带花括号的
#          `[ -n "$FORCE" ]`、`case "$FORCE" in`、`[[ $FORCE ]]` 都能绕过 ✗)
#   F2 `_launch_audit` 必须存在,并且 `FORCE-STOP` 与 `REFUSE` 两个动作名必须
#      在 `_launch_audit` 的函数体里,或作为 `_launch_audit <动作>` 的调用点出现
#      (proc.sh 现状:动作名在 stop 分支的调用点 :170/:175,函数体是通用 logger;
#       —— 用户 2026-10-07 已确认该解释可接受 ✓)
#
# ⚠️ proc.sh 缺失/为空 ⇒ 判【内部错 exit 2】,绝不判"通过"(防恒绿)✓
#
# 用法: scripts/gates/check_force_audit.sh [repo_root]
# 退出码:0 = 全绿 · 1 = 有命中 · 2 = 用法/内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G12: 找不到 repo root: $ROOT" >&2; exit 2; }

python3 - "$ROOT" <<'PY'
import os, re, sys

root = sys.argv[1]
PROC = "scripts/proc.sh"
path = os.path.join(root, PROC)
if not os.path.isfile(path):
    print("G12 ⛔ 内部错:找不到 %s" % PROC, file=sys.stderr)
    sys.exit(2)

def strip_comment(line):
    out = []; i = 0; q = None
    while i < len(line):
        c = line[i]
        if q:
            out.append(c)
            if c == q:
                q = None
            i += 1
            continue
        if c in "'\"":
            q = c; out.append(c); i += 1; continue
        if c == "#" and (i == 0 or line[i-1] in " \t\n;&|"):
            break
        out.append(c); i += 1
    return "".join(out)

lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
codes = [strip_comment(l) for l in lines]
text  = "\n".join(codes)

# ⭐ 防恒绿:proc.sh 为空 ⇒ 内部错 ✓
if not [c for c in codes if c.strip()]:
    print("G12 ⛔ 内部错:%s 为空 ⇒ 门会恒绿,拒绝判通过" % PROC, file=sys.stderr)
    sys.exit(2)

# ⭐ 任何对 FORCE 的变量读取都算($FORCE / ${FORCE} / ${FORCE:-} / $FORCE" …)✓
FORCE_READ = re.compile(r"\$\{?FORCE\b")
IGNORE_WORD = re.compile(r"忽略|ignore", re.I)
FORCE_ACTION = re.compile(r"_force\s*=\s*1|_force\s*=\s*\"1\"|\bFORCE\s*=\s*1\b")

viol = []
for i, code in enumerate(codes, 1):
    if not FORCE_READ.search(code):
        continue
    # 唯一放行:精确形态的"忽略它"告警行(同含 忽略 + --force,且不据此放行)✓
    if IGNORE_WORD.search(code) and "--force" in code and not FORCE_ACTION.search(code):
        continue
    viol.append((i, "F1 proc.sh 读环境变量 FORCE 做判断/赋值(静默旁路 ⇒ 必须改成 --force --reason=)",
                 lines[i-1]))

# ── F2:审计动作名 ───────────────────────────────────────────────────────────
start = None
for i, code in enumerate(codes):
    if re.match(r"\s*_launch_audit\s*\(\s*\)\s*\{", code):
        start = i; break

body_txt = ""
if start is None:
    viol.append((0, "F2 找不到 `_launch_audit` 函数定义(审计机制不存在)", ""))
else:
    depth = 0; end = len(codes) - 1
    for j in range(start, len(codes)):
        depth += codes[j].count("{") - codes[j].count("}")
        if depth <= 0:
            end = j; break
    body_txt = "\n".join(codes[start:end+1])

call_fs = bool(re.search(r"_launch_audit\s+FORCE-STOP\b", text))
call_rf = bool(re.search(r"_launch_audit\s+REFUSE\b", text))
body_ok = ("FORCE-STOP" in body_txt) and ("REFUSE" in body_txt)
if start is not None and not body_ok and not (call_fs and call_rf):
    miss = []
    if not (("FORCE-STOP" in body_txt) or call_fs):
        miss.append("FORCE-STOP")
    if not (("REFUSE" in body_txt) or call_rf):
        miss.append("REFUSE")
    viol.append((0, "F2 `_launch_audit` 既没在函数体、也没在调用点记录动作名:%s" % ",".join(miss), ""))

print("G12 检查 %s" % PROC)
if start is not None and not body_ok and (call_fs and call_rf):
    print("   (info) `_launch_audit` 是通用 logger;动作名靠调用点满足:"
          "FORCE-STOP=%s REFUSE=%s" % (call_fs, call_rf))
if viol:
    print("G12 ⛔ 不通过:%d 处" % len(viol))
    for ln, why, txt in viol:
        loc = "%s:%d" % (PROC, ln) if ln else PROC
        print("   %s: %s" % (loc, why))
        if txt.strip():
            print("       %s" % txt.strip())
    print("   ⇒ 修法:删掉对环境变量 FORCE 的判断;旁路只走 `--force --reason=<理由>`,")
    print("           并让 `_launch_audit` 记录 `FORCE-STOP` / `REFUSE` 两个动作名 ✗")
    sys.exit(1)
print("G12 ✅ 通过:proc.sh 不用环境变量 FORCE 做授权;审计含 FORCE-STOP 与 REFUSE")
sys.exit(0)
PY
