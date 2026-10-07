#!/usr/bin/env bash
# G13 —— 【禁止用 status 子串匹配当判据】静态门禁(2026-10-07)
#
# 来由(proc.sh v4 审计 ③):`status` 曾恒 exit 0 ⇒ 消费端 `if ! status` 恒假;
#   为了绕过它,调用方普遍写成
#       proc.sh status X | grep -q "NOT RUNNING"
#   ⇒ 子串匹配**同时**掩盖了退出码,而且把"英文输出格式"变成了 API:
#     改一个词(如 `NOT RUNNING` → `not running`)就静默失明 ✗✓
#
# fail 条件:
#   S1 同一【逻辑单元】里 `proc.sh … status …` 被管道接到 `grep`
#      (例:`bash scripts/proc.sh status dense_sample 2>/dev/null | grep -q "NOT RUNNING"`)
#
# ⚠️ 用户指定的第二条("用日志字符串做就绪判据",如 `tail … | grep -q …` 紧跟 `exit 0`)
#    **本次不做硬断言**(易误伤"读日志确认"的正常用法);本门只做上面这一条硬断言 ✓
#
# 正确写法:`status` 的退出码自带语义 —— 0=RUNNING · 1=PID 文件在但进程已死 · 3=无 pidfile
#           (见 scripts/proc.sh 的 status 分支)⇒ 直接 `if ! bash scripts/proc.sh status X; then`
#
# 用法: scripts/gates/check_no_status_grep.sh [repo_root]
# 退出码:0 = 全绿 · 1 = 有命中 · 2 = 用法/内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G13: 找不到 repo root: $ROOT" >&2; exit 2; }

python3 - "$ROOT" <<'PY'
import os, re, subprocess, sys

root = sys.argv[1]

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

def logical_units(lines):
    res = []; cur = None
    for idx, raw in enumerate(lines, 1):
        code = strip_comment(raw)
        if cur is None:
            cur = [idx, raw, code]
        else:
            cur[1] += "\n" + raw
            cur[2] += "\n" + code
        if code.rstrip().endswith("\\"):
            continue
        res.append(tuple(cur)); cur = None
    if cur:
        res.append(tuple(cur))
    return res

# 模式串拆开写,防自命中:`proc.sh` → `proc` + 点 + `sh`
PIPE_GREP = re.compile(r"proc" + r"\.sh" + r"[^\n|]*\bstatus\b[^\n|]*\|[^\n|]*\bgrep\b")

viol = []; scanned = 0
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__")]
    for f in sorted(fn):
        if not f.endswith(".sh"):
            continue
        scanned += 1
        rel = os.path.relpath(os.path.join(dp, f), root)
        try:
            lines = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        for start, rawunit, code in logical_units(lines):
            if PIPE_GREP.search(code):
                viol.append((rel, start, rawunit))

# ⭐ 防恒绿:扫到 0 个文件(或少于 git tracked)⇒ 内部错,绝不判通过 ✓
if scanned == 0:
    print("G13 ⛔ 内部错:扫描到 0 个 *.sh ⇒ 门会恒绿,拒绝判通过", file=sys.stderr)
    sys.exit(2)
try:
    n_tr = len([x for x in subprocess.run(["git", "-C", root, "ls-files", "-z", "--", "*.sh"],
                                          capture_output=True).stdout.split(b"\0") if x])
except Exception:
    n_tr = -1
if n_tr > 0 and scanned < n_tr:
    print("G13 ⛔ 内部错:scanned=%d < git tracked=%d ⇒ 有漏扫,拒绝判通过" % (scanned, n_tr),
          file=sys.stderr)
    sys.exit(2)

print("G13 扫描 %d 个 *.sh(git tracked=%s)" % (scanned, ("?" if n_tr < 0 else n_tr)))
if viol:
    print("G13 ⛔ 不通过:%d 处把 `proc.sh status` 与 grep 混进同一管道" % len(viol))
    for rel, ln, txt in viol:
        print("   %s:%d: status 子串匹配(退出码被管道掩盖,输出格式被当 API)" % (rel, ln))
        print("       %s" % txt.split("\n")[0].strip())
    print("   ⇒ 修法:用退出码(0=RUNNING / 1=死 / 3=无 pidfile),不要子串匹配")
    print('      ✅ `if ! bash scripts/proc.sh status "$n"; then echo "未在运行"; fi`')
    print("   (说明:`tail … | grep -q …` 当就绪判据的第二条本次不做硬断言)")
    sys.exit(1)
print("G13 ✅ 通过:没有把 `proc.sh status` 接到 grep 的管道")
sys.exit(0)
PY
