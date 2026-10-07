#!/usr/bin/env bash
# G2 —— 【归属门自洽性】静态门禁(2026-10-07)
#
# 来由:上一个会话为"停进程前判归属"连错三版,根因全是【同一事实存在两处表示、
#       而两处从不交叉校验】:
#   ① 生产名单被抄进 runner ⇒ 与唯一真源漂移(证据:`v41_8070` 漏一个就误判)
#   ② `KILLABLE=1` 被放在 CV 判据【之前】的分支里 ⇒ 门恒放行(杀生产的洞 ✗✗)
#   ③ 比较里手打中文字面量 ⇒ 赋值 `"✓ 我的调试实例"` / 比较 `"✓ 我自己的调试实例"`
#      两处不一致 ⇒ 门恒拒(证据:whoami_proc v2 的 :67 与 :75)
#   ⇒ 本门把这三条"自洽性"变成机械断言 ✓
#
# fail 条件(任一命中即 fail):
#   A1 `PROD_PORTS=` / `PROD_PIDFILES_RE=`(带词边界的【定义】)出现在
#      `scripts/lib_proc_identity.sh` 之外的任何 `*.sh` 里(⇒ 有人在抄名单)
#   A2 `scripts/whoami_proc.sh` 里 `KILLABLE=1` 必须【恰好一次】,且必须出现在
#      `[ "$CV_RAW" = "0" ]` 这个比较【之后】(行号比较;锚点缺失也算 fail)
#   A3 `scripts/whoami_proc.sh` 里不得对【含中文的字符串字面量】做 `=`/`==`/`!=` 比较
#      (中文只允许出现在 echo/printf/注释/变量定义里 —— 比较必须比变量 ✓)
#
# ⚠️ 本门 A1 只认【赋值语句】(`NAME=` / `export NAME=` / `local NAME=`,行首或 `;`/空白之前):
#    * `PI_PROD_PORTS=` 这类前缀变量不算重复定义 ✓
#    * `grep -c 'PROD_PORTS='`(测试里"反查定义次数"的**引用**)不算重复定义 ✓
#      —— 否则 G2 会把"检查名单唯一性"的测试自己判死(实测 scripts/tests/test_whoami_gate.sh)
# ⚠️ 本门先去掉注释再判:注释里引用名单(说明"真源在哪")是允许的 ✓
#
# 用法: scripts/gates/check_gate_selfconsistency.sh [repo_root]
# 退出码:0 = 全绿 · 1 = 有命中 · 2 = 用法/内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G2: 找不到 repo root: $ROOT" >&2; exit 2; }

python3 - "$ROOT" <<'PY'
import os, re, sys

root = sys.argv[1]
LIB = "scripts/lib_proc_identity.sh"
WHO = "scripts/whoami_proc.sh"
CN  = "\u4e00-\u9fff"

def strip_comment(line):
    """去掉整行/行尾注释(引号感知)。与 G1 同款:只跳到本行行尾,绝不吞掉后续行 ✓"""
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

# ⚠️ 名单定义名【在运行期拼出来】—— 否则本门自己的源码里就出现该字面量 ⇒ 自命中 ✗
NAME_A = "PROD_" + "PORTS" + "="
NAME_B = "PROD_" + "PIDFILES" + "_RE" + "="
# 只认赋值语句:`(行首|;|空白) [export|local|declare|readonly] NAME=`
DEF = re.compile(r"(?:^|[;\s])(?:(?:export|local|declare|readonly)\s+)?"
                 r"(?:" + re.escape(NAME_A) + "|" + re.escape(NAME_B) + ")")

viol = []
scanned = 0

# ── A1:名单只允许定义一次 ────────────────────────────────────────────────────
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__")]
    for f in sorted(fn):
        if not f.endswith(".sh"):
            continue
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, root)
        scanned += 1
        if rel == LIB:
            continue
        try:
            lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            code = strip_comment(line)
            if DEF.search(code):
                viol.append((rel, i,
                             "A1 重复定义生产名单(唯一真源是 %s)⇒ 必然漂移" % LIB,
                             line.rstrip()))

# ── A2/A3:whoami_proc.sh 自洽性 ─────────────────────────────────────────────
who = os.path.join(root, WHO)
if not os.path.isfile(who):
    print("G2 ⛔ 内部错:找不到 %s" % WHO, file=sys.stderr)
    sys.exit(2)
wlines = open(who, encoding="utf-8", errors="replace").read().split("\n")

kill_lines = [i for i, l in enumerate(wlines, 1)
              if re.search(r"(?<![A-Za-z0-9_])KILLABLE=1(?![0-9])", strip_comment(l))]
anchor = [i for i, l in enumerate(wlines, 1)
          if re.search(r'CV_RAW"\s*=\s*"0"', strip_comment(l))]

if len(kill_lines) != 1:
    viol.append((WHO, kill_lines[0] if kill_lines else 1,
                 "A2 `KILLABLE=1` 必须恰好出现一次(实际 %d 次)⇒ 放行路径不唯一" % len(kill_lines),
                 wlines[kill_lines[0]-1].rstrip() if kill_lines else ""))
elif not anchor:
    viol.append((WHO, kill_lines[0],
                 'A2 找不到 `[ "$CV_RAW" = "0" ]` 锚点 ⇒ 无法证明放行发生在 CV 判据之后',
                 wlines[kill_lines[0]-1].rstrip()))
elif kill_lines[0] <= anchor[0]:
    viol.append((WHO, kill_lines[0],
                 "A2 `KILLABLE=1`(第 %d 行)出现在 CV 判据(第 %d 行)之前 ⇒ 无条件放行(杀生产的洞)"
                 % (kill_lines[0], anchor[0]),
                 wlines[kill_lines[0]-1].rstrip()))

# A3:含中文的字面量参与 =/==/!= 比较(只在 test 行上判,避免误伤同行的中文变量定义)
CMP  = re.compile(r'(?:!=|==|=)\s*"[^"]*[' + CN + r']|"[^"]*[' + CN + r'][^"]*"\s*(?:!=|==|=)')
TEST = re.compile(r'\[\[?|\btest\b')
for i, line in enumerate(wlines, 1):
    code = strip_comment(line)
    if not TEST.search(code):
        continue
    if CMP.search(code):
        viol.append((WHO, i,
                     "A3 对【中文字面量】做相等/不等比较 ⇒ 赋值/比较两处表示必然漂移(要么恒拒要么恒放行)",
                     line.rstrip()))

print("G2 扫描 %d 个 *.sh(名单真源 = %s)" % (scanned, LIB))
if viol:
    print("G2 ⛔ 不通过:%d 处【同一事实两处表示】" % len(viol))
    for rel, ln, why, txt in viol:
        print("   %s:%d: %s" % (rel, ln, why))
        print("       %s" % txt.strip())
    print("   ⇒ 修法:A1 改成 `. \"$(dirname \"${BASH_SOURCE[0]}\")/lib_proc_identity.sh\"` 后引用变量;")
    print("           A2 把唯一的 `KILLABLE=1` 放到 `[ \"$CV_RAW\" = \"0\" ]` 分支内;")
    print("           A3 比较变量,中文只写在 echo/printf/注释/定义里 ✗")
    sys.exit(1)
print("G2 ✅ 通过:名单只定义一次;whoami_proc.sh 的放行点唯一且在 CV 判据之后;无中文比较")
sys.exit(0)
PY
