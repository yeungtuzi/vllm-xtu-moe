#!/usr/bin/env bash
# G5 —— 【stop 调用点必须检查返回码】静态门禁(2026-10-07)
#
# 来由:2026-10-07 事故② —— `proc.sh stop dsv41_prod` 只停了包装脚本(真服务还在),
#   调用方**没看返回码**就继续 `spawn` ⇒ 又起了一个实例(端口冲突/双实例)✗
#   `proc.sh stop` 的退出码是有语义的:0=已停 · 3=无 PID 文件/非法名 · 4=归属不是
#   "我自己的调试实例"⇒ **被拒** · 5=发信号后仍存活。⇒ 把 rc 丢掉 = 把门关掉 ✗
#
# fail 条件(任一命中即 fail):
#   A1 同一【逻辑单元】里出现 `proc.sh ... stop`,并且:
#      (a) 同时出现 `>/dev/null`(含 `2>/dev/null`)或 `|| true`;或
#      (b) stop **之后接了管道 `|`** —— 退出码被下游命令(如 `tail`/`grep`)掩盖 ✗
#          ⭐ 审计 2026-10-07 证伪:`bash scripts/proc.sh stop "$n" | tail -3` 的 rc=0
#             (取的是 tail 的 rc)⇒ 必须补上这条,否则漏检
#          ⚠️ 唯一正确读法是 `${PIPESTATUS[0]}`;显式用了 PIPESTATUS 则放行 ✓
#          ⚠️ 只判"stop 这条命令自己"被管道:截到最近的 `;`/`&&`/`||` 为止,
#             所以 `out=$(proc.sh stop x); echo "$out" | grep` 不误伤 ✓
#   A2 `$( ... proc.sh ... stop ... )` 命令替换的结果被丢弃
#      (既没 `VAR=$(...)` / `VAR="$(...)"` 捕获,也不在 `if/while/[ ]` 条件里)
#      ⚠️ 捕获必须认【带引号】的写法:`_s_out="$(... 2>&1)"; _s_rc=$?`
#         —— 这是审计推荐的两条修法之一;不认引号会把修好的调用点误判成违规
#         (实测踩过:13 处假阳性)✗
#      ⚠️ 已知上限:本门只静态确认"结果被捕获",不跨行追踪"捕获后是否真的看了 $?"
#         (跨行追踪误伤太大 —— `rc=$?` 常写在下一行);A1 才是"丢输出"的硬判据 ✓
#
# 豁免:
#   `scripts/gates/allow_stop_callsites.txt`(每行 `<路径或glob><TAB><理由>`)
#   命中的文件整体跳过。当前名单【为空】—— 原 `dev-docs/mywork/`(含 smallmodel 线)已于
#   2026-10-07 按用户指示整体归档,不再预设任何豁免 ✓
#
# 用法: scripts/gates/check_stop_callsites.sh [repo_root]
# 退出码:0 = 全绿 · 1 = 有命中 · 2 = 用法/内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G5: 找不到 repo root: $ROOT" >&2; exit 2; }

python3 - "$ROOT" <<'PY'
import fnmatch, os, re, subprocess, sys

root = sys.argv[1]
allow_file = os.path.join(root, "scripts/gates/allow_stop_callsites.txt")

# ── 读豁免名单(格式错 ⇒ 内部错 exit 2,不静默放过)─────────────────────────
exempt = {}
if os.path.isfile(allow_file):
    with open(allow_file, encoding="utf-8") as fh:
        for raw in fh:
            ln = raw.rstrip("\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) < 2 or not parts[1].strip():
                print("G5: allow_stop_callsites.txt 格式错(需要 `<路径或glob>\\t<理由>`): %s" % ln)
                sys.exit(2)
            exempt[parts[0].strip()] = parts[1].strip()

def exempt_reason(rel):
    for pat, why in exempt.items():
        if fnmatch.fnmatch(rel, pat):
            return why
    return None

def strip_comment(line):
    """去掉整行/行尾注释(引号感知)"""
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
    """把续行(`\\` 结尾)合并成一个逻辑单元 ⇒ 跨行的 `stop ... \\` + `>/dev/null` 不漏 ✓"""
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

# ⚠️ 判据里的模式串【拆开写】:防止本门自己的源码被自己命中(实测过这类静默失明)✗
STOP_CALL = re.compile(r'proc\.sh["\']?\s+\bstop\b')
DISCARD   = re.compile(r'\|{2}\s*true\b|>\s*/dev/null')
SUBST     = re.compile(r'\$\(')
# 捕获认 `VAR=$(...)` 与 `VAR="$(...)"` / `VAR='$(...)'`(审计修法之一)✓
CAPTURE   = re.compile(r'[A-Za-z_][A-Za-z0-9_]*=["\']?\$\(|\$\(\s*$')
COND      = re.compile(r'\[\[?|\b(?:if|elif|while|until)\b|(?:^|\s)!')

viol = []
scanned = 0
skipped = []
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__")]
    for f in sorted(fn):
        if not f.endswith(".sh"):
            continue
        p = os.path.join(dp, f); rel = os.path.relpath(p, root)
        why = exempt_reason(rel)
        if why:
            skipped.append((rel, why)); continue
        try:
            lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        scanned += 1
        for start, rawunit, code in logical_units(lines):
            # 先把续行 `\`+换行压平 ⇒ 跨行的 `| tail` 也能看到 ✓
            code = re.sub(r"\\\n", " ", code)
            m0 = STOP_CALL.search(code)
            if not m0:
                continue
            # A1(a):同一逻辑单元里把输出丢进 /dev/null 或 `|| true`
            if DISCARD.search(code):
                bad = ">/dev/null" if re.search(r">\s*/dev/null", code) else "|| true"
                viol.append((rel, start, "A1 stop 之后跟了 `%s` ⇒ 退出码被掩盖" % bad, rawunit))
                continue
            # A1(b):stop 这条命令自己被管道
            if "PIPESTATUS" not in code:
                piped = False
                for m in STOP_CALL.finditer(code):
                    tail = code[m.end():]
                    cut = len(tail)
                    for sep in (";", "&&", "||", "\n"):
                        j = tail.find(sep)
                        if j >= 0:
                            cut = min(cut, j)
                    if re.search(r"(?<!\|)\|(?!\|)", tail[:cut]):
                        piped = True; break
                if piped:
                    viol.append((rel, start,
                                 "A1 stop 之后接了管道 `|` ⇒ 退出码被下游命令掩盖(应去管道或用 PIPESTATUS[0])",
                                 rawunit))
                    continue
            # A2:命令替换里的 stop,结果被丢弃
            if SUBST.search(code) and not CAPTURE.search(code) and not COND.search(code):
                viol.append((rel, start,
                             "A2 `$( ... stop ... )` 的退出码被丢弃(既未捕获也非条件)",
                             rawunit))

# ⭐ 防恒绿:扫到 0 个文件(或少于 git tracked)⇒ 内部错,绝不判通过 ✓
if scanned == 0:
    print("G5 ⛔ 内部错:扫描到 0 个 *.sh ⇒ 门会恒绿,拒绝判通过", file=sys.stderr)
    sys.exit(2)
try:
    n_tr = len([x for x in subprocess.run(["git", "-C", root, "ls-files", "-z", "--", "*.sh"],
                                          capture_output=True).stdout.split(b"\0") if x])
except Exception:
    n_tr = -1
if n_tr > 0 and scanned < n_tr:
    print("G5 ⛔ 内部错:scanned=%d < git tracked=%d ⇒ 有漏扫,拒绝判通过" % (scanned, n_tr),
          file=sys.stderr)
    sys.exit(2)

print("G5 扫描 %d 个 *.sh(豁免 %d 类 glob;git tracked=%s)" %
      (scanned, len(exempt), ("?" if n_tr < 0 else n_tr)))
for rel, why in skipped:
    print("   ⏭  豁免 %s —— %s" % (rel, why))
if viol:
    print("G5 ⛔ 不通过:%d 处 stop 调用点丢弃了退出码" % len(viol))
    for rel, ln, why, txt in viol:
        first = txt.split("\n")[0]
        print("   %s:%d: %s" % (rel, ln, why))
        print("       %s" % first.strip())
        if "\n" in txt:
            print("       ...(续行已合并:整个逻辑单元都算这一处)")
    print("   ⇒ 修法:检查返回码;被拒(4)⇒ 不得继续 spawn")
    print("      ✅ 正确写法示例:")
    print('         if ! bash "$R/scripts/proc.sh" stop "$name"; then')
    print('           echo "stop 被拒 ⇒ 绝不 spawn" >&2; return 1; fi')
    sys.exit(1)
print("G5 ✅ 通过:所有 proc.sh stop 调用点都检查了返回码")
sys.exit(0)
PY
