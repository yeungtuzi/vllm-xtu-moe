#!/usr/bin/env bash
# G6 —— 【拼写 / 语法 / 路径写法】静态门禁(2026-10-07)
#
# 来由(每一条都对应一次真实事故):
#   ① 把 `scripts` 与 `proc.sh` 之间的斜杠误写成**点号** ⇒ 曾让 4 个文件全错、门恒拒 ✗
#   ② `bash -n` 是底线:语法都不过的脚本不该进仓库 ✓
#   ③ "续行链被注释打断" —— `bash -n` **查不出来** ✗(AGENTS 明写):
#         cmd --a \
#         # 说明
#           --b        ⇒ `--b` 变成新命令(serve_v41.sh 2026-09-23 踩过)
#   ④ 用 `$0` 推脚本目录 ⇒ 被 `source`/`bash -c` 时路径错;正确是 `${BASH_SOURCE[0]}` ✓
#   ⑤ 硬编码固定 `/tmp/<名字>` ⇒ 并发实例互相覆盖(要临时文件请用 `mktemp`)✓
#
# fail 条件(R1-R4 = 致命):
#   R1 任一 `*.sh` 的【代码】(去注释后)出现 `scripts` `<点>` `proc.sh` 笔误
#   R2 任一 `*.sh` 不过 `bash -n`
#   R3 某【代码】行以 `\` 结尾,而下一行是纯注释行
#   R4 任一 `*.sh` 的【代码】里出现 `dirname` + `$0`
#   R5 任一 `*.sh` 的【代码】里出现硬编码 `/tmp/<字面名字>`(同行有 `mktemp` 的跳过)
#
# ⚠️ R5 为什么只是【警告】而不是致命(2026-10-07 依用户反馈调整):
#    实测全仓一次报 109 处,连 `OUT="${OUT:-/tmp/bench_prefill}"` 这种"有 env 覆盖的
#    默认值"也被判死 ⇒ 一个必然大面积报红的门,实际效果是被绕过/逼出逃生口 ✗
#    (本项目已踩过"门恒拒 ⇒ 没人用"的坑)。且 R5 与"误杀生产"无关,属卫生/健壮性问题
#    (并发同机跑两个实例时才会互相覆盖),不是本次验收判据。
#    ⇒ R5 只打印 ⚠️ 警告 + 前 10 条例子,**不影响退出码**;R1-R4 保持致命 ✓
#
# ⚠️ R1/R4 先去注释再判;R1/R4 还跳过 `echo/printf` 引号串里的"报告式提及"
#    (实测 scripts/tests/run_all.sh 在 echo 里报告该笔误,不是笔误本身)✓
# ⚠️ R3 只看【代码行】的续行:纯注释块里的 `# ... \` 是文档,不是断链 ✓
#
# 用法: scripts/gates/check_spelling_and_syntax.sh [repo_root]
# 退出码:0 = 无致命命中(R5 警告不影响)· 1 = 有 R1-R4 命中 · 2 = 用法/内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G6: 找不到 repo root: $ROOT" >&2; exit 2; }

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

# ⚠️ 模式串运行时拼装 ⇒ 本门自己的源码里不出现这些字面量(防自命中)✗
TYPO   = re.compile("scripts" + r"\." + "proc" + r"\.sh")
DIR0   = re.compile("dirname" + r"\s+[\"']?\$0\b")
TMPFIX = re.compile("(?<![A-Za-z0-9_])/tmp/[A-Za-z0-9][A-Za-z0-9._-]*")
DOCSTR = re.compile(r"\b(?:echo|printf)\b")

def in_quote(s, pos):
    """pos 处的字符是否落在单/双引号里"""
    q = None; i = 0
    while i < pos:
        c = s[i]
        if q:
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        i += 1
    return q is not None

def doc_mention(code, rx):
    """doc 提及 ≠ 真笔误:匹配落在 `echo/printf` 的引号串里就跳过 ✓
       (实测 scripts/tests/run_all.sh 在 echo 串里【报告】该笔误,不是笔误本身)"""
    m = rx.search(code)
    if not m:
        return False
    return bool(DOCSTR.search(code[:m.start()])) and in_quote(code, m.start())

r1 = []; r2 = []; r3 = []; r4 = []; r5 = []
scanned = 0

for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__")]
    for f in sorted(fn):
        if not f.endswith(".sh"):
            continue
        p = os.path.join(dp, f); rel = os.path.relpath(p, root)
        scanned += 1
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except OSError as e:
            r2.append((rel, "0", "读文件失败: %s" % e, "")); continue
        lines = text.split("\n")
        codes = [strip_comment(l) for l in lines]

        # R2 语法
        cp = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        if cp.returncode != 0:
            err = (cp.stderr.strip().split("\n") or ["bash -n 失败"])[0]
            r2.append((rel, "0", "bash -n 不通过: %s" % err, ""))

        for i, line in enumerate(lines, 1):
            code = codes[i-1]
            if TYPO.search(code) and not doc_mention(code, TYPO):
                r1.append((rel, i, "点号笔误:scripts 与 proc.sh 之间应是 `/`", line))
            if DIR0.search(code) and not doc_mention(code, DIR0):
                r4.append((rel, i, "用 $0 推路径 ⇒ 应改 ${BASH_SOURCE[0]}", line))
            if "mktemp" not in code and TMPFIX.search(code):
                r5.append((rel, i, "硬编码固定 /tmp 路径 ⇒ 应改 mktemp", line))
            # R3 续行链被注释打断(只看代码行的 `\`;纯注释块里的 `\` 是文档)
            if (code.rstrip().endswith("\\") and not line.lstrip().startswith("#")
                    and i < len(lines) and re.match(r"\s*#", lines[i])):
                r3.append((rel, i, "续行 `\\` 后紧跟注释行 ⇒ 链条被打断(`bash -n` 查不出)",
                           line + "\n" + lines[i]))
                continue

fatal = ([(x, "R1 点号笔误") for x in r1] + [(x, "R2 bash -n 语法错") for x in r2]
         + [(x, "R3 续行链被注释打断") for x in r3] + [(x, "R4 用 $0 推路径") for x in r4])

# ⭐ 防恒绿:扫到 0 个文件(或少于 git tracked)⇒ 内部错,绝不判通过 ✓
if scanned == 0:
    print("G6 ⛔ 内部错:扫描到 0 个 *.sh ⇒ 门会恒绿,拒绝判通过", file=sys.stderr)
    sys.exit(2)
try:
    n_tr = len([x for x in subprocess.run(["git", "-C", root, "ls-files", "-z", "--", "*.sh"],
                                          capture_output=True).stdout.split(b"\0") if x])
except Exception:
    n_tr = -1
if n_tr > 0 and scanned < n_tr:
    print("G6 ⛔ 内部错:scanned=%d < git tracked=%d ⇒ 有漏扫,拒绝判通过" % (scanned, n_tr),
          file=sys.stderr)
    sys.exit(2)

print("G6 扫描 %d 个 *.sh(git tracked=%s)" % (scanned, ("?" if n_tr < 0 else n_tr)))
if r5:
    # R5 只是警告(见脚本头注释):只报前 10 条,不影响退出码 ✓
    print("   ⚠️ R5 警告:%d 处硬编码 /tmp(不阻塞;建议逐步改 mktemp)。前 %d 条:"
          % (len(r5), min(10, len(r5))))
    for rel, ln, why, line in r5[:10]:
        print("      %s:%s: %s" % (rel, ln, line.strip()))
    if len(r5) > 10:
        print("      …(其余 %d 条略)" % (len(r5) - 10))
if fatal:
    print("G6 ⛔ 不通过:%d 处(R1=%d · R2=%d · R3=%d · R4=%d)"
          % (len(fatal), len(r1), len(r2), len(r3), len(r4)))
    for (rel, ln, why, txt), tag in fatal:
        print("   [%s] %s:%s: %s" % (tag, rel, ln, why))
        for t in (txt if isinstance(txt, str) else str(txt)).split("\n")[:2]:
            if t.strip():
                print("       %s" % t.rstrip())
    print("   ⇒ 修法:R1 把点号改回斜杠 `scripts/proc.sh`;R2 修语法到 `bash -n` 过;")
    print("           R3 把 `\\` 续行链里的注释挪到链外;R4 改 `${BASH_SOURCE[0]}` ✗")
    print("           (R5 的 /tmp 警告不阻塞本门,但建议按提示逐步收敛)")
    sys.exit(1)
if r5:
    print("G6 ✅ 通过(R1-R4 全绿):无点号笔误 · 全部 bash -n 通过 · 无断链续行 · 无 $0 推路径")
    print("   ⚠️ 另有 %d 处 /tmp 硬编码警告(不阻塞)" % len(r5))
else:
    print("G6 ✅ 通过:无点号笔误 · 全部 bash -n 通过 · 无断链续行 · 无 $0 推路径 · 无硬编码 /tmp")
sys.exit(0)
PY
