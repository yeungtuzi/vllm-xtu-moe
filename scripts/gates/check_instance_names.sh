#!/usr/bin/env bash
# G7 —— 【实例名 / 保留端口 / 自杀模式】静态门禁(2026-10-07)
#
# 来由(2026-10-07 事故③ 加 端口冲突):
#   ① 实例名会直接拼进 `<logdir>/<name>.pid` ⇒ 名字里带空格/`/`/`:` 会写出
#      目录穿越或匹配不到的 PID 文件(门与 proc.sh 对同一名字理解不一致)✗
#   ② 调试 runner 用【生产保留端口】起实例 ⇒ 与生产抢端口/被误认领 ✗
#      (2026-10-07 实证:在活生产上起第二个全量实例)
#   ③ **runner 名 == 被管实例名** ⇒ 脚本第一行 `stop <自己>` ⇒ 当场自杀 ✗
#      (这是当天第 3 次事故的直接形态;判据来自 AGENTS 第 0 条推论①)
#
# fail 条件:
#   N1 传给 `proc.sh spawn|adopt|stop` 的【字面量】实例名不含 `[A-Za-z0-9._-]`,
#      或是 `.` / `..`(变量 `"$var"` 形式跳过 —— 那是运行期值,不是字面量 ✓)
#   N2 任一 `*.sh` 的代码里出现【字面量保留端口】`PORT=<生产端口>` / `--port <生产端口>`
#      / `MAINLINE_PORT=` / `FORK_PORT=`
#   N3 `scripts/*.sh` 之类的 runner 里出现 `stop <自己文件名去掉 .sh>`(自杀模式)
#
# ⚠️ 用法/文档字符串(echo/printf/sed/grep 里的名字)不算调用 ⇒ 跳过 ✓
# ⭐⭐ 保留端口名单【只能来自唯一真源】`scripts/lib_proc_identity.sh` 的 `$PROD_PORTS`:
#    ⛔ 禁止在门里手写名单 —— 手写 3 个(8070/5555/8080)会**漏检**另外 4 个
#       (9090/3000/9100/8787),这正是本项目"同一事实两处表示"的老病(审计 2026-10-07 证伪)✗
#    ⇒ 本门 source 真源后把 `$PROD_PORTS` 传给 python 动态生成判据 ✓
#
# ⚠️ 扫到 0 个 *.sh(或少于 git tracked)⇒ 判【内部错 exit 2】,绝不判"通过"(防恒绿)✓
#
# 用法: scripts/gates/check_instance_names.sh [repo_root]
# 退出码:0 = 全绿 · 1 = 有命中 · 2 = 用法/内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G7: 找不到 repo root: $ROOT" >&2; exit 2; }

# ⭐ 生产保留端口名单:唯一真源(绝不在本门手写)✓
LIB="$ROOT/scripts/lib_proc_identity.sh"
[ -r "$LIB" ] || LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib_proc_identity.sh"
[ -r "$LIB" ] || { echo "G7: 找不到唯一真源 lib_proc_identity.sh" >&2; exit 2; }
# shellcheck source=lib_proc_identity.sh
. "$LIB"

python3 - "$ROOT" "$PROD_PORTS" <<'PY'
import os, re, subprocess, sys

root = sys.argv[1]
RESERVED = [p for p in sys.argv[2].split() if p]   # ⭐ 来自唯一真源 $PROD_PORTS

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

# 字面量端口运行时拼装(名单来自 $PROD_PORTS,本门不写死任何一个端口)✓
PORTLIT = re.compile(r"(?<![A-Za-z0-9_])(?:PORT|MAINLINE_PORT|FORK_PORT|--port)\s*[= ]\s*[\"']?(?:"
                     + "|".join(re.escape(p) for p in RESERVED) + r")\b")
PORTLIST = "/".join(RESERVED)
OP  = re.compile(r'proc\.sh["\']?\s+(spawn|adopt|stop)\s+')
# 输出/文档用途的字符串不算"调用":echo/printf/sed/grep + 内嵌 Python 的 print ✓
DOC = re.compile(r"\b(?:echo|printf|sed|grep|print)\b")
NAME_OK = re.compile(r"^[A-Za-z0-9._-]+$")
# 占位符/glob 只有在**不含** `[A-Za-z0-9._-]` 特有危险字符时才跳过;
# ⚠️ `/` 与 `:` 正是最危险的实例名字符(路径穿越/拼错)⇒ 绝不能当占位符放过 ✗
PLACEHOLDER = set("<>{}[]*?!=|&;\\")

def read_name(code, pos):
    """取操作符后面的第一个名字 token;返回 (名字, 是否为变量形式, 是否跳过)"""
    rest = code[pos:]
    if rest[:1] in ("'", '"'):
        q = rest[0]; end = rest.find(q, 1)
        if end < 0:
            return "", False, True
        return rest[1:end], rest[1:end].startswith("$"), False
    tok = re.split(r"[\s;|&)]", rest, 1)[0]
    tok = tok.rstrip(";")
    return tok, tok.startswith("$"), False

n1 = []; n2 = []; n3 = []
scanned = 0
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__")]
    for f in sorted(fn):
        if not f.endswith(".sh"):
            continue
        rel = os.path.relpath(os.path.join(dp, f), root)
        scanned += 1
        try:
            lines = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        selfname = os.path.basename(rel)[:-3]        # 去掉 .sh
        for i, raw in enumerate(lines, 1):
            code = strip_comment(raw)
            if not code.strip():
                continue
            # N2 字面量保留端口
            if PORTLIT.search(code):
                n2.append((rel, i, "调试脚本里出现字面量保留端口(%s)" % PORTLIST, raw))
            # N1 / N3
            if "proc.sh" not in code:
                continue
            for m in OP.finditer(code):
                if DOC.search(code[:m.start()]):
                    continue                       # 用法/文档串里的"调用"不是调用
                name, is_var, skip = read_name(code, m.end())
                if skip or not name or is_var:
                    continue
                if set(name) & PLACEHOLDER:
                    continue                       # `<name>` / glob 等占位符
                if name in (".", ".."):
                    n1.append((rel, i, "非法实例名 `%s`(是路径穿越)" % name, raw)); continue
                if not NAME_OK.match(name):
                    n1.append((rel, i, "非法实例名 `%s`(只允许 [A-Za-z0-9._-])" % name, raw))
                if m.group(1) == "stop" and name == selfname:
                    n3.append((rel, i, "自杀模式:runner 名 == 被管实例名(`stop %s`)" % name, raw))

allv = ([(x, "N1 非法实例名") for x in n1] + [(x, "N2 保留端口") for x in n2]
        + [(x, "N3 自杀模式") for x in n3])

# ⭐ 防恒绿:扫到 0 个文件(或少于 git tracked)⇒ 内部错,绝不判通过 ✓
if scanned == 0:
    print("G7 ⛔ 内部错:扫描到 0 个 *.sh ⇒ 门会恒绿,拒绝判通过", file=sys.stderr)
    sys.exit(2)
try:
    n_tr = len([x for x in subprocess.run(["git", "-C", root, "ls-files", "-z", "--", "*.sh"],
                                          capture_output=True).stdout.split(b"\0") if x])
except Exception:
    n_tr = -1
if n_tr > 0 and scanned < n_tr:
    print("G7 ⛔ 内部错:scanned=%d < git tracked=%d ⇒ 有漏扫,拒绝判通过" % (scanned, n_tr),
          file=sys.stderr)
    sys.exit(2)

print("G7 扫描 %d 个 *.sh(git tracked=%s;保留端口来自真源:%s)" %
      (scanned, ("?" if n_tr < 0 else n_tr), PORTLIST))
if allv:
    print("G7 ⛔ 不通过:%d 处(N1=%d · N2=%d · N3=%d)" % (len(allv), len(n1), len(n2), len(n3)))
    for (rel, ln, why, txt), tag in allv:
        print("   [%s] %s:%d: %s" % (tag, rel, ln, why))
        print("       %s" % txt.strip())
    print("   ⇒ 修法:N1 实例名只留 [A-Za-z0-9._-](过 pi_valid_name);")
    print("           N2 调试实例改 8090-8099 等非保留端口(生产名单见 lib_proc_identity.sh);")
    print("           N3 runner 名必须 ≠ 被管实例名(别让脚本第一行 stop 自己)✗")
    sys.exit(1)
print("G7 ✅ 通过:实例名合法 · 无字面量保留端口(%s) · 无自杀模式" % PORTLIST)
sys.exit(0)
PY
