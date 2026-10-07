#!/usr/bin/env bash
# G8 —— 【禁止裸起后台进程】静态门禁(2026-10-07)
#
# 来由:AGENTS 第 0 条 —— 长驻进程必须走 `scripts/proc.sh spawn`(写 PID + 持久日志);
#   随手 `nohup ... &` / `setsid ... &` ⇒ **PID 不可追溯、日志可能丢** ✗
#   ⇒ 出事故时"这个进程是谁起的、怎么停"就说不清(2026-10-07 事故②的温床)✓
#
# fail 条件:
#   在 `scripts/proc.sh` **之外**的任一 `*.sh` 里,同一【逻辑单元】(续行合并)中
#   同时出现 `nohup`/`setsid` 与**后台 `&`** ⇒ fail
#   (⚠️ `&&`、`2>&1`、`&>` 都不是后台符,不算 ✓)
#
# 豁免:
#   * 同一逻辑单元里有 `# allow-bare-launch: <理由>` ⇒ 跳过(必须写明理由)
#   * `scripts/proc.sh` 本身(它就是"受管启动"的实现,内部用 setsid)✓
#
# ⚠️ 关于"tracked":`git ls-files '*.sh'` 只有 60 个(全在 scripts/),
#   而 dev-docs/ 是未跟踪的整改现场且 G1 已在那里发现地雷 ⇒ 本门实际扫描
#   **仓库内全部 `*.sh`(排除 .git)**,是 tracked 的超集(打印两个计数以便核对)✓
#
# 用法: scripts/gates/check_no_bare_launch.sh [repo_root]
# 退出码:0 = 全绿 · 1 = 有命中 · 2 = 用法/内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G8: 找不到 repo root: $ROOT" >&2; exit 2; }

python3 - "$ROOT" <<'PY'
import os, re, subprocess, sys

root = sys.argv[1]
IMPL = "scripts/proc.sh"        # 受管启动的实现,允许其内部 setsid

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

NO_RE   = re.compile(r"\b(nohup|setsid)\b")
# 后台 `&`:排除 `&&` / `&>` / `>&`(即 `2>&1` 之类)
BACKGROUND = re.compile(r"(?<!&)(?<!>)\&(?!&)(?!>)")
MARKER  = "allow-bare-launch:"

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

try:
    tracked = subprocess.run(["git", "-C", root, "ls-files", "-z", "--", "*.sh"],
                             capture_output=True).stdout.split(b"\0")
    n_tracked = len([x for x in tracked if x])
except Exception:
    n_tracked = -1

viol = []; scanned = 0; exempted = 0; file_exempt = {}
# ⭐ 文件级豁免名单(与 G1/G5 同构):每行 `<路径或glob><TAB><理由>` ✓
#   为什么需要:① `serve_*.sh` 是 **proc.sh spawn 的载荷脚本** —— 它们内部 nohup/setsid
#   正是"把真服务交出去"的方式(bringup 注释已记录:PID 文件记的是包装脚本,真 PID 靠
#   adopt 认领)⇒ 让它们再走 proc.sh spawn 会嵌套管理、本末倒置 ✗
#   ② 其余 runner 自己写 PID 文件与日志(`echo $! > …pid` / `> $log`)⇒ 可追溯性由自己
#   承担;迁移到 proc.sh spawn 列为【显式技术债】,不在本次验收必备集内 ✓
_ALLOW = os.path.join(root, "scripts/gates/allow_bare_launch.txt")
if os.path.isfile(_ALLOW):
    for _ln in open(_ALLOW, encoding="utf-8"):
        _ln = _ln.rstrip("\n")
        if not _ln.strip() or _ln.lstrip().startswith("#"):
            continue
        _p = _ln.split("\t")
        if len(_p) < 2 or not _p[1].strip():
            print("G8: allow_bare_launch.txt 格式错(需要 `<路径或glob>\\t<理由>`): %s" % _ln)
            sys.exit(2)
        file_exempt[_p[0].strip()] = _p[1].strip()

for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__")]
    for f in sorted(fn):
        if not f.endswith(".sh"):
            continue
        rel = os.path.relpath(os.path.join(dp, f), root)
        scanned += 1
        if rel == IMPL:
            exempted += 1; continue
        if rel in file_exempt:
            exempted += 1; continue
        try:
            lines = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        for start, rawunit, code in logical_units(lines):
            if MARKER in rawunit:
                continue
            bg = BACKGROUND.search(code)
            if bg is None:
                continue
            if not NO_RE.search(code):
                continue
            what = "nohup" if re.search(r"\bnohup\b", code) else "setsid"
            viol.append((rel, start, "裸起后台进程(`%s ... &`)⇒ PID/日志不可追溯" % what, rawunit))

print("G8 扫描 %d 个 *.sh(git tracked=%s;豁免实现文件 %d 个)" %
      (scanned, ("?" if n_tracked < 0 else n_tracked), exempted))
if viol:
    print("G8 ⛔ 不通过:%d 处裸起后台进程" % len(viol))
    for rel, ln, why, txt in viol:
        print("   %s:%d: %s" % (rel, ln, why))
        print("       %s" % txt.split("\n")[0].strip())
        if "\n" in txt:
            print("       ...(续行已合并:整条启动命令算这一处)")
    print("   ⇒ 修法:改走 `scripts/proc.sh spawn <name> <cmd...>`(写 <name>.pid + <name>.log);")
    print("           确需裸起请在【同一逻辑单元】写 `# allow-bare-launch: <理由>` ✗")
    sys.exit(1)
print("G8 ✅ 通过:proc.sh 之外没有裸起后台进程")
sys.exit(0)
PY
