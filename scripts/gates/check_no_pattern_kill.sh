#!/usr/bin/env bash
# G1 —— 【禁止"按模式/名字杀进程"】的静态门禁(2026-10-07)
#
# 来由(用户 2026-10-07 明令 + 一天四次事故,根因全是"按模式找进程"):
#   ① `pkill -f "run_matrix.sh"` 命中我自己的命令行 ⇒ 自杀
#   ③ runner 名 = 被管实例名 ⇒ 脚本第一行 stop 自己 ⇒ 自杀
#   ④ `ps | grep vllm.entrypoints` 把【生产】当"孤儿"杀了 ✗✗
#   ⇒ 所以"能不能杀"必须只由【PID 文件 / 端口 / CUDA_VISIBLE_DEVICES】决定,
#     而"要杀哪些 PID"必须只由【PID 派生】决定(禁止用进程名/命令行模式去【选目标】)✗
#
# fail 条件(任一命中即 fail):
#   R1 `pkill` 而没有 `-P`(按名字/整命令行匹配,会命中自己或生产)
#   R2 `killall`
#   R3 同一逻辑单元里既有 kill 又有【模式派生目标】:
#        `pgrep` / `ps … grep` / `nvidia-smi --query-compute-apps` / `/proc/*/cmdline` 通配
#   R4 Python 里 `os.kill` 且同块里有 `/proc/[0-9]*/cmdline` 通配
#
# 豁免:
#   * 行内 `# allow-pattern-kill: <理由>`(同一逻辑单元内即可)
#   * `scripts/gates/allow_pattern_kill.txt`(每行 `相对路径<TAB>理由`)—— 只给"别的任务线"用
#
# 用法: scripts/gates/check_no_pattern_kill.sh [repo_root]
# 退出码:0 = 全绿 · 1 = 有命中 · 2 = 用法/内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G1: 找不到 repo root: $ROOT" >&2; exit 2; }

python3 - "$ROOT" <<'PY'
import os, re, sys

root = sys.argv[1]
allow_file = os.path.join(root, "scripts/gates/allow_pattern_kill.txt")

file_exempt = {}
if os.path.isfile(allow_file):
    with open(allow_file, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) < 2:
                print(f"G1: allow_pattern_kill.txt 格式错(需要 `<路径>\\t<理由>`): {ln}")
                sys.exit(2)
            file_exempt[parts[0].strip()] = parts[1].strip()

def strip_comment(line):
    """去掉整行注释与行尾注释(粗略处理引号,足够本用途)"""
    s = line
    out = []
    i = 0
    q = None
    while i < len(s):
        c = s[i]
        if q:
            out.append(c)
            if c == q:
                q = None
            i += 1
            continue
        if c in "'\"":
            # ⭐ 三引号必须一起认:否则 docstring 会"开-关-开"错位 ⇒ 之后整段被当成"在引号里",
            #    于是**行尾注释再也不被剥掉** ⇒ 注释里的 `ps|grep`/`os.kill` 被当成代码
            #    (实测:门因此命中自己的说明文字)✗
            if s[i:i+3] in ('"""', "'''"):
                t = s[i:i+3]; j = s.find(t, i+3)
                if j < 0:
                    out.append(t); i += 3; continue
                out.append(t); i = j + 3; continue
            q = c
            out.append(c)
            i += 1
            continue
        if c == "#":
            # `#` 前是空白/行首/换行 ⇒ 视为注释开始
            if i == 0 or s[i-1] in " \t\n;&|":
                # ⚠️ 只能跳到【本行行尾】,绝不能 break 整串 ✗
                #    (踩过:heredoc/for 块是多行单元,一句注释后面的 `os.kill` 会被整段吞掉
                #     ⇒ G1 对 kill_serve.sh 静默失明 —— 正是"检查器自己有洞"这类事故)
                j = s.find("\n", i)
                if j < 0:
                    break
                out.append("\n")
                i = j + 1
                continue
        out.append(c)
        i += 1
    return "".join(out)

def strip_strings(s):
    """把引号串的【内容】抹掉,只留引号本身。

    ⭐ 为什么需要(2026-10-07 独立审计 blocker 6 的根因):
      命令名(pkill/kill/pgrep/ps/awk)从不出现在引号里;而"告警消息 / 正则字面量 /
      路径字面量"里必然出现这些词 ⇒ 若连引号内容一起匹配,门就会**命中自己的源码**
      (实测:R1/R2/R3 三条一起报在 check_no_pattern_kill.sh 自己身上)✗
    ⚠️ 只用于【命令名/模式】类判据;`/proc/[0-9]*` 这类**路径字面量**必须用去注释后的原文判
       (它本身就在引号里)⇒ 两套变体分工明确 ✓
    """
    out = []; i = 0; q = None
    while i < len(s):
        c = s[i]
        if q:
            if c == q:
                q = None; out.append(c)
            elif c == "\n":
                q = None; out.append("\n")     # 未闭合引号不得吞掉后续行
        elif c in "'\"":
            # 三引号(docstring)必须先认:`"""..."""` 若按单引号处理会"开-关-开"错位,
            # 于是 docstring 的内容被当成代码 ⇒ 门会命中自己的文档字面量(实测踩到)
            if s[i:i+3] in ('"""', "'''"):
                t = s[i:i+3]; j = s.find(t, i+3)
                if j < 0:
                    out.append(t); i += 3; continue
                out.append(t); i = j + 3; continue
            q = c; out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)

def logical_units(text):
    """把文件切成逻辑单元:heredoc 整块 / for-while 整块 / 续行合并的整行"""
    lines = text.split("\n")
    units = []
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        # ---- heredoc ----
        m = re.search(r"<<[-]?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", raw)
        if m:
            tag = m.group(1)
            blk = [raw]
            i += 1
            while i < n:
                blk.append(lines[i])
                if lines[i].strip() == tag:
                    i += 1
                    break
                i += 1
            units.append((blk[0], "\n".join(blk)))
            continue
        # ---- for/while 块(含 do…done)----
        if re.match(r"\s*(for|while)\b", raw) and not re.search(r"\bdone\b", raw):
            depth = 1
            blk = [raw]
            i += 1
            while i < n and depth > 0:
                blk.append(lines[i])
                s = strip_comment(lines[i])
                depth += len(re.findall(r"\b(for|while)\b", s)) - len(re.findall(r"\bdone\b", s))
                i += 1
            units.append((blk[0], "\n".join(blk)))
            continue
        # ---- 续行合并 ----
        if raw.rstrip().endswith("\\"):
            blk = [raw]
            i += 1
            while i < n and blk[-1].rstrip().endswith("\\"):
                blk.append(lines[i])
                i += 1
            units.append((blk[0], "\n".join(blk)))
            continue
        units.append((raw, raw))
        i += 1
    return units

KILL_RE   = re.compile(r"(^|[^\w-])(kill|killall|pkill)([^\w-]|$)|xargs\s+kill|os\.kill\s*\(")
PKILL_RE  = re.compile(r"(^|[^\w-])pkill([^\w-]|$)")
KILLALL_RE= re.compile(r"(^|[^\w-])killall([^\w-]|$)")

def pattern_source(ns, raw):
    """这个片段里有没有【按名字/模式选进程】的来源?返回原因或 None。

    ⚠️ 关键区分(否则会误伤正当写法 ⇒ 误报逼出 allow 标记 ⇒ 门形同不存在 ✗):
      * `pgrep -P <pid>`  = **PID 派生**(按父进程找子进程)⇒ 安全 ✓
      * `pgrep -f <pat>` / `pgrep <名字>` = **按名字/命令行模式** ⇒ 危险 ✗
      * 只读的 `nvidia-smi --query-compute-apps`(同一单元里没有 kill)不会被 R3 调用
    ns  = 抹掉引号内容的变体(判命令名);raw = 去注释原文(判 /proc 路径字面量)✓
    """
    if re.search(r"(^|[^\w-])pgrep\s+-[A-Za-z]*f[A-Za-z]*\b", ns):
        return "pgrep -f 目标"
    if re.search(r"(^|[^\w-])pgrep\s+(?!-)[^\s|;)]", ns):
        return "pgrep 按名字目标"
    if re.search(r"(^|[^\w-])ps\b[^\n]*\|[^\n]*grep", ns):
        return "ps|grep 目标"
    # ⭐ `ps | awk '/名字/'` —— 2026-10-07 漏网过:实测 dev-docs/w4a8-assets/phase_run.sh:48 用它在
    #   **全机**找 `VLLM::EngineCor` 再批量 kill ⇒ 会把【生产的 EngineCore】一起 TERM 掉 ✗✗
    #   (与 ps|grep 同类,只是换成了 awk)⇒ 必须一并拦下 ✓
    # ⚠️ 这一条必须用 raw(去注释原文)判:awk 的"名字"本身就在引号里(`awk '/名字/'`),
    #    若用去引号变体会把模式一起抹掉 ⇒ 真的地雷反而漏检(实测 R3b 回归用例变绿)✗
    if re.search(r"\bps\b[^\n]*\|[^\n]*awk\b[^\n]*/[A-Za-z][A-Za-z0-9_:.-]{2,}/", raw):
        return "ps|awk 名字匹配"
    if re.search(r"nvidia-smi[^\n]*--query-compute-apps", ns):
        return "nvidia-smi 全卡目标"
    if re.search(r"/proc/\[[0-9]\*\]/cmdline|glob\.glob\(['\"]/proc", raw):
        return "/proc 通配扫描"
    return None

violations = []
scanned = 0
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__")]
    for fn in filenames:
        if not fn.endswith(".sh"):
            continue
        path = os.path.join(dirpath, fn)
        rel = os.path.relpath(path, root)
        if rel in file_exempt:
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        scanned += 1
        # ---- "污染变量":值来自【模式匹配进程】的变量(跨行数据流)----
        #   实测漏网:tune_sweep_serve.sh `pids=$(nvidia-smi --query-compute-apps…)`
        #             + 下一行 `for p in $pids; do kill -9 "$p"; done` ✗
        #   ⚠️ 必须【只在同一行】且【直接包含模式命令】才算污染,否则会把"PID 文件派生"
        #      的正当写法也误伤 ⇒ 误报会逼出 allow 标记 ⇒ 门形同不存在 ✗(上一个会话的教训)
        _clean = strip_comment(text)
        tainted = set()
        for _m in re.finditer(r"(?:^|[;\s])([A-Za-z_][A-Za-z0-9_]*)=\$\(([^)\n]*)\)", _clean):
            if pattern_source(strip_strings(_m.group(2)), _m.group(2)):
                tainted.add(_m.group(1))
        # ⚠️ 不把 `for p in $(pgrep …)` 的循环变量也算污染:同名变量常被后面的
        #    "PID 文件派生"循环复用 ⇒ 会误伤(longprefill_probe 实测误报)✗;
        #    而那种循环本身已被 R3 命中(kill + 模式命令在【同一单元】)✓
        def _tainted_target(code):
            """只有当【kill 的参数】或【for 的列表】就是那个污染变量时才算命中。
               ⚠️ `kill -0 $VAR` 是**只读探活**,不是杀 ⇒ 必须排除(否则误伤正当写法)✗"""
            for t in tainted:
                v = re.escape(t)
                if re.search(r"\bkill\b(?!\s+-0\b)[^\n]*\$\{?" + v + r"\}?(\W|$)", code):
                    return t
                if re.search(r"\bfor\s+\w+\s+in\s+[^\n]*\$\{?" + v + r"\}?(\W|$)", code):
                    return t
            return None
        for first, body in logical_units(text):
            code = strip_comment(body)
            code_ns = strip_strings(code)   # 判命令名用(见 strip_strings 注释)
            # ⭐⭐ 2026-10-07 独立审计(blocker 6):原判据是"body 里【含】allow-pattern-kill:" ✗
            #   ⇒ 同一单元里任何一句注释只要【提到】这个标记就整单元豁免(实测:
            #     `# note: never add allow-pattern-kill: here` 会让真地雷免检)✗
            #   ⇒ 现在只认:单元【首行】上、`#` 之后、理由非空的**行尾**注释 ✓
            #     (纯注释行没有代码部分 ⇒ 不构成豁免;必须"这行是代码 + 行尾注明了理由")
            _first_code = strip_comment(first)
            _m_marker = re.search(r"#[ \t]*allow-pattern-kill:[ \t]*(\S.*)$", first)
            if (_m_marker and _m_marker.group(1).strip()
                    and _first_code.strip()
                    and not first.lstrip().startswith("#")):
                continue
            # R1: pkill 而无 -P —— ⛔ 独立审计(blocker 5):原来只扫 `first`(单元首行)✗
            #   ⇒ for/while 块与 heredoc 里【第二行起】的 `pkill -f` 完全不被检测,
            #     而那正是门要防的历史事故形态 ⇒ 现在扫【整个单元】✓
            _cc = code_ns
            for m in PKILL_RE.finditer(_cc):
                # ⛔ 2026-10-07 独立审计(blocker 4):原来用"匹配点后 60 字符窗口里有没有 -P" ✗
                #   ⇒ 单元内稍后出现字面量 -P(例如 `pkill -f X || pkill -P $sup`)就让**真正的
                #      `pkill -f` 完全免检** ⇒ G1 对历史事故命令形态形同不存在 ✗
                #   ⇒ 现在只在该次 pkill【自己的参数段】(截到 ; | && 换行)里找 -P/--parent ✓
                _seg = re.split(r"[;|\n]|&&", _cc[m.start():], 1)[0]
                if not re.search(r"(?:^|\s)(?:--parent|-P)\b", _seg):
                    violations.append((rel, first.strip()[:110], "R1 pkill 按名字/模式(未用 -P)"))
            # R2: killall(同样扫整个单元)
            if KILLALL_RE.search(_cc):
                violations.append((rel, first.strip()[:110], "R2 killall"))
            # R3: 同单元 kill + 模式派生目标
            if KILL_RE.search(code_ns):
                _why = pattern_source(code_ns, code)
                if _why:
                    violations.append((rel, first.strip()[:110], "R3 kill + " + _why))
                else:
                    # R3b: kill/for 的目标【就是】那个"模式匹配得到的变量"
                    _t = _tainted_target(code_ns)
                    if _t:
                        violations.append((rel, first.strip()[:110],
                                           f"R3b kill/for 的目标来自模式匹配变量 ${_t}"))
            # R4: heredoc 里的 os.kill + /proc 通配扫描(用去注释后的代码,避免门自己命中自己的说明文字)
            if "os.kill" in code and re.search(r"/proc/\[0-9\]\*|glob\.glob\(['\"]/proc", code):
                tgt = (rel, first.strip()[:110], "R4 python os.kill + /proc 通配扫描")
                if tgt not in violations:
                    violations.append(tgt)

# 去重并排序
seen = set(); uniq = []
for v in violations:
    if v not in seen:
        seen.add(v); uniq.append(v)

# ⭐ 2026-10-07 独立审计(minor 但很危险):"一个 *.sh 都没扫到"仍会打印 ✅ 通过 ⇒
#   目录结构/扩展名一变,门就**恒绿** ✗ ⇒ 现在 scanned==0 直接判内部错 exit 2 ✓
if scanned == 0:
    print("G1: 内部错 —— 一个 *.sh 都没扫到 ⇒ 拒绝判绿(检查 root/扩展名)✗")
    sys.exit(2)

print(f"G1 扫描 {scanned} 个 *.sh(豁免 {len(file_exempt)} 个文件)")
if uniq:
    print(f"G1 ⛔ 不通过:{len(uniq)} 处【按模式/名字杀进程】")
    for rel, line, why in uniq:
        print(f"   {rel}: {why}\n       {line}")
    print("   ⇒ 修法:改成【PID 文件派生 + 归属校验】(见 scripts/lib_proc_identity.sh)," )
    print("     确需保留请在同行写 `# allow-pattern-kill: <理由>` ✗")
    sys.exit(1)
print("G1 ✅ 通过:没有按模式/名字杀进程的写法")
sys.exit(0)
PY
