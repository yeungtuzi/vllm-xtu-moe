#!/usr/bin/env python3
# ============================================================================
# check_md_emphasis.py —— 检查 Markdown 里【不会渲染】的 * / ** 强调定界符
#
# 为什么需要它:
#   CommonMark 的强调用【flanking 规则】判定 * / ** 能否开/闭。中文写作里
#   极易踩的一条是:
#       ⛔ 在**"显存放不下专家权重"**的情况下      ← ** 紧跟【全角引号】⇒ 可开=False
#       ✅ 在"**显存放不下专家权重**"的情况下      ← 把标点移到定界符【外面】
#   因为 `**` 后面紧跟标点(引号/括号/顿号…)、前面又是汉字时,该定界符
#   【不是 left-flanking】⇒ 开不了;于是它会与后面的定界符错配 ⇒
#   **粗体落到错误的文字上,并且剩下一个 `**` 按字面显示**。
#
# 用法:
#   python3 scripts/check_md_emphasis.py <file.md> [...]     # 指定文件
#   python3 scripts/check_md_emphasis.py --all               # 扫 git 跟踪的 *.md
#   python3 scripts/check_md_emphasis.py --all --include-dev-docs
#
# 退出码:0 = 全部定界符都能正确配对;1 = 存在【未闭合】或【惰性】定界符
#
# ⚠️ 边界:这是 CommonMark 强调算法的【简化实现】(够用于查错),
#    不追求 100% 等价 —— 目的是把"会显示字面 ** 的地方"抓出来 ✓
# ============================================================================
import re
import subprocess
import sys
import unicodedata as ud


def is_punct(ch: str) -> bool:
    return bool(ch) and ud.category(ch[0]) in (
        "Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps")


def is_ws(ch: str) -> bool:
    return (not ch) or ch[0].isspace()


def mask_code(text: str) -> str:
    """把【围栏代码块】与【行内代码】替换成等长的 'x' ⇒ 偏移不变、且不误报 ✓"""
    out = list(text)
    # 围栏代码块(``` 或 ~~~ 起止)
    for m in re.finditer(r"(?m)^[ \t]*(```|~~~).*?$.*?(?:^[ \t]*\1.*?$|\Z)",
                         text, re.S):
        for i in range(m.start(), m.end()):
            if out[i] != "\n":
                out[i] = "x"
    # 行内代码(`...`,含多反引号)
    for m in re.finditer(r"(`+)(?:(?!\1).)*?\1", "".join(out), re.S):
        for i in range(m.start(), m.end()):
            if out[i] != "\n":
                out[i] = "x"
    return "".join(out)


def is_list_bullet(text: str, i: int, run: str, after: str) -> bool:
    """单星号且作列表标记(行首 / 引用块内首列)⇒ 不参与强调 ✓"""
    if len(run) != 1 or after != " ":
        return False
    j = i - 1
    while j >= 0 and text[j] in " \t>":
        j -= 1
    return j < 0 or text[j] == "\n"


def blocks(text: str):
    """把文本切成【强调不能跨过】的块:空行 / 空的引用行(`>`)。
    CommonMark 的定界符栈是按块重置的 ⇒ 不切块会把'一段的错'传染给后面各段 ✗"""
    out, start = [], 0
    for m in re.finditer(r"(?m)^[ \t]*(?:>[ \t]*)?$", text):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(text)))
    return [(a, b) for a, b in out if b > a]


def scan(path: str):
    raw = open(path, encoding="utf-8").read()
    text = mask_code(raw)
    pairs = 0
    problems = []
    for (a, b) in blocks(text):
        seg = text[a:b]
        runs = []
        for m in re.finditer(r"\*+", seg):
            i, j = m.start(), m.end()
            before = seg[i - 1] if i else ""
            after = seg[j] if j < len(seg) else ""
            if is_list_bullet(seg, i, m.group(0), after):
                continue
            # 两侧都是空白 ⇒ 不可能构成强调(典型:乘法号 `a * b`)⇒ 有意为之,跳过 ✓
            if is_ws(before) and is_ws(after):
                continue
            left = (not is_ws(after)) and (not is_punct(after) or is_ws(before) or is_punct(before))
            right = (not is_ws(before)) and (not is_punct(before) or is_ws(after) or is_punct(after))
            runs.append({"pos": a + i, "tok": m.group(0),
                         "open": left, "close": right})
        stack, literal = [], []
        for r in runs:
            if r["close"]:
                k = next((x for x in range(len(stack) - 1, -1, -1)
                          if stack[x]["tok"] == r["tok"] and stack[x]["open"]), None)
                if k is not None:
                    stack.pop(k)
                    pairs += 1
                    continue
            if r["open"]:
                stack.append(r)
            else:
                literal.append(r)
        problems.extend(("未闭合", s) for s in stack)
        problems.extend(("惰性(字面显示)", r) for r in literal)
    return raw, pairs, problems


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def main(argv):
    args = argv[1:]
    all_files = "--all" in args
    dev = "--include-dev-docs" in args
    files = [a for a in args if not a.startswith("--")]
    if all_files:
        files = subprocess.run(["git", "ls-files", "*.md"],
                               capture_output=True, text=True,
                               check=True).stdout.split()
        if dev:
            # ⚠️ dev-docs/ 被 .gitignore 排除 ⇒ git ls-files 看不到它,
            #    必须从【文件系统】枚举,否则 --include-dev-docs 是空转的 ✗
            import glob as _glob
            files += sorted(_glob.glob("dev-docs/**/*.md", recursive=True))
        else:
            files = [f for f in files if not f.startswith("dev-docs/")]
    if not files:
        print(__doc__ or "usage: check_md_emphasis.py <file.md> [...]", file=sys.stderr)
        return 2
    bad = 0
    for f in files:
        try:
            raw, pairs, problems = scan(f)
        except (OSError, UnicodeDecodeError) as e:
            print(f"⚠️  跳过 {f}: {e}")
            continue
        if problems:
            bad += 1
            print(f"⛔ {f}: ✅配对 {pairs} / ⛔问题 {len(problems)}")
            for kind, r in problems:
                ln = line_of(raw, r["pos"])
                ctx = raw[max(0, r["pos"] - 20):r["pos"] + 6].replace("\n", "⏎")
                print(f"     行{ln}: {kind} {r['tok']} ⇒ …{ctx}…")
        else:
            print(f"✅ {f}: ✅配对 {pairs} / ⛔问题 0")
    if bad:
        print(f"\n⛔ {bad} 个文件有【不会渲染】的强调定界符 ⇒ "
              f"把标点移到 `**` 外面(例:在\"**文字**\"的情况下)✓")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
