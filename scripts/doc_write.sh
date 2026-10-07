#!/usr/bin/env bash
# ============================================================================
# doc_write.sh —— 共享文档的【独占写入】
#   落实 AGENTS.md【最高规则 · 2026-10-06】/ IRON_RULES **R36**
#
# 用户原话(定为最高规则):
#   "写入文档要【先用独占写入方式打开(先检查是否失败)】,写完释放,防止重复写入"
#
# ⭐⭐ 语义边界(最容易搞反的一条):
#   · 锁只排他【写者 vs 写者】;⭐⭐【读永远不加锁、不受任何影响】✓
#   · 锁加在【独立锁文件】<file>.lock 上 ⇒ 连正文的 inode 都不碰 ⇒ 更不干扰读 ✓
#   · `flock` 是 advisory:不调用它的进程(read/grep/cat/tail/编辑器…)完全不会被阻塞 ✓
#
# 用法:
#   bash scripts/doc_write.sh <file> --append              <<'EOF'  ...  EOF
#   bash scripts/doc_write.sh <file> --append-auto         <<'EOF'  ...  EOF
#        ⭐ 两者差在【编号】:--append 要求调用方自己写的 `### #NN` 不撞号(撞 ⇒ exit 4,不写);
#           --append-auto 由【脚本在锁内】把首行 `### #NN` 改成 `#<现有最大+1>` ✓
#           ⇒ ⭐ 多会话共写台账时**一律用 --append-auto**(R36 第 4 条的机械执行点 ✓)
#   bash scripts/doc_write.sh <file> --replace '<old>' '<new>'
#   bash scripts/doc_write.sh <file> --truncate-at '<marker>'   # 自 marker 删到文末(撤销误记)
#   bash scripts/doc_write.sh <file> --check     # 只查锁是否空闲(不写)
#
# 退出码:0 成功 / 2 用法或输入错 / 3 拿不到锁(⭐ 一个字节都没写)
#         / 4 编号撞号 或 待替换文本不唯一 或 找不到标记(⭐ 均未写)
# ============================================================================
set -uo pipefail

usage() {
  sed -n '2,20p' "$0" >&2
  exit 2
}

[ $# -ge 2 ] || usage
F="$1"; MODE="$2"; shift 2
[ -f "$F" ] || { echo "⛔ doc_write: 目标文件不存在: $F" >&2; exit 2; }
LOCK="${F}.lock"

# ---- ① 独占打开 + ② 检查是否失败(非阻塞;失败即【不写】)-------------------
exec 9>>"$LOCK" || { echo "⛔ doc_write: 无法打开锁文件 $LOCK" >&2; exit 3; }
if ! flock -n 9; then
  echo "⛔ doc_write: 独占写入失败 —— 另一端正在写 $F(锁被占: $LOCK)" >&2
  echo "   ⇒ 按最高规则【不要写】;请退避后重试(先确认对方已释放)" >&2
  exit 3
fi
# 拿到锁 ⇒ 进入临界区;无论如何退出都要释放
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
trap release EXIT

case "$MODE" in
  --check)
    echo "✓ doc_write: 锁空闲($LOCK)⇒ 现在可以写"
    exit 0
    ;;

  --append|--append-auto)
    tmp="$(mktemp)"; cat > "$tmp"
    if [ ! -s "$tmp" ]; then
      echo "⛔ doc_write: 输入为空,未写" >&2; rm -f "$tmp"; exit 2
    fi
    # ⭐ 编号在【锁内】分配 / 校验(落实 R36 第 4 条):--append 撞号即拒写;--append-auto 由脚本取号
    AUTO=0; [ "$MODE" = "--append-auto" ] && AUTO=1
    python3 - "$F" "$tmp" "$AUTO" <<'PY'
import re, sys
f, tmp, auto = sys.argv[1], sys.argv[2], int(sys.argv[3])
src = open(f, encoding='utf-8').read()
body = open(tmp, encoding='utf-8').read()
mx = max([int(m) for m in re.findall(r'^#{2,3} #(\d+)', src, re.M)] or [0])
if auto:
    body2 = re.sub(r'^### #\d+', lambda m: f"### #{mx+1}", body, count=1, flags=re.M)
    if body2 == body:
        print("⛔ --append-auto: 首行没有 '### #NN' 可分配 ⇒ 未写", file=sys.stderr)
        sys.exit(4)
    body, tag = body2, f"#{mx+1}"
else:
    ids = [int(x) for x in re.findall(r'^### #(\d+)', body, re.M)]
    bad = [i for i in ids if i <= mx]
    if bad:
        print(f"⛔ doc_write: 撞号 —— 本条要写 {bad},而现有最大是 #{mx}"
              f" ⇒ ⭐ 未写(改用 --append-auto 让脚本在锁内取号)", file=sys.stderr)
        sys.exit(4)
    tag = f"#{ids[-1]}" if ids else "-"
open(f, 'a', encoding='utf-8').write(body)
print(f"✓ doc_write: 已【独占】追加 {len(body.encode())} 字节 → {f}(现有最大 #{mx};本条 {tag})")
PY
    rc=$?; rm -f "$tmp"; [ "$rc" -eq 0 ] || exit "$rc"
    ;;

  --replace)
    [ $# -eq 2 ] || { echo "usage: doc_write.sh <file> --replace <old> <new>" >&2; exit 2; }
    OLD="$1"; NEW="$2"
    python3 - "$F" "$OLD" "$NEW" <<'PY'
import os, sys, tempfile
f, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(f, encoding='utf-8').read()
cnt = src.count(old)
if cnt != 1:
    print(f"⛔ doc_write: 待替换文本出现 {cnt} 次(要求恰好 1 次)⇒ 未写", file=sys.stderr)
    sys.exit(4)
out = src.replace(old, new)
d = os.path.dirname(os.path.abspath(f)) or '.'
fd, t = tempfile.mkstemp(dir=d, prefix='.docwrite.')
try:
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(out)
    os.replace(t, f)          # 原子替换:读者要么看到旧版、要么看到新版,不会看到半截 ✓
except BaseException:
    try: os.unlink(t)
    except OSError: pass
    raise
print(f"✓ doc_write: 已【独占】定点替换 → {f}")
PY
    ;;

  --truncate-at)
    # 从 <marker> 起删到文末(含紧邻其前的分隔线)—— 用于【撤销一次误记】
    [ $# -eq 1 ] || { echo "usage: doc_write.sh <file> --truncate-at <marker>" >&2; exit 2; }
    python3 - "$F" "$1" <<'PY'
import os, sys, tempfile
f, marker = sys.argv[1], sys.argv[2]
src = open(f, encoding='utf-8').read()
i = src.find(marker)
if i < 0:
    print(f"⛔ doc_write: 找不到标记 {marker!r} ⇒ 未写", file=sys.stderr)
    sys.exit(4)
head = src[:i].rstrip()
if head.endswith('---'):
    head = head[:-3].rstrip()
head += '\n'
d = os.path.dirname(os.path.abspath(f)) or '.'
fd, t = tempfile.mkstemp(dir=d, prefix='.docwrite.')
try:
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(head)
    os.replace(t, f)
except BaseException:
    try: os.unlink(t)
    except OSError: pass
    raise
print(f"✓ doc_write: 已【独占】撤销 {len(src)-len(head)} 字节(自 {marker!r} 起至文末)→ {f}")
PY
    ;;

  *) usage ;;
esac
