#!/usr/bin/env python3
"""造一个【reasoning 重】的回放集 —— 用于量 R6 的"tokens/step ↔ GiB"兑换率。

═══════════════════════════════════════════════════════════════════════════
 动机
═══════════════════════════════════════════════════════════════════════════
`dsh_assembled_bench.jsonl` 的**输出偏复制**(同源 ngram 命中 **59.9%**)⇒ 在那个工作点上
**ngram(3.53) ≈ dspark(3.60)**。但 ngram 只在"前文出现过同样 n-gram"时有效,
而真实会话里 **52% 的生成位置是 `reasoning`**(命中仅 29.7%)⇒ **两个方法不是等价替代** ✗

⇒ 需要一个**推理重、没什么可复制**的工作点:
    ⭐ **保留 bench 的 ~9.3K 公共前缀**(system + 25 tool defs + runtime context + 一条真历史)
    ⭐ **把尾部换成【全新的真实用户提问】**(取自 `dsh_user_prompts.txt`)
    ⇒ 前缀缓存照样命中(~9.3K),但尾部是**新问题** ⇒ 模型必须**推理**,而不是接续/复制 ✓

用法:
  python3 scripts/build_reasoning_replay.py --n 30 --out /tmp/dsh_reasoning_bench.jsonl
然后:
  PORT=8070 CONC=4 OUT=/tmp/dsh_reasoning_outputs.jsonl \
    python3 scripts/replay_dsh_bench.py --dataset /tmp/dsh_reasoning_bench.jsonl
  python3 scripts/measure_ngram_copy_rate.py --pairs-jsonl /tmp/dsh_reasoning_outputs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

BENCH = "/home/user/lvllm/process_data/data/dsh_assembled_bench.jsonl"
USERS = "/home/user/lvllm/process_data/data/dsh_user_prompts.txt"
MARK = "CURRENT USER"


def blocks(path: str):
    """把用户提问文件切成块:空行分隔;过滤太短/太长/明显是运行时的块。"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    raw = re.split(r"\n\s*\n", txt)
    out = []
    for b in raw:
        b = b.strip()
        if not b:
            continue
        if len(b) < 25 or len(b) > 1200:
            continue
        # 跳过运行时/策略样板
        if b.startswith("Current runtime context") or b.startswith("Approval policy"):
            continue
        if b.startswith("Current DSH file policy"):
            continue
        out.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--users", default=USERS)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="/tmp/dsh_reasoning_bench.jsonl")
    ap.add_argument("--tokens", type=int, default=192)
    args = ap.parse_args()

    first = json.loads(open(args.bench, encoding="utf-8").readline())
    full = first["prompt"]
    i = full.rfind(MARK)
    if i < 0:
        print(f"✗ 在 bench prompt 里找不到 {MARK!r}"); sys.exit(2)
    prefix = full[: i + len(MARK)] + "\n"
    print(f"公共前缀: {len(prefix):,} 字符(取自 bench 第 1 条,到最后一个 {MARK!r})")

    bs = blocks(args.users)
    print(f"可用用户提问块: {len(bs)}(已过滤 <25 / >1200 字符与运行时样板)")
    if len(bs) < args.n:
        print(f"⚠️ 只有 {len(bs)} 块,按实际数量走")
    sel = bs[: args.n]

    with open(args.out, "w", encoding="utf-8") as f:
        for b in sel:
            f.write(json.dumps({"prompt": prefix + b, "output_tokens": args.tokens},
                               ensure_ascii=False) + "\n")
    print(f"✅ 已写 {len(sel)} 条 → {args.out}")
    print("   样例(第 1 条的尾部 120 字符): " + repr(sel[0][-120:]))
    print("⚠️ 这些是【全新提问】⇒ 输出应是推理为主,可复制内容少 ⇒ 正是要的工作点 ✓")


if __name__ == "__main__":
    main()
