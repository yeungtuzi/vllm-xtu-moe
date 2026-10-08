#!/usr/bin/env python3
"""离线测【prompt-lookup / ngram 草稿】在本仓【真实 agent 负载】上的接受率。

═══════════════════════════════════════════════════════════════════════════
 为什么需要它(R6 的零重启判据)
═══════════════════════════════════════════════════════════════════════════
R6 的候选实现 = vLLM 内建的 `ngram` proposer(`{"method":"ngram",...}`),
**但生产现在开着 dspark** —— `SpeculativeMethod` 是**单个 Literal** ⇒ **两者不能叠加,只能换** ⇄
⇒ 换上之前必须先知道:**我们的负载里"从前文复制"的比例够不够高?**

本脚本就用**真实的 DSH 会话**(`~/.dsh/sessions/--home-user-lvllm--/*/session.v4.jsonl.zstd`)
按模型实际看到的顺序拼出 token 序列,然后在**助手生成**的每个位置上做与 ngram proposer 同构的查表:

    取最近 L 个 token(前缀),在【之前出现过的地方】找同样的 L-gram;
    找到 ⇒ 它会草稿出接下来的 K 个 token ⇒ 与真实后续逐位比对,数接受长度 ✓

⚠️ 这是**离线近似**,因为:① 真实服务里是**增量**生成的(本脚本用完整序列,前缀一致);
   ② 未计入采样温度/投机窗口的具体调度;③ 会话里是文本、重新过 tokenizer。
⇒ **结论只用于"值不值得换 dspark"的量级判断**,不能替代上机 A/B ✓

用法:
  python3 scripts/measure_ngram_copy_rate.py --sessions 4 --max-tokens 200000
  python3 scripts/measure_ngram_copy_rate.py --l-min 2 --l-max 4 --k 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter

TOK_DEFAULT = os.path.expanduser(
    "~/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master/tokenizer.json")
SESS_GLOB = os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--/*/session.v4.jsonl.zstd")


def read_events(path: str):
    """用 zstd CLI 解压(python 的 zstandard 未装,CLI 有 ✓)。"""
    p = subprocess.run(["zstd", "-dc", path], capture_output=True)
    if p.returncode != 0:
        return []
    out = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def content_text(content) -> str:
    """把 message.content([{type,text},...]) 拼成文本;按类型分开统计。"""
    parts = []
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    for c in content:
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        if t in ("text", "reasoning") and c.get("text"):
            parts.append(c["text"])
        elif t == "tool-call":
            # ⭐ 工具调用的参数(改文件/写文件)正是"复制上下文"最集中的地方
            args = c.get("arguments") or c.get("args") or c.get("input")
            parts.append(json.dumps(args, ensure_ascii=False) if args is not None else "")
    return "\n".join(p for p in parts if p)


def segments(path: str, max_tokens: int, tok):
    """按模型看到的顺序产出 (kind, text):prompt 侧(system/user/tool-result) 与 gen 侧(assistant)。"""
    evs = read_events(path)
    if not evs:
        return
    total = 0
    for e in evs:
        t = e.get("type"); d = e.get("data") or {}
        if t == "system/message":
            txt = content_text((d.get("message") or {}).get("content"))
            if txt:
                yield ("prompt", txt)
        elif t == "user/message":
            txt = content_text(d.get("content") if "content" in d else (d.get("message") or {}).get("content"))
            if txt:
                yield ("prompt", txt)
        elif t == "tool/result":
            txt = content_text(d.get("content") if "content" in d else (d.get("message") or {}).get("content"))
            if txt:
                yield ("prompt", txt)
        elif t == "assistant/message":
            msg = d.get("message") or {}
            for c in (msg.get("content") or []):
                if not isinstance(c, dict):
                    continue
                ct = c.get("type")
                if ct in ("text", "reasoning") and c.get("text"):
                    yield (f"gen:{ct}", c["text"])
                elif ct == "tool-call":
                    args = c.get("arguments") or c.get("args") or c.get("input")
                    if args is not None:
                        yield ("gen:tool-call", json.dumps(args, ensure_ascii=False))
        if total > max_tokens * 8:
            return


def measure(paths, tok, l_min, l_max, k, max_tokens):
    """⭐ 两遍法:**先把完整 token 序列建好**(离线回放才知道"真实后续"),
    再【增量模拟】ngram 查表 —— 索引只含 i 之前的位置 ✓

    ⚠️ 第一版写错的地方:直接拿 `seq[i:i+k]` 当"真实后续"比 —— 而增量生成时 `seq[i:]` 还不存在
    ⇒ 接受长度恒为 0 ⇒ 整组统计作废 ✗(已修)
    """
    n_gen = hit = acc_total = 0
    tokens_total = 0
    by_kind, by_kind_hit, by_kind_acc = Counter(), Counter(), Counter()

    for path in paths:
        # ---- 第一遍:拼完整序列 [(tid, is_gen, kind), ...] ----
        full = []
        for kind, text in segments(path, max_tokens, tok):
            for t in tok.encode(text).ids:
                full.append((t, kind.startswith("gen:"), kind))
                tokens_total += 1
            if tokens_total > max_tokens * 8:
                break
        ids = [x[0] for x in full]

        # ---- 第二遍:增量模拟(索引只含 i 之前)----
        idx = {}
        for i, (_, is_gen, kind) in enumerate(full):
            if is_gen and i >= l_min:
                n_gen += 1
                by_kind[kind] += 1
                best = 0
                for L in range(l_max, l_min - 1, -1):   # 从最长 L 开始,命中即用
                    if i - L < 0:
                        continue
                    j = idx.get(tuple(ids[i - L:i]))
                    if j is None:
                        continue
                    m = 0
                    while (m < k and j + m < i and i + m < len(ids)
                           and ids[j + m] == ids[i + m]):
                        m += 1
                    best = m
                    break
                if best > 0:
                    hit += 1
                    acc_total += best
                    by_kind_hit[kind] += 1
                    by_kind_acc[kind] += best
            # 把以 i 结尾的 L-gram 记入索引(供 i 之后查)
            for L in range(l_min, l_max + 1):
                if i - L >= 0:
                    idx[tuple(ids[i - L:i])] = i

    return dict(n_gen=n_gen, hit=hit, acc_total=acc_total,
                by_kind=by_kind, by_kind_hit=by_kind_hit, by_kind_acc=by_kind_acc,
                tokens_total=tokens_total)


def measure_pairs(path_jsonl, tok, l_min, l_max, k, max_tokens):
    """⭐ 同源模式:读 {prompt, output};prompt = 上下文,output = 生成部分。
    这样与 `replay_dsh_bench.py` 的 dspark 接受率**同源**(同一批 prompt)✓"""
    n_gen = hit = acc_total = 0
    by_kind, by_kind_hit, by_kind_acc = Counter(), Counter(), Counter()
    with open(path_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            ctx = tok.encode(d.get("prompt", "")).ids[-max_tokens:]
            gen = tok.encode(d.get("output", "")).ids
            if not gen:
                continue
            full = [(t, False, "prompt") for t in ctx] + [(t, True, "gen:output") for t in gen]
            ids = [x[0] for x in full]
            base = len(ctx)
            idx = {}
            for i, (_, is_gen, kind) in enumerate(full):
                if is_gen and i >= l_min and i >= base:
                    n_gen += 1; by_kind[kind] += 1
                    best = 0
                    for L in range(l_max, l_min - 1, -1):
                        if i - L < 0:
                            continue
                        j = idx.get(tuple(ids[i - L:i]))
                        if j is None:
                            continue
                        m = 0
                        while (m < k and j + m < i and i + m < len(ids)
                               and ids[j + m] == ids[i + m]):
                            m += 1
                        best = m
                        break
                    if best > 0:
                        hit += 1; acc_total += best
                        by_kind_hit[kind] += 1; by_kind_acc[kind] += best
                for L in range(l_min, l_max + 1):
                    if i - L >= 0:
                        idx[tuple(ids[i - L:i])] = i
    return dict(n_gen=n_gen, hit=hit, acc_total=acc_total,
                by_kind=by_kind, by_kind_hit=by_kind_hit, by_kind_acc=by_kind_acc,
                tokens_total=n_gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=TOK_DEFAULT)
    ap.add_argument("--sessions", type=int, default=4, help="用最大的前 N 个会话")
    ap.add_argument("--max-tokens", type=int, default=200000, help="每会话 token 上限")
    ap.add_argument("--l-min", type=int, default=2)
    ap.add_argument("--l-max", type=int, default=4)
    ap.add_argument("--k", type=int, default=5, help="最多草稿几个 token(=num_speculative_tokens)")
    ap.add_argument("--pairs-jsonl", default=os.environ.get("PAIRS"),
                    help="⭐ 同源模式:读 {prompt,output} 的 jsonl(由 replay_dsh_bench.py 产出)")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    if args.pairs_jsonl:
        print(f"⭐ 同源模式(同一批 prompt + 同一批输出): {args.pairs_jsonl}")
        r = measure_pairs(args.pairs_jsonl, tok, args.l_min, args.l_max, args.k, args.max_tokens)
        return _report(r, args)
    cands = sorted(glob.glob(SESS_GLOB),
                   key=lambda p: -os.path.getsize(p))
    if not cands:
        print("✗ 找不到会话文件"); sys.exit(2)
    paths = cands[:args.sessions]
    print(f"分词器: {args.tokenizer}")
    print(f"会话(取最大的 {len(paths)} 个):")
    for p in paths:
        print(f"  {os.path.getsize(p)/1e6:7.1f} MB  {os.path.basename(os.path.dirname(p))}")
    print(f"参数: L∈[{args.l_min},{args.l_max}]  K={args.k}  每会话上限 {args.max_tokens} token")
    print()

    if args.pairs_jsonl:
        print(f"⭐ 同源模式: {args.pairs_jsonl}")
        r = measure_pairs(args.pairs_jsonl, tok, args.l_min, args.l_max, args.k, args.max_tokens)
    else:
        r = measure(paths, tok, args.l_min, args.l_max, args.k, args.max_tokens)
    return _report(r, args)


def _report(r, args):
    n = r["n_gen"] or 1
    print(f"助手侧生成位置(全部) : {r['n_gen']:>9,}")
    print(f"  命中(能查到前文同 L-gram): {r['hit']:>9,}  ⇒ 命中率 {100*r['hit']/n:5.1f}%")
    print(f"  命中时的平均接受长度   : {r['acc_total']/max(1,r['hit']):5.2f} / 最多 {args.k}")
    print()
    print("⭐ 分类型(这才是关键):")
    print(f"  {'类型':<16}{'位置数':>10}{'命中率':>9}{'命中时接受':>11}")
    for kind, cnt in r["by_kind"].most_common():
        h = r["by_kind_hit"][kind]
        a = r["by_kind_acc"][kind]
        print(f"  {kind:<16}{cnt:>10,}{100*h/max(1,cnt):>8.1f}%{(a/max(1,h)):>11.2f}")
    print()
    print(f"→ 粗估:若真开 ngram,平均每步可多出 "
          f"{(r['hit']/n)*(r['acc_total']/max(1,r['hit'])):.2f} 个 token/位置")
    print("⚠️ 离线近似,只用于量级判断;真实收益必须上机 A/B ✓")


if __name__ == "__main__":
    main()
