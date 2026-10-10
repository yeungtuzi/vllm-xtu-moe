#!/usr/bin/env python3
"""回放 `process_data/data/dsh_assembled_bench.jsonl`(DSH 形状的真实 prompt),
量【当前配置下 dspark 的真实接受率】与延迟。

═══════════════════════════════════════════════════════════════════════════
 为什么
═══════════════════════════════════════════════════════════════════════════
`notes/results.txt` Round 5(2026-09-07)在**同一份** DSH-assembled prompts 上测到:
    dspark5 接受长度 **1.60**(低于 crossover ~2.1)⇒ **净亏**:总吞吐 −17%、中位 TPOT 65.4 → **201.6 ms**
    ⇒ 原文:"The premise that DSH-shaped traffic flips spec +ive is **REFUTED** by this reconstruction."
但那份是 **V4-Flash-0731 + 重构 assembly**,而现在是 **V4.1-Flash + dspark k=5**。
本仓自己写下的正解是:*"measure acceptance on **GENUINE DSH HTTP traffic** … rather than reconstruction"* ✓
⇒ **本脚本就是对活的生产服务做这件事**(零重启)✓

⚠️ 只发 HTTP:`/v1/completions`(prompt 是**预装配好**的,含 `<|im_start|>` ⇒ 必须走 completions,不能走 chat)✓
⚠️ 接受率取 `/metrics` 的**前后增量** ⇒ 不受此前探针污染的绝对计数影响 ✓

用法:
  PORT=8070 CONC=4 python3 scripts/replay_dsh_bench.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import threading
import time

import requests

DS = "/home/user/lvllm/process_data/data/dsh_assembled_bench.jsonl"

COUNTERS = [
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_emitted_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
]


def snap(metrics_url: str) -> dict:
    out = {}
    try:
        txt = requests.get(metrics_url, timeout=30).text
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ metrics 读不到: {e}")
        return out
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for c in COUNTERS:
            if line.startswith(c + "{") or line.startswith(c + " "):
                v = line.rsplit(" ", 1)[-1]
                try:
                    out[c] = out.get(c, 0.0) + float(v)
                except ValueError:
                    pass
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([\d.eE+]+)', line)
        if m:
            out[f"pos{m.group(1)}"] = float(m.group(2))
    return out


def one(url, model, prompt, max_tokens, sink, tag):
    # ⭐⭐ 2026-10-09 修(用户指出我的 A/B 口径错误后查出**两个**测量缺陷):
    #  ① `ignore_eos=True` 在投机路径下**不足以**保证跑满 ⇒ 补 `min_tokens = max_tokens` ✓
    #  ② ⭐⭐ **最致命的一条**:原先用"收到的 SSE chunk 数"当 token 数 —— 而**投机解码下一次 flush
    #     会带 ~A 个 token**(实测 196 chunk × ~5.2 ≈ 1024 token)⇒ 投机的"token 数"被**低估 ~A 倍**,
    #     于是我把"chunk/s"和"token/s"直接比,得出"投机慢 3.8×"的**错误结论** ✗✗
    #     ⇒ 现在改为**读服务端 `usage.completion_tokens`**(stream_options.include_usage)✓
    #     并把 token 数写回 `n`,让所有上层统计口径统一 ✓
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    marks, first, texts = [], None, []
    ntok = None          # ⭐ 服务端口径的真实生成 token 数(优先于 chunk 数)✓
    try:
        with requests.post(url, json=body, stream=True, timeout=1800) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                p = line[6:]
                if p.strip() == b"[DONE]":
                    break
                try:
                    d = json.loads(p)
                except Exception:
                    continue
                ch = (d.get("choices") or [{}])[0].get("text")
                if ch:
                    now = time.time()
                    if first is None:
                        first = now - t0
                    marks.append(now - t0)
                    texts.append(ch)   # ⭐ 保存输出:供"同一批输出的 ngram 复制率"比对
                _u = d.get("usage") or {}
                if _u.get("completion_tokens"):
                    ntok = int(_u["completion_tokens"])   # ⭐ 真实 token 数 ✓

    except Exception as e:  # noqa: BLE001
        sink[tag] = {"error": repr(e)}
        return
    sink[tag] = {"ttft": first, "marks": marks, "text": "".join(texts),
                 # ⭐ 优先用服务端 usage 的真实 token 数;拿不到才退回 chunk 数(并标注)✓
                 "n": int(ntok) if ntok else len(marks),
                 "n_src": "usage" if ntok else "chunks"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8070)))
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--conc", type=int, default=int(os.environ.get("CONC", 4)))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", 0)), help="0=全部 50 条")
    ap.add_argument("--dataset", default=DS)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    url = base + "/v1/completions"
    murl = base + "/metrics"

    recs = [json.loads(l) for l in open(args.dataset, encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    print(f"数据集: {args.dataset}  ({len(recs)} 条)")
    print(f"并发 {args.conc} · max_tokens={recs[0].get('output_tokens', 192)} · 走 /v1/completions(预装配 prompt)")

    m0 = snap(murl)
    print(f"metrics 基线: drafts={m0.get('vllm:spec_decode_num_drafts_total')} "
          f"accepted={m0.get('vllm:spec_decode_num_accepted_tokens_total')}")
    print()

    sink: dict = {}
    lock = threading.Semaphore(args.conc)
    ths = []
    t_start = time.time()
    for i, r in enumerate(recs):
        def job(i=i, r=r):
            with lock:
                one(url, args.model, r["prompt"], int(r.get("output_tokens", 192)), sink, f"r{i}")
        th = threading.Thread(target=job, daemon=True); th.start(); ths.append(th)
    for th in ths:
        th.join()
    wall = time.time() - t_start
    m1 = snap(murl)

    ok = [v for v in sink.values() if v.get("marks")]
    itls, ttfts, ntok, nchunks = [], [], 0, 0
    for v in ok:
        m = v["marks"]
        ntok += int(v.get("n") or 0)        # ⭐ 服务端 usage 的真实 token 数 ✓
        nchunks += len(m)                   # chunk 数(诊断用:投机下 ≠ token 数)
        if v.get("ttft"):
            ttfts.append(v["ttft"])
        itls += [(m[j] - m[j - 1]) * 1000.0 for j in range(1, len(m))]

    d = {k: (m1.get(k, 0) - m0.get(k, 0)) for k in set(list(m0) + list(m1))}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    dtt = d.get("vllm:spec_decode_num_draft_tokens_total", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    emitted = d.get("vllm:num_emitted_tokens_total", 0)

    print(f"完成 {len(ok)}/{len(recs)} 条 · 墙钟 {wall:.1f}s")
    print(f"  生成 token {ntok}(服务端 usage) · 输出吞吐 {ntok/wall:.2f} tok/s")
    if ntok and nchunks:
        _tt = statistics.median(ttfts) if ttfts else 0.0
        _dec = max(0.1, wall - _tt)
        print(f"  ⭐ 解码段速率 {ntok/_dec:.2f} tok/s(排除 TTFT {_tt:.2f}s;"
              f" {ntok} token / {nchunks} chunk ⇒ 每 chunk **{ntok/max(1,nchunks):.2f} token**"
              f" —— 非投机应≈1.00,投机应≈接受长度)✓")
    if ttfts:
        print(f"  TTFT  中位 {statistics.median(ttfts):.2f}s  (mean {statistics.mean(ttfts):.2f}s)")
    if itls:
        print(f"  ITL   中位 {statistics.median(itls):.1f}ms  (mean {statistics.mean(itls):.1f}ms)")
    print()
    print("=== ⭐ dspark 接受率(本次回放的增量)===")
    print(f"  drafts={drafts:.0f} · draft_tokens={dtt:.0f} · accepted={acc:.0f} · emitted={emitted:.0f}")
    if drafts:
        print(f"  每步草稿   = {dtt/drafts:.2f}")
        print(f"  接受率     = {100*acc/max(1,dtt):.2f}%   (accepted/draft_tokens)")
        print(f"  ⭐ 接受长度 = {1 + acc/drafts:.2f} token/步   ← Round5 的 DSH 基线是 1.60")
        print(f"  每步落地   = {emitted/max(1,drafts):.2f} token/步")
    pos = [d.get(f"pos{i}", 0) for i in range(5)]
    if drafts and sum(pos) >= 0:
        print(f"  逐位接受   = " + " ".join(f"pos{i}:{100*p/max(1,drafts):.1f}%" for i, p in enumerate(pos)))
    pcq = d.get("vllm:prefix_cache_queries_total", 0)
    pch = d.get("vllm:prefix_cache_hits_total", 0)
    if pcq:
        print(f"  前缀缓存命中率 = {100*pch/pcq:.1f}%  (queries={pcq:.0f})")
    # ⭐ 落盘 {prompt, output} 供同源 ngram 复制率比对
    outp = os.environ.get("OUT") or "/tmp/dsh_bench_outputs.jsonl"
    with open(outp, "w", encoding="utf-8") as f:
        for i, r in enumerate(recs):
            v = sink.get(f"r{i}", {})
            f.write(json.dumps({"prompt": r["prompt"], "output": v.get("text", ""),
                                "tokens": v.get("n"), "ttft": v.get("ttft")}, ensure_ascii=False) + "\n")
    print(f"✅ 已落盘 {{prompt,output}} → {outp}(供同源 ngram 复制率)✓")
    print()
    print("⚠️ 单次回放;要下结论请与离线 ngram 复制率【同源】比较,并重复一轮 ✓")


if __name__ == "__main__":
    main()
