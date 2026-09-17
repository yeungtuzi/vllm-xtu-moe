#!/usr/bin/env python3
"""③ CED 预填充捷径的 **A/B 验收工具**(生成口径)。

为什么是生成口径:`--kv-sharing-fast-prefill` 模式下 runner 有硬断言
(`--kv-sharing-fast-prefill produces incorrect logprobs for prompt tokens`)⇒ §572 建的
`prompt_logprobs` 黄金基线**不能**用于该模式。而 §572 已实测"**首个生成 token / 贪心文本
在 native prefill 噪声下始终稳定**"⇒ 用贪心生成文本做等价判据。

同一份语料(5 条既有 + 3 条确定性长 prompt ≈300/1024/4096 token,只有长 prompt 才触发 CED),
两次运行(toggle 关闭 / 开启)各跑一遍,记录 **生成文本 + 耗时**(耗时是预填充收益的直接信号)。

用法:
  python3 report/tuning/probes/ced_ab.py --port 8320 --out /tmp/ced_off.json      # 基线
  python3 report/tuning/probes/ced_ab.py --port 8321 --check /tmp/ced_off.json    # 对照
License: Apache-2.0
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from capture_prefill_golden import _cases  # noqa: E402  (同一份语料,避免两处漂移)


def _post(port, path, payload, timeout=1800.0):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _run(port, model, ntok):
    out = {}
    for cid, prompt in _cases():
        t0 = time.time()
        try:
            r = _post(port, "/v1/completions", {
                "model": model, "prompt": prompt, "max_tokens": ntok,
                "temperature": 0.0, "ignore_eos": True,
            })
            ch = r["choices"][0]
            out[cid] = {"text": ch.get("text", ""),
                        "secs": round(time.time() - t0, 2),
                        "prompt_chars": len(prompt),
                        "usage": r.get("usage", {})}
            print(f"[ced-ab] {cid}: {out[cid]['secs']:8.2f}s  "
                  f"chars={len(prompt):6d}  {out[cid]['text'][:40]!r}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[ced-ab] {cid}: FAILED {type(exc).__name__}: {exc}", flush=True)
            out[cid] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="dsv41-xtu")
    ap.add_argument("--ntok", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        base = json.load(open(args.out))
        cur = _run(args.port, args.model, args.ntok)
        bad = 0
        print("\n=== 生成文本等价性(CED 关 vs CED 开)===")
        for cid in sorted(set(base) | set(cur)):
            b, c = base.get(cid, {}), cur.get(cid, {})
            same = b.get("text") == c.get("text")
            tb, tc = b.get("secs"), c.get("secs")
            sp = (tc / tb) if (tb and tc) else None
            tag = "PASS" if same else "FAIL"
            if not same:
                bad += 1
            print(f"   {tag} {cid}: 文本{'相同' if same else '**不同**'}  "
                  f"耗时 {tb}s → {tc}s" + (f" (×{sp:.2f})" if sp else ""))
            if not same:
                print(f"        基线: {b.get('text','')[:80]!r}")
                print(f"        实验: {c.get('text','')[:80]!r}")
        print(f"\n[ced-ab] {'全部一致' if bad == 0 else f'{bad} 项不一致'}")
        return 0 if bad == 0 else 1

    res = _run(args.port, args.model, args.ntok)
    with open(args.out, "w") as fh:
        json.dump(res, fh)
    print(f"[ced-ab] 写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
