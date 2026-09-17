#!/usr/bin/env python3
"""捕获 **CED 预填充捷径的自等价黄金基线**(§572)。

为什么需要它:③ 的"预填充只算前一半层 + 回放最后 2·w_win−1 个位置"与今天的路径**数学等价**
(论证见 NOTES §571b),所以**验收不需要 lk_moe 做对照**,只需要"改完之后输出与今天逐位一致"。
这个脚本负责把"今天"钉死成一个可复现的黄金文件;`check_ced_prefill.py` 负责改完之后比对。

测什么:
* **prefill logits 指纹** —— 对每个 prompt 取 `prompt_logprobs=0`,记录**每个位置**上模型给
  实际 token 的 logprob(这是对 prefill 全部 KV/隐藏状态的敏感指纹);
* **首个生成 token**(temperature=0)及其 top logprobs。

长度很关键:CED 只在 `T > 2·w_win−1 = 255` 时才有区别;短 prompt 天然一致 ⇒ 语料里**必须**有长 prompt。
语料 = `report/tuning/correctness_prompts.json`(5 条)+ 3 条**确定性合成长 prompt**
(重复同一句,长度约 300/1024/4096 token)。

用法:
  python3 report/tuning/probes/capture_prefill_golden.py --port 8316 --out report/tuning/ced_prefill_golden.json
  python3 report/tuning/probes/capture_prefill_golden.py --port 8316 --check    # 只比对不覆盖
License: Apache-2.0
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

# __file__ = <repo>/report/tuning/probes/xxx.py ⇒ 要上溯 4 层才是仓库根
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
PROMPTS = os.path.join(ROOT, "report/tuning/correctness_prompts.json")

_SENT = ("The lambda calculus is a formal system in mathematical logic for expressing "
         "computation based on function abstraction and application using variable "
         "binding and substitution. ")


def _post(port, path, payload, timeout=900.0):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _cases():
    """(id, prompt) 列表:短(既有语料) + 长(合成、确定性)。"""
    out = []
    if os.path.exists(PROMPTS):
        for p in json.load(open(PROMPTS)):
            txt = p.get("messages", [{}])[0].get("content") or p.get("prompt", "")
            if txt:
                out.append((f"corr_{p['id']}", txt))
    for n_tok in (300, 1024, 4096):
        # _SENT 约 25 token ⇒ 重复次数由目标长度反推
        out.append((f"long_{n_tok}", _SENT * max(4, n_tok // 25)))
    return out


def _fingerprint(port, model, prompt, skip=0):
    r = _post(port, "/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": 1, "ignore_eos": True,
        "temperature": 0.0, "prompt_logprobs": 0, "logprobs": 1,
    })
    ch = r["choices"][0]
    pls = ch.get("prompt_logprobs") or []
    lps, toks = [], []
    for i, e in enumerate(pls):
        if i < skip or not e:
            continue
        for tid, info in e.items():
            if isinstance(info, dict) and "logprob" in info:
                lps.append(float(info["logprob"]))
                toks.append(int(tid))
                break
    lp = ch.get("logprobs") or {}
    return {"n": len(lps), "logprobs": lps, "token_ids": toks,
            "first_text": ch.get("text", ""),
            "first_tokens": (lp.get("tokens") or [])[:1],
            "first_token_logprobs": (lp.get("token_logprobs") or [])[:1],
            "finish_reason": ch.get("finish_reason"),
            "usage": r.get("usage", {})}


def _capture(port, model):
    res = {}
    for cid, prompt in _cases():
        t0 = time.time()
        try:
            res[cid] = _fingerprint(port, model, prompt)
            res[cid]["prompt_chars"] = len(prompt)
            print(f"[golden] {cid}: n={res[cid]['n']} tok, "
                  f"first={res[cid]['first_text'][:24]!r}, {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[golden] {cid}: FAILED {type(exc).__name__}: {exc}", flush=True)
    return res


def _compare(a, b) -> int:
    bad = 0
    for cid in sorted(set(a) | set(b)):
        if cid not in a or cid not in b:
            print(f"   !! {cid}: 只在一边存在"); bad += 1; continue
        x, y = a[cid], b[cid]
        if x["n"] != y["n"]:
            print(f"   FAIL {cid}: 位置数 {x['n']} vs {y['n']}"); bad += 1; continue
        d = [abs(p - q) for p, q in zip(x["logprobs"], y["logprobs"])]
        mx = max(d) if d else 0.0
        same_tok = x["token_ids"] == y["token_ids"]
        same_txt = x["first_text"] == y["first_text"]
        same_first = (x.get("first_tokens") == y.get("first_tokens"))
        same_lp = (x.get("first_token_logprobs") == y.get("first_token_logprobs"))
        ok = (mx == 0.0) and same_tok and same_txt and same_first and same_lp
        print(f"   {'PASS' if ok else 'FAIL'} {cid}: n={x['n']} max|Δlogprob|={mx:.3e} "
              f"tokens={'同' if same_tok else '不同'} 首文本={'同' if same_txt else '不同'} "
              f"首token={'同' if same_first else '不同'} 首logprob={'同' if same_lp else '不同'}")
        if not ok:
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="dsv41-xtu")
    ap.add_argument("--out", default=os.path.join(ROOT, "report/tuning/ced_prefill_golden.json"))
    ap.add_argument("--check", action="store_true", help="与已有黄金文件比对(不覆盖)")
    args = ap.parse_args()

    if args.check:
        if not os.path.exists(args.out):
            print(f"[golden] 没有黄金文件 {args.out}", file=sys.stderr); return 2
        gold = json.load(open(args.out))
        print(f"[golden] 比对基线(捕获于 {gold.get('meta', {}).get('captured_at', '?')}, "
              f"config={gold.get('meta', {}).get('config', '?')})")
        bad = _compare(gold["cases"], _capture(args.port, args.model))
        print(f"[golden] {'一致' if bad == 0 else f'不一致:{bad} 项'}")
        return 0 if bad == 0 else 1

    cases = _capture(args.port, args.model)
    doc = {
        "meta": {
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "CED 预填充捷径(§571/§572)的自等价黄金基线:改完之后必须与此逐位一致",
            "config": os.environ.get("CED_GOLDEN_CONFIG", "(未标)"),
            "tolerance": "max|Δlogprob| 必须为 0.0 且 token_ids/首文本完全相同;"
                         "若实现上不可避免浮点重排,才退到数值门 1.873e-02",
            "window_note": "CED 只在 T > 2*sliding_window-1 = 255 时才有区别",
        },
        "cases": cases,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh)
    print(f"[golden] 写入 {args.out}({len(cases)} 例,{os.path.getsize(args.out)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
