#!/usr/bin/env python
"""目标场景客户端:自然文本 + 精确上下文长度 + 逐流延迟。

为什么不用 `vllm bench serve`:
1. `--dataset-name custom` 需要 pandas(本环境 py3.12 没有,只有 3.10 的旧 wheel);
2. `--dataset-name random` 的随机 token 会把 draft 接受率打到地板(见 NOTES §35.6),
   完全不能代表目标场景;
3. 我们要的核心指标是**单路 token 间延迟**(可交互性),自己算最直接。

用法:
  L=1024 C=1 N=8 OUT=128 python scripts/bench_nat_client.py
  L=1024 C=2 N=8 OUT=128 TAG=nat1024_c2
输出:report/tuning/raw/<TAG>.json + 一行进 summary.jsonl(natural_text=1, client=self)
"""
import asyncio
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "report/tuning/raw")
os.makedirs(RAW, exist_ok=True)

L = int(os.environ.get("L", "1024"))
C = int(os.environ.get("C", "1"))
N = int(os.environ.get("N", "8"))
OUT = int(os.environ.get("OUT", "128"))
PORT = os.environ.get("PORT", "8070")
MODEL = os.environ.get("MODEL", "DeepSeek-V4-Flash-xiaotu")
SERVER_TAG = os.environ.get("SERVER_TAG", "unknown")
TAG = os.environ.get("TAG", f"nat{L}_c{C}_n{N}_out{OUT}")
DS = os.environ.get("DS", os.path.join(ROOT, f"report/tuning/datasets/nat{L}.jsonl"))

import aiohttp  # noqa: E402


async def one(session, sem, prompt, idx, results):
    # 注意:投机解码下 vLLM 每个 decode **步**只发一个 SSE chunk(该步验收的多个
    # token 一起送出)。所以 chunk 数 = 步数,token 数必须从 usage 里取
    # (stream_options.include_usage)。把 chunk 当 token 会把吞吐低估 (1+k_accept)/1 倍。
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": OUT,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with sem:
        t0 = time.perf_counter()
        ttft = None
        marks = []
        nchunk = 0
        ntok = 0
        try:
            async with session.post(
                f"http://127.0.0.1:{PORT}/v1/completions", json=body
            ) as r:
                if r.status != 200:
                    results.append({"idx": idx, "error": f"http {r.status} {await r.text()}"})
                    return
                async for raw in r.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                    except Exception:
                        continue
                    ch = j.get("choices") or []
                    if j.get("usage"):
                        ntok = int(j["usage"].get("completion_tokens") or ntok)
                    if ch and ch[0].get("text"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        nchunk += 1
                        marks.append(now)
        except Exception as e:  # noqa: BLE001
            results.append({"idx": idx, "error": repr(e)})
            return
        t1 = time.perf_counter()
        itl = [b - a for a, b in zip(marks, marks[1:])]
        if not ntok:
            ntok = nchunk  # 兜底:没有 usage 就退回 chunk 计数(会低估)
        dec = (t1 - t0) - (ttft or 0)
        results.append(
            {
                "idx": idx,
                "ttft_ms": (ttft or 0) * 1e3,
                "n_out": ntok,
                "n_steps": nchunk,
                "tokens_per_step": (ntok / nchunk) if nchunk else None,
                "e2e_s": t1 - t0,
                "decode_s": dec,
                "mean_step_ms": (statistics.fmean(itl) * 1e3) if itl else None,
                "mean_itl_ms": (dec * 1e3 / ntok) if ntok else None,
                "p50_step_ms": (statistics.median(itl) * 1e3) if itl else None,
                "p95_step_ms": (sorted(itl)[int(0.95 * len(itl)) - 1] * 1e3) if len(itl) > 2 else None,
                "tok_per_s": ntok / dec if dec > 0 else None,
            }
        )


async def main():
    prompts = [json.loads(l)["prompt"] for l in open(DS)][:N]
    sem = asyncio.Semaphore(C)
    results = []
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        await asyncio.gather(*[one(s, sem, p, i, results) for i, p in enumerate(prompts)])
    dur = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    tot_out = sum(r["n_out"] for r in ok)
    agg = tot_out / dur
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": TAG,
        "server_tag": SERVER_TAG,
        "natural_text": 1,
        "client": "bench_nat_client",
        "concurrency": C,
        "num_prompts": len(prompts),
        "output_len": OUT,
        "input_len": f"nat{L}",
        "completed": len(ok),
        "failed": len(bad),
        "duration_s": round(dur, 2),
        "total_output_tokens": tot_out,
        "out_tok_per_s": round(agg, 2),
        "per_stream_tok_per_s": round(agg / C, 2),
        "mean_ttft_ms": round(statistics.fmean([r["ttft_ms"] for r in ok]), 1) if ok else None,
        "mean_tpot_ms": round(statistics.fmean([r["mean_itl_ms"] for r in ok if r["mean_itl_ms"]]), 2) if ok else None,
        "mean_step_ms": round(statistics.fmean([r["mean_step_ms"] for r in ok if r["mean_step_ms"]]), 2) if ok else None,
        "p95_step_ms": round(statistics.fmean([r["p95_step_ms"] for r in ok if r["p95_step_ms"]]), 2) if ok else None,
        "mean_tokens_per_step": round(statistics.fmean([r["tokens_per_step"] for r in ok if r["tokens_per_step"]]), 3) if ok else None,
        "per_req_tok_per_s": [round(r["tok_per_s"], 2) for r in ok if r["tok_per_s"]],
    }
    with open(os.path.join(RAW, f"{TAG}.json"), "w") as f:
        json.dump({"summary": rec, "requests": results}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(ROOT, "report/tuning/summary.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"[nat-client] {TAG}: {rec['out_tok_per_s']} tok/s "
        f"({rec['per_stream_tok_per_s']}/stream) ttft {rec['mean_ttft_ms']} ms "
        f"tpot {rec['mean_tpot_ms']} ms step {rec['mean_step_ms']} ms "
        f"({rec['mean_tokens_per_step']} tok/step) "
        f"per-req {rec['per_req_tok_per_s']}"
    )
    if bad:
        print(f"[nat-client] failures: {bad[:2]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
