#!/usr/bin/env python
"""Concurrency test against a live vLLM server (the realistic continuous-batching
path, unlike the offline LLM API which steps before all prompts are queued).

Fires C requests of L tokens at once and reports wall time + aggregate prefill
throughput. Env: PORT (8000), LEN (2048), CONC (1,2,4,8), MODEL name, OUT jsonl.
"""
import concurrent.futures as cf
import json
import os
import time
import urllib.request

PORT = int(os.environ.get("PORT", "8000"))
LEN = int(os.environ.get("LEN", "2048"))
CONC = [int(v) for v in os.environ.get("CONC", "1,2,4,8").split(",")]
MODEL = os.environ.get("MODEL", "DeepSeek-V4-Flash-xiaotu")
OUT = os.environ.get("OUT", "server_conc.jsonl")

FILLER = ("Marie Curie was a physicist and chemist who did pioneering research "
          "on radioactivity and discovered polonium and radium. ")


def make_prompt(n_tokens):
    units = max(1, int(n_tokens / 22) + 2)
    return (FILLER * units)[: int(n_tokens * 5.9)]


def one(prompt):
    # NOTE: prompts must differ from the FIRST token on, otherwise vLLM's prefix
    # cache (enabled by default) serves them from cache and the measurement
    # would report cache-hit speed, not prefill throughput.
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "temperature": 0.0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        json.loads(r.read())
    return time.time() - t0


def main():
    base = make_prompt(LEN)
    for c in CONC:
        prompts = [f"Request {i} of batch {c}, unique-tag {time.time_ns()}: " + base
                   for i in range(c)]
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=c) as ex:
            lat = list(ex.map(one, prompts))
        dt = time.time() - t0
        ntok = c * LEN
        rec = {"kind": "server_concurrency", "conc": c, "tokens": LEN,
               "total_tokens": ntok, "wall_s": round(dt, 3),
               "tok_per_s": round(ntok / dt, 1),
               "max_lat_s": round(max(lat), 3),
               "min_lat_s": round(min(lat), 3)}
        print("JSON " + json.dumps(rec), flush=True)
        print(f"[srv] c={c} x {LEN} tok: wall {dt:.2f}s "
              f"-> {ntok/dt:.0f} tok/s aggregate, latency "
              f"{min(lat):.2f}~{max(lat):.2f}s", flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
