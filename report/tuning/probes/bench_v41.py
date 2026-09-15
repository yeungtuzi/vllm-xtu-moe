#!/usr/bin/env python
"""DeepSeek-V4.1-Flash 单机性能基准(纯 stdlib,不依赖额外包)。

用法:
    python bench_v41.py <port> <label> [--conc N] [--maxtok M] [--prompt P]

测量:
  * 单请求延迟(预热 + 计时),拆出 prefill/decode 的粗粒度观察;
  * N 路并发吞吐(tok/s);
  * 所有请求 temperature=0,保证可比。
"""
import json
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8108"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"


def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


CONC = arg("--conc", 1)
MAXTOK = arg("--maxtok", 32)
PROMPT = arg("--prompt", "The capital of France is")
REQS = arg("--reqs", CONC)

URL = f"http://127.0.0.1:{PORT}/v1/completions"


def one(prompt, maxtok):
    body = json.dumps(
        {"model": "dsv41", "prompt": prompt, "max_tokens": maxtok, "temperature": 0}
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    return dt, ct, d["choices"][0]["text"]


print(f"=== bench label={LABEL} port={PORT} conc={CONC} reqs={REQS} maxtok={MAXTOK} ===")

# --- 预热(第一次会把 CPU 引擎的页激活)---
dt, ct, txt = one(PROMPT, 4)
print(f"warmup      : {dt:7.2f}s  {ct:3d} tok   text={txt!r}")

# --- 单请求:decode 为主 ---
for mt in (MAXTOK, MAXTOK * 2):
    dt, ct, txt = one(PROMPT, mt)
    print(f"single {mt:4d}tok: {dt:7.2f}s  {ct:3d} tok  {ct / dt:7.2f} tok/s  text={txt[:40]!r}")

# --- 并发吞吐 ---
if CONC > 1:
    res = []
    lock = threading.Lock()

    def worker():
        dt, ct, _ = one(PROMPT, MAXTOK)
        with lock:
            res.append((dt, ct))

    ts = [threading.Thread(target=worker) for _ in range(REQS)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    tot = sum(c for _, c in res)
    print(
        f"conc={CONC:2d}      : wall={wall:7.2f}s  total={tot:4d} tok  "
        f"aggregate={tot / wall:7.2f} tok/s   per-req avg={sum(d for d, _ in res) / len(res):6.2f}s"
    )
