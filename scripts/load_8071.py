#!/usr/bin/env python
"""Simple concurrent load check against the 8071 test server (no external deps).

Env: PORT (8071), N (32), MAXTOK (32), PROMPT.
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = int(os.environ.get("PORT", "8071"))
N = int(os.environ.get("N", "32"))
MAXTOK = int(os.environ.get("MAXTOK", "32"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
MODEL = os.environ.get("SERVED_NAME", "DeepSeek-V4-Flash-xiaotu")
URL = f"http://127.0.0.1:{PORT}/v1/completions"


def one(_i: int) -> tuple[int, str]:
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": MAXTOK,
            "temperature": 0,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.load(resp)
    ch = d["choices"][0]
    return d["usage"]["completion_tokens"], ch["text"]


def main() -> None:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        outs = list(ex.map(one, range(N)))
    dt = time.time() - t0
    total = sum(n for n, _ in outs)
    print(f"[load] N={N} {total} tok in {dt:.2f}s -> {total/max(dt,1e-9):.2f} tok/s")
    print(f"[load] sample text: {outs[0][1][:100]!r}")


if __name__ == "__main__":
    main()
