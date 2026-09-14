#!/usr/bin/env python
"""TTFT / prefill-throughput probe against a running OpenAI-compatible server.

Measures **true time-to-first-token** (streaming, first chunk) for a given prompt
length, plus the effective prefill rate. This is the verification client for the
GPU-prefill three switches (see docs/GPU_PREFILL_MAINLINE.md §2.4): with GPU
prefill off, long prompts are served by the CPU engine at ~45 t/s and TTFT grows
linearly; with it on, TTFT is dominated by a ~constant weight-DMA term.

Usage:
  python scripts/probe_ttft.py                       # 512 on :8071
  PORT=8071 LENS=256,1024,2048 python scripts/probe_ttft.py
  LABEL=gpu python scripts/probe_ttft.py             # tag rows for A/B

Env:
  PORT    server port (default 8071)
  LENS    comma list of target prompt token counts (default 256,1024,2048)
  LABEL   free-form label copied into every row (for A/B comparison)
  REP     repetitions per length, keeps the best (default 1)
  OUT     jsonl output path (default report/tuning/ttft_<LABEL>.jsonl)

Emits one JSON object per line (also appended to OUT).

License: Apache-2.0
"""

import json
import os
import sys
import time
import urllib.request

PORT = int(os.environ.get("PORT", "8071"))
LENS = [int(v) for v in os.environ.get("LENS", "256,1024,2048").split(",") if v]
LABEL = os.environ.get("LABEL", "na")
REP = int(os.environ.get("REP", "1"))
OUT = os.environ.get("OUT", f"report/tuning/ttft_{LABEL}.jsonl")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# ~4.7 chars/token English filler; content is irrelevant, only length matters.
FILLER = ("Marie Curie was a physicist and chemist who did pioneering research "
          "on radioactivity and discovered polonium and radium. ")


def make_prompt(n_tokens: int) -> str:
    units = max(1, int(n_tokens / 21) + 2)
    return (FILLER * units)[: int(n_tokens * 4.7)]


def emit(**kw):
    kw["label"] = LABEL
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = json.dumps(kw, ensure_ascii=False)
    print("JSON " + line, flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def one(prompt: str, label: str) -> tuple:
    """Return (ttft_s, total_s, n_chunks) for one streaming request."""
    body = json.dumps({
        "model": label,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
    return (ttft if ttft is not None else time.perf_counter() - t0,
            time.perf_counter() - t0, n)


def main() -> int:
    print(f"[ttft] port={PORT} lens={LENS} label={LABEL} rep={REP} out={OUT}",
          flush=True)
    model = LABEL
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/v1/models", timeout=30) as r:
            model = json.loads(r.read())["data"][0]["id"]
    except Exception as e:  # noqa: BLE001
        print(f"[ttft] cannot reach server: {type(e).__name__}: {e}", flush=True)
        return 1
    print(f"[ttft] model={model}", flush=True)

    for L in LENS:
        prompt = make_prompt(L)
        best = None
        for i in range(REP):
            ttft, total, n = one(prompt, model)
            if best is None or ttft < best[0]:
                best = (ttft, total, n, i)
        ttft, total, n, i = best
        # Prompt length in tokens is unknown client-side; report the char count
        # so the server-side tokenizer count can be cross-checked.
        emit(kind="ttft", target_len=L, chars=len(prompt), ttft_s=round(ttft, 3),
             total_s=round(total, 3), chunks=n, best_rep=i,
             prefill_tok_per_s=round(L / ttft, 1) if ttft > 0 else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
