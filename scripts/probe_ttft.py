#!/usr/bin/env python
"""TTFT / prefill-throughput probe against a running OpenAI-compatible server.

Measures **true time-to-first-token** (streaming, first chunk) for a given prompt
length, plus the effective prefill rate. This is the verification client for the
GPU-prefill three switches (see dev-docs/GPU_PREFILL_MAINLINE.md §2.4): with GPU
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
  OUT     jsonl output path (default dev-docs/report/tuning/ttft_<LABEL>.jsonl)
  UNIQUE  1 (默认) = **每次重复都用全新 prompt**(首 token 就不同)⇒ 量的是**真预填充**;
          0 = 复用同一 prompt(旧行为)⇒ 量到的是 **prefix-cache 命中 + 首步**,不是预填充吞吐。

⚠️ **尺子教训 (§591)**:UNIQUE=0 时同一 prompt 的第 2 次起整段 KV 命中缓存,TTFT 会掉到
~600 ms;历史上"8K 预热后只要 ~1 s"就是被这样量出来的假数据。凡是报"预填充吞吐"的行,
必须带 `unique=1` 且 `cached_tokens=0`(本脚本会打印服务端返回的 usage 以便核对)。

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
OUT = os.environ.get("OUT", f"dev-docs/report/tuning/ttft_{LABEL}.jsonl")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# ~4.7 chars/token English filler; content is irrelevant, only length matters.
FILLER = ("Marie Curie was a physicist and chemist who did pioneering research "
          "on radioactivity and discovered polonium and radium. ")


import random

# 每次调用都不一样、且**从第一个词就不同**的填充词表(保证首块哈希不同 ⇒ 不可能命中前缀缓存)。
_SALT_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
               "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
               "xray yankee zulu quark boson lepton gluon photon meson baryon neutron "
               "basalt granite gneiss schist marble quartz feldspar mica talc gypsum "
               "volta ampere faraday tesla henry weber ohm siemens coulomb joule watt").split()


def make_prompt(n_tokens: int, unique: bool = True, salt: str = "") -> str:
    units = max(1, int(n_tokens / 21) + 2)
    body = (FILLER * units)[: int(n_tokens * 4.7)]
    if not unique:
        return body
    # 前缀 = 64 个随机词(≈64-128 token),足以让**第一个 KV 块**与任何历史请求都不同。
    pre = " ".join(random.choice(_SALT_WORDS) for _ in range(64))
    return f"{salt} {pre} || " + body


def emit(**kw):
    kw["label"] = LABEL
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = json.dumps(kw, ensure_ascii=False)
    print("JSON " + line, flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def one(prompt: str, label: str) -> tuple:
    """Return (ttft_s, total_s, n_chunks, prompt_tokens, cached_tokens)."""
    body = json.dumps({
        "model": label,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
        # include_usage ⇒ 能核对"服务端实际算了多少 token、其中多少命中缓存"。
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    ptoks = ctoks = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:  # noqa: BLE001
                obj = {}
            u = obj.get("usage") or {}
            if u:
                ptoks = u.get("prompt_tokens", ptoks)
                d = u.get("prompt_tokens_details") or {}
                ctoks = d.get("cached_tokens", ctoks)
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
    return (ttft if ttft is not None else time.perf_counter() - t0,
            time.perf_counter() - t0, n, ptoks, ctoks)


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

    unique = os.environ.get("UNIQUE", "1") == "1"
    for L in LENS:
        rows = []
        for i in range(REP):
            # UNIQUE ⇒ 每次都新建 prompt(全新 salt + 随机前缀)⇒ 不可能命中前缀缓存
            prompt = make_prompt(L, unique=unique, salt=f"[r{i}]")
            ttft, total, n, ptoks, ctoks = one(prompt, model)
            rows.append((ttft, total, n, i, ptoks, ctoks))
        ttft, total, n, i, ptoks, ctoks = min(rows, key=lambda r: r[0])
        ntok = ptoks if ptoks else L
        emit(kind="ttft", target_len=L, unique=int(unique), chars=len(prompt),
             prompt_tokens=ptoks, cached_tokens=ctoks,
             ttft_s=round(ttft, 3), total_s=round(total, 3), chunks=n, best_rep=i,
             all_ttft_s=[round(r[0], 3) for r in rows],
             prefill_tok_per_s=round(ntok / ttft, 1) if ttft > 0 else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
