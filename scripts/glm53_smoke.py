#!/usr/bin/env python
"""Smoke/measurement client for a running GLM-5.3-Flash OpenAI endpoint.

Sends a few greedy prompts (streaming) and reports, per request:
  * the generated text (first line, to eyeball coherence),
  * TTFT (time to first token),
  * TPOT (mean inter-token time after the first),
  * output tok/s including TTFT (the same definition as `vllm bench serve`).

Usage:
  python scripts/glm53_smoke.py --port 8073
  python scripts/glm53_smoke.py --port 8073 --max-tokens 64 --prompts 5
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

DEFAULT_PROMPTS = [
    "The capital of France is",
    "1+1=",
    "Write a haiku about autumn.",
    "In one sentence, explain why the sky is blue.",
    "List the first five prime numbers, separated by commas.",
]


def stream_completion(port: str, prompt: str, max_tokens: int, model: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    pieces = []
    ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for ch in obj.get("choices", []):
                txt = ch.get("text")
                if txt:
                    if first is None:
                        first = time.perf_counter() - t0
                    pieces.append(txt)
                    ntok += 1
    total = time.perf_counter() - t0
    return {
        "text": "".join(pieces),
        "ttft": first if first is not None else float("nan"),
        "total": total,
        "ntok": ntok,
        "tpot": (total - first) / max(1, ntok - 1) if first is not None and ntok > 1 else float("nan"),
        "out_tok_s": ntok / total if total > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8073")
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--prompts", type=int, default=len(DEFAULT_PROMPTS))
    args = ap.parse_args()

    print(f"endpoint 127.0.0.1:{args.port}  model={args.model}  max_tokens={args.max_tokens}")
    print(f"{'#':>2} {'TTFT ms':>9} {'TPOT ms':>9} {'out tok/s':>10} {'ntok':>5}  text")
    for i, p in enumerate(DEFAULT_PROMPTS[: args.prompts]):
        r = stream_completion(args.port, p, args.max_tokens, args.model)
        head = r["text"].replace("\n", " ")[:70]
        print(
            f"{i:>2} {r['ttft']*1e3:>9.1f} {r['tpot']*1e3:>9.2f} "
            f"{r['out_tok_s']:>10.2f} {r['ntok']:>5}  {head}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
