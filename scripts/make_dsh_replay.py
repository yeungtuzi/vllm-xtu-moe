#!/usr/bin/env python
"""Rebuild the **DSH session-replay** prefill dataset from *this machine's* session log.

Why this exists
---------------
ShareGPT's average prompt is only ~227 tokens, which is far below the GPU-prefill
switch (`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=4096`), so a ShareGPT benchmark can
never exercise the long-prefill path. A realistic long-prompt workload is a
**replay of an actual agent session**: agentic coding turns are thousands of
tokens, which is exactly the regime GPU prefill targets.

The generated data is *this session's conversation text*, so it is **never
committed** (see `.gitignore`). Rebuild it locally when needed.

Usage
-----
    python scripts/make_dsh_replay.py                  # newest session, 2K..32K
    python scripts/make_dsh_replay.py --session <path/to/session.v3.jsonl.zstd>
    python scripts/make_dsh_replay.py --lens 2048,4096,8192 --outdir report/tuning/replay

Outputs, per requested length N:  ``dsh_<N>.json`` (ShareGPT-style) and
``dsh_<N>.jsonl`` (``{"prompt": ..., "output_tokens": 128}`` for
``vllm bench serve --dataset-name custom``).

NOTE the label: this is **a DSH dev-session replay, not ShareGPT**. Any figure or
table built from it must say so.

License: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

SESSIONS_ROOT = os.path.expanduser("~/.dsh/sessions")
OUT_TOKENS = 128


def newest_session() -> str:
    """Newest ``session.v3.jsonl.zstd`` under ``~/.dsh/sessions``."""
    best, best_m = None, -1.0
    for root, _dirs, files in os.walk(SESSIONS_ROOT):
        for fn in files:
            if fn.startswith("session.v3") and fn.endswith(".jsonl.zstd"):
                p = os.path.join(root, fn)
                m = os.path.getmtime(p)
                if m > best_m:
                    best, best_m = p, m
    if best is None:
        raise SystemExit(f"no session.v3.jsonl.zstd found under {SESSIONS_ROOT}")
    return best


def load_turns(path: str) -> list[list]:
    """[(user_text, [assistant/tool texts]), ...] in transcript order."""
    raw = subprocess.run(["zstd", "-d", "-c", path], capture_output=True, check=True)
    turns: list[list] = []
    cur: list | None = None
    for line in raw.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ, data = rec.get("type"), rec.get("data") or {}
        if typ == "user/message" and (data.get("source") or {}).get("kind") == "user":
            text = "\n".join(
                c.get("text", "") for c in (data.get("content") or []) if c.get("type") == "text"
            )
            if text.strip():
                cur = [text, []]
                turns.append(cur)
        elif cur is not None and typ == "assistant/message":
            for c in (data.get("message") or {}).get("content") or []:
                if c.get("type") == "text" and c.get("text"):
                    cur[1].append(c["text"])
                elif c.get("type") == "tool-call":
                    cur[1].append(f"[tool-call {c.get('name')}] {str(c.get('arguments'))[:2000]}")
        elif cur is not None and typ == "tool/result":
            body = data.get("content") or data.get("result") or data
            body = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
            cur[1].append("[tool-result] " + body[:4000])
    return turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="path to session.v3.jsonl.zstd")
    ap.add_argument("--lens", default="2048,4096,8192,16384,32768")
    ap.add_argument("--outdir", default="report/tuning/replay")
    ap.add_argument("--turns", type=int, default=40, help="how many trailing turns to concatenate")
    ap.add_argument("--tokenizer", default=os.environ.get("DSH_REPLAY_TOKENIZER", ""))
    args = ap.parse_args()

    path = args.session or newest_session()
    turns = load_turns(path)
    if not turns:
        raise SystemExit("no user turns found")
    body: list[str] = []
    for user, rest in turns[-args.turns :]:
        body.append(f"=== USER ===\n{user}")
        for item in rest:
            body.append(f"=== ASSISTANT/TOOL ===\n{item}")
    full = "\n\n".join(body)
    print(f"session={path}\nturns={len(turns)} chars={len(full)}", file=sys.stderr)

    tok_path = args.tokenizer
    if not tok_path:
        raise SystemExit(
            "set --tokenizer (or DSH_REPLAY_TOKENIZER) to the model snapshot; token lengths "
            "must be exact for a length sweep"
        )
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    ids = tk.encode(full)
    print(f"full_tokens={len(ids)}", file=sys.stderr)

    os.makedirs(args.outdir, exist_ok=True)
    written = 0
    for n in (int(v) for v in args.lens.split(",") if v):
        if len(ids) < n:
            print(f"  skip {n}: only {len(ids)} tokens available", file=sys.stderr)
            continue
        text = tk.decode(ids[-n:])  # keep the most recent context
        entry = {
            "id": f"dsh_replay_{n}",
            "conversations": [
                {"from": "human", "value": text},
                {"from": "gpt", "value": "ok"},
            ],
        }
        with open(os.path.join(args.outdir, f"dsh_{n}.json"), "w") as fh:
            json.dump([entry], fh, ensure_ascii=False)
        with open(os.path.join(args.outdir, f"dsh_{n}.jsonl"), "w") as fh:
            fh.write(json.dumps({"prompt": text, "output_tokens": OUT_TOKENS}, ensure_ascii=False))
            fh.write("\n")
        written += 1
        print(f"  {n}: ok", file=sys.stderr)
    print(f"{args.outdir}: {written} lengths written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
