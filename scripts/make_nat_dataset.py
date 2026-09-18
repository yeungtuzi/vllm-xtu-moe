#!/usr/bin/env python
"""生成**自然文本 + 精确上下文长度**的基准数据集。

动机(dev-docs/report/tuning/NOTES.md §36):之前的单路基线用 sharegpt 的短 prompt
(实测只有 ~21 token/条),而 `--dataset-name random` 的随机 token 会让 draft
接受率崩塌(位置0 由 0.67 掉到 0.35)。两者都不是目标场景。
本脚本从 ShareGPT 的真实对话里拼出**指定 token 长度**的 prompt,
写到 dev-docs/report/tuning/datasets/nat<LEN>.jsonl(每行 {"prompt": ...}),
供 `vllm bench serve --dataset-name custom --dataset-path <file>` 使用。

用法:LENS="128 512 1024 4096" N=8 scripts/make_nat_dataset.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SG = os.environ.get("SG", "/home/user/lvllm/ShareGPT_V3_unfiltered_cleaned_split.json")
TOK = os.environ.get(
    "TOKENIZER",
    "/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master",
)
LENS = [int(v) for v in os.environ.get("LENS", "128 512 1024 4096").split()]
N = int(os.environ.get("N", "8"))
OUTDIR = os.path.join(ROOT, "dev-docs/report/tuning/datasets")
os.makedirs(OUTDIR, exist_ok=True)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)

convs = json.load(open(SG))
texts = []
for c in convs:
    for turn in c.get("conversations", []):
        v = turn.get("value", "")
        if isinstance(v, str) and len(v) > 200:
            texts.append(v)
print(f"[nat-dataset] sharegpt 段落数 {len(texts)}", flush=True)

# 所有段落连成一条长流,再按 token 长度切 → 每条长度可控且文本自然。
stream = "\n\n".join(texts)
ids = tok(stream, add_special_tokens=False)["input_ids"]
print(f"[nat-dataset] 总 token {len(ids)}", flush=True)

start = 0
for L in LENS:
    rows = []
    for i in range(N):
        chunk = ids[start : start + L]
        start += L + 37  # 留一点间隔,避免各条完全重合
        if len(chunk) < L:
            start = 0
            chunk = ids[start : start + L]
            start += L + 37
        rows.append({"prompt": tok.decode(chunk)})
    path = os.path.join(OUTDIR, f"nat{L}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 复核:回读 tokenizer 确认长度
    back = tok(rows[0]["prompt"], add_special_tokens=False)["input_ids"]
    print(f"[nat-dataset] {path}  n={N} 目标={L} 实测首条={len(back)}", flush=True)
