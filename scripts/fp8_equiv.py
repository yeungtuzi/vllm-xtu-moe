#!/usr/bin/env python
"""真实 fp8 模型的 CPU 专家 vs GPU 专家数值对照(同一 prompt 的前向 logprob)。

用 `prompt_logprobs`:同一条 prompt 的前向,两条路径的输入完全相同,所以 logprob
差异直接反映专家计算的数值一致性(与采样噪声无关)。

  MOE_MODE=gpu python scripts/fp8_equiv.py /tmp/fp8_gpu.json
  MOE_MODE=cpu python scripts/fp8_equiv.py /tmp/fp8_cpu.json
  python scripts/fp8_equiv.py --compare /tmp/fp8_gpu.json /tmp/fp8_cpu.json
"""
from __future__ import annotations

import json
import os
import sys
import time

MODE = os.environ.get("MOE_MODE", "gpu")
if MODE == "cpu":
    os.environ["VLLM_EXPERTS_LOAD_DEVICE"] = "cpu"
    os.environ.setdefault("XIAOTU_MOE_SINGLECOPY", "1")
else:
    os.environ["VLLM_EXPERTS_LOAD_DEVICE"] = "gpu"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get(
    "SMOKE_MODEL",
    "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8"
    "/snapshots/5a5a776300a41aaa681dd7ff0106608ef2bc90db",
)
QUESTIONS = [
    "中国的首都是哪座城市?只回答城市名。",
    "1+1 等于几?只回答数字。",
]


def main() -> int:
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        return compare(sys.argv[2], sys.argv[3])
    out_path = sys.argv[1] if len(sys.argv) > 1 else f"/tmp/fp8_equiv_{MODE}.json"

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_num_seqs=2,
        enforce_eager=True,
        trust_remote_code=False,
        dtype="bfloat16",
        enable_prefix_caching=False,
        kernel_config={"enable_jit_warmup": False},
    )
    print(f"[fp8equiv] {MODE}: loaded in {time.time() - t0:.1f}s", flush=True)

    tok = llm.get_tokenizer()

    def ids_of(q):
        msgs = [{"role": "user", "content": q}]
        out = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=False,
            enable_thinking=False,
        )
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        return list(out)

    results = []
    for q in QUESTIONS:
        ids = ids_of(q)
        t = time.time()
        o = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True,
                           logprobs=5, prompt_logprobs=5),
        )[0]
        dt = time.time() - t
        plp = []
        for entry in (o.prompt_logprobs or []):
            if entry is None:
                plp.append(None)
                continue
            plp.append({str(k): round(float(v.logprob), 6) for k, v in entry.items()})
        results.append(
            {
                "question": q,
                "prompt_ids": ids,
                "prompt_text": tok.decode(ids),
                "prompt_logprobs": plp,
                "gen_ids": list(o.outputs[0].token_ids),
                "gen_text": o.outputs[0].text,
            }
        )
        print(f"[fp8equiv] {MODE}: {dt:.1f}s  {q} -> {o.outputs[0].text!r}", flush=True)

    json.dump({"mode": MODE, "results": results}, open(out_path, "w"), indent=1)
    print(f"[fp8equiv] wrote {out_path}", flush=True)
    return 0


def compare(a_path: str, b_path: str) -> int:
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    print(f"[fp8equiv] {a['mode']} vs {b['mode']}")
    worst_all = 0.0
    for ra, rb in zip(a["results"], b["results"]):
        print(f"  Q: {ra['question']}")
        print(f"    gpu gen: {ra['gen_text']!r}")
        print(f"    cpu gen: {rb['gen_text']!r}")
        n = 0
        worst = 0.0
        for pa, pb in zip(ra["prompt_logprobs"], rb["prompt_logprobs"]):
            if pa is None or pb is None:
                continue
            for k in set(pa) & set(pb):
                d = abs(pa[k] - pb[k])
                worst = max(worst, d)
                n += 1
        worst_all = max(worst_all, worst)
        print(f"    prompt-logprob overlap n={n} max|d|={worst:.4f}")
    print(f"  overall max|d| = {worst_all:.4f}")
    return 0 if worst_all < 0.1 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
