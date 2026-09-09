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

MODEL = os.environ.get("SMOKE_MODEL", "")
if not MODEL:
    raise SystemExit("set SMOKE_MODEL=<path or HF repo id of an fp8 MoE model>")
QUESTIONS = [
    "中国的首都是哪座城市?只回答城市名。",
    "1+1 等于几?只回答数字。",
]
# 可用 QUESTIONS_INDEX=0,1 只跑其中一条(用于定位"跨请求状态污染")
if os.environ.get("QUESTIONS_INDEX"):
    QUESTIONS = [QUESTIONS[int(i)] for i in os.environ["QUESTIONS_INDEX"].split(",")]


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
        max_num_seqs=int(os.environ.get("MAX_SEQS", "2")),
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
    """比较两条路径。真正有意义的指标:
      - top-1 token 是否一致(采样决策是否相同)
      - **实际 prompt token** 的 logprob 偏差(而不是 top-5 尾部 token 的并集)
    """
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    print(f"[fp8equiv] {a['mode']} vs {b['mode']}")
    ok = True
    for ra, rb in zip(a["results"], b["results"]):
        print(f"  Q: {ra['question']}")
        print(f"    {a['mode']} gen: {ra['gen_text']!r}")
        print(f"    {b['mode']} gen: {rb['gen_text']!r}")
        n = agree = 0
        deltas = []
        for i, (pa, pb) in enumerate(
            zip(ra["prompt_logprobs"], rb["prompt_logprobs"])
        ):
            if pa is None or pb is None:
                continue
            n += 1
            agree += int(max(pa, key=pa.get) == max(pb, key=pb.get))
            tok = str(ra["prompt_ids"][i])
            if tok in pa and tok in pb:
                deltas.append(abs(pa[tok] - pb[tok]))
        mean_d = sum(deltas) / len(deltas) if deltas else float("nan")
        max_d = max(deltas) if deltas else float("nan")
        print(f"    top-1 agreement {agree}/{n}; actual-token |dlogprob| "
              f"mean={mean_d:.4f} max={max_d:.4f}")
        # 判据:top-1 全一致,且实际 token 的 logprob 偏差在 ~0.3 以内
        # (GPU 侧对激活做动态 fp8 量化,CPU 侧用 bf16,存在系统性小差异)
        if n == 0 or agree != n or max_d > 0.3:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
