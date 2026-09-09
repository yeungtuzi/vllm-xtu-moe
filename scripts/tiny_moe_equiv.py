#!/usr/bin/env python
"""通用路径的**端到端数值等价性**测试(小模型,CPU 专家 vs GPU 专家)。

目的:验证 `mixed_experts.py` 的通用后端在**真实 vLLM 推理流程**里给出的结果,
与主线 GPU 专家内核一致(不是"能跑就行")。用 `hf-internal-testing/tiny-random-
MixtralForCausalLM`(bf16,4 专家 top-2,2 层)跑两次:

    MOE_MODE=gpu python scripts/tiny_moe_equiv.py out_gpu.json
    MOE_MODE=cpu python scripts/tiny_moe_equiv.py out_cpu.json
    python scripts/tiny_moe_equiv.py --compare out_gpu.json out_cpu.json

对比:每个生成 token 的 id + 该 token 的 logprob。

用法(完整):
    TINY_MODEL=/path/to/tiny-random-MixtralForCausalLM
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
    "TINY_MODEL",
    "/home/user/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-MixtralForCausalLM"
    "/snapshots/ccb12fe2fc142cb752085506c3db22572290e90c",
)
PROMPTS = [
    [100, 200, 300, 400, 500],
    [42] * 16,
    [7, 13, 29, 31, 37, 41, 43, 47],
]
MAX_TOKENS = 16


def run(out_path: str) -> int:
    from vllm import LLM, SamplingParams

    if os.environ.get("SIMULATE_STOCK_VLLM") == "1":
        # 模拟"原生 vLLM":先让插件应用 shim(此时 env=cpu),再把 envs 里的
        # VLLM_EXPERTS_LOAD_DEVICE 改成 gpu —— 这样**源码里的**混合模式分支全部
        # 失效(等价于未打补丁的主线),只剩插件的 monkey-patch 在工作。
        import vllm_xiaotu_moe  # noqa: F401
        import vllm.envs as _envs

        _envs.VLLM_EXPERTS_LOAD_DEVICE = "gpu"
        print("[equiv] SIMULATE_STOCK_VLLM=1 (source-level mixed-mode branches off)", flush=True)

    t0 = time.time()
    moe_backend = os.environ.get("MOE_BACKEND", "auto")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        moe_backend=moe_backend,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
        trust_remote_code=False,
        kernel_config={"enable_jit_warmup": False},
    )
    print(
        f"[equiv] {MODE} (moe_backend={moe_backend}): loaded in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    sp = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        ignore_eos=True,
        logprobs=1,
        prompt_logprobs=2,   # 同一 prompt 的前向:两种后端应给同样的 logprob
    )
    outs = llm.generate(
        [{"prompt_token_ids": p} for p in PROMPTS], sp
    )
    results = []
    for p, o in zip(PROMPTS, outs):
        ids = list(o.outputs[0].token_ids)
        lps = []
        for step in range(len(ids)):
            lp = o.outputs[0].logprobs[step][ids[step]]
            lps.append(round(float(lp.logprob), 6))
        # prompt_logprobs[i] 是"给定前 i-1 个 token 时第 i 个 token 的 logprob";
        # 两条路径的输入完全相同,所以这是最干净的数值对照。
        plps = []
        margins = []
        for i, tid in enumerate(p):
            entry = o.prompt_logprobs[i] if o.prompt_logprobs else None
            if entry is None or tid not in entry:
                plps.append(None)
                margins.append(None)
                continue
            vals = sorted(
                (float(v.logprob) for v in entry.values()), reverse=True
            )
            plps.append(round(float(entry[tid].logprob), 6))
            margins.append(round(vals[0] - vals[1], 6) if len(vals) > 1 else None)
        results.append(
            {
                "prompt": p,
                "token_ids": ids,
                "logprobs": lps,
                "prompt_logprobs": plps,
                "prompt_margins": margins,
            }
        )
    json.dump(
        {"mode": MODE, "moe_backend": moe_backend, "results": results},
        open(out_path, "w"),
        indent=1,
    )
    print(f"[equiv] {MODE}: wrote {out_path}", flush=True)
    return 0


def compare(a_path: str, b_path: str) -> int:
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    total = match = 0
    worst = 0.0
    for ra, rb in zip(a["results"], b["results"]):
        for i, (ta, tb) in enumerate(zip(ra["token_ids"], rb["token_ids"])):
            total += 1
            match += int(ta == tb)
            if ta == tb:
                worst = max(worst, abs(ra["logprobs"][i] - rb["logprobs"][i]))

    # prompt logprobs:相同输入下的前向对照(与采样噪声无关)
    pn = 0
    pworst = 0.0
    pover = 0
    min_margin = None
    worst_ratio = 0.0
    deltas: list[float] = []
    for ra, rb in zip(a["results"], b["results"]):
        for la, lb, mg in zip(
            ra.get("prompt_logprobs", []),
            rb.get("prompt_logprobs", []),
            ra.get("prompt_margins", []),
        ):
            if la is None or lb is None:
                continue
            pn += 1
            d = abs(la - lb)
            pworst = max(pworst, d)
            if mg is not None:
                min_margin = mg if min_margin is None else min(min_margin, mg)
                if mg > 0:
                    worst_ratio = max(worst_ratio, d / mg)
            # 与"是否改变采样决策"直接相关的判据:delta 是否超过 top1-top2 间距
            if mg is not None and d > mg:
                pover += 1
            deltas.append(d)
    print(
        f"[equiv] {a['mode']}/{a.get('moe_backend')} vs "
        f"{b['mode']}/{b.get('moe_backend')}:\n"
        f"        prompt-logprob: n={pn} max|d|={pworst:.5f} "
        f"median|d|={sorted(deltas)[len(deltas) // 2]:.5f} "
        f"min top1-top2 margin={min_margin} "
        f"max(|d|/margin)={worst_ratio:.4f} "
        f"#(|d|>margin)={pover}   <- 相同输入,与采样噪声无关\n"
        f"        greedy tokens : {match}/{total} ({100 * match / max(total, 1):.1f}%), "
        f"max|dlogprob| on matching tokens={worst:.4f}"
    )
    # 判据:|Δlogprob| 必须远小于 top1-top2 间距(不改变采样决策)
    # 判据:①Δ 不得大于 top1-top2 间距(否则会改变采样决策);②greedy token 全同
    return 0 if (pn > 0 and worst_ratio < 1.0 and match == total) else 1


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        raise SystemExit(compare(sys.argv[2], sys.argv[3]))
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else f"/tmp/equiv_{MODE}.json"))
