#!/usr/bin/env python
"""批量 decode 吞吐基准(在模型能跑通后用于测 >100 tok/s)。

用法:MODEL=...  CONCURRENCY=N  python scripts/bench_llm.py
给 N 条短 prompt、每条生成 O 个 token,测 decode 阶段 tok/s(近似=批处理吞吐)。
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_PRECOMPILED", "1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get("TEST_MODEL", "")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "32"))
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", "32"))
MAXLEN = int(os.environ.get("GLM_MAXLEN", "1024"))
SKIP_TOK = os.environ.get("SKIP_TOK", "1") == "1"
EXPERT_DEV = os.environ.get("VLLM_EXPERTS_LOAD_DEVICE", "cpu")

from vllm import LLM, SamplingParams  # noqa: E402


def main() -> int:
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=MAXLEN,
        max_num_seqs=CONCURRENCY,
        enforce_eager=True,
        load_format=os.environ.get("GLM_LOAD_FORMAT", "auto"),
        trust_remote_code=False,
        skip_tokenizer_init=SKIP_TOK,
        kernel_config={"enable_jit_warmup": False},
    )
    print(f"[bench] constructed {time.time()-t0:.1f}s, experts_device={EXPERT_DEV}, "
          f"concurrency={CONCURRENCY}, out_tokens={OUT_TOKENS}", flush=True)

    if SKIP_TOK:
        prompts = [{"prompt_token_ids": [100, 200, 300, 400]} for _ in range(CONCURRENCY)]
    else:
        prompts = ["The capital of France is " for _ in range(CONCURRENCY)]

    sp = SamplingParams(max_tokens=OUT_TOKENS, temperature=0.0, ignore_eos=True)
    tt = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - tt
    total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[bench] {CONCURRENCY} reqs, {total_tok} tok in {dt:.2f}s -> "
          f"{total_tok/max(dt,1e-9):.2f} tok/s decode", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
