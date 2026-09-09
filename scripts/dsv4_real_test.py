#!/usr/bin/env python
"""DS-V4 real weights on A100 + mixed mode: correctness sample THEN high-concurrency throughput.

One model construct; first a single-request correctness probe (print token_ids+text),
then a CONCURRENCY-way decode to measure batched tok/s. Env: CONCURRENCY (default 32),
OUT_TOKENS (default 64), PROMPT.
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
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", "64"))
MAXLEN = int(os.environ.get("GLM_MAXLEN", "1024"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")

from vllm import LLM, SamplingParams  # noqa: E402


def main() -> int:
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=MAXLEN,
        max_num_batched_tokens=int(os.environ.get("MAX_NBT", "4096")),
        max_num_seqs=max(CONCURRENCY * 2, 64),
        enforce_eager=True,
        load_format="auto",
        trust_remote_code=False,
        kernel_config={"enable_jit_warmup": False},
    )
    print(f"[dsv4] constructed {time.time()-t0:.1f}s", flush=True)

    # --- correctness probe: one request ---
    sp = SamplingParams(max_tokens=OUT_TOKENS, temperature=0.0)
    tt = time.time()
    one = llm.generate([PROMPT], sp)[0]
    pid = one.outputs[0].token_ids
    print(f"[dsv4] probe token_ids[:40]: {pid[:40]}", flush=True)
    print(f"[dsv4] probe text: {one.outputs[0].text[:120]!r}", flush=True)
    print(f"[dsv4] probe {len(pid)} tok in {time.time()-tt:.2f}s -> "
          f"{len(pid)/(time.time()-tt):.2f} tok/s (single)", flush=True)

    # --- high-concurrency throughput ---
    prompts = [PROMPT for _ in range(CONCURRENCY)]
    sp2 = SamplingParams(max_tokens=OUT_TOKENS, temperature=0.0, ignore_eos=True)
    tt = time.time()
    outs = llm.generate(prompts, sp2)
    dt = time.time() - tt
    total = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[dsv4] {CONCURRENCY} reqs {total} tok in {dt:.2f}s -> "
          f"{total/max(dt,1e-9):.2f} tok/s decode (concurrency={CONCURRENCY})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
