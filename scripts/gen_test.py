#!/usr/bin/env python
"""Full E2E construct + load + engines + generation on GPU2 (hybrid OOT plugin)."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_PRECOMPILED", "1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")

from vllm import LLM, SamplingParams

def main():
    t0 = time.time()
    print("[test] constructing LLM (hybrid OOT)...", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        max_num_seqs=64,
        enforce_eager=True,
        trust_remote_code=False,
    )
    print(f"[test] model ready in {time.time()-t0:.1f}s", flush=True)

    prompts = ["Hello, can you introduce yourself in one sentence?"]
    sp = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=True)
    tt = time.time()
    out = llm.generate(prompts, sp)
    dt = time.time() - tt
    print(f"[test] single prompt {len(out[0].outputs[0].token_ids)} tok "
          f"in {dt:.2f}s -> {len(out[0].outputs[0].token_ids)/dt:.1f} tok/s", flush=True)
    print("[test] text:", repr(out[0].outputs[0].text[:200]), flush=True)
    print("[test] DONE", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
