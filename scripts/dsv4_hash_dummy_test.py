#!/usr/bin/env python
"""Fast iteration: --load-format dummy for DS-V4 hybrid hash/attention debug."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_PRECOMPILED", "1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
MODEL = os.environ.get("TEST_MODEL", "")
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", "4"))
from vllm import LLM, SamplingParams
t0=__import__('time').time()
llm = LLM(model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.85,
          max_model_len=512, max_num_seqs=64, enforce_eager=True,
          load_format="dummy", trust_remote_code=False,
          kernel_config={"enable_jit_warmup": False})
print(f"[dummy] constructed {__import__('time').time()-t0:.1f}s", flush=True)
sp = SamplingParams(max_tokens=OUT_TOKENS, temperature=0.0)
one = llm.generate(["The capital of France is"], sp)[0]
print(f"[dummy] token_ids[:10]: {one.outputs[0].token_ids[:10]}", flush=True)
print(f"[dummy] text: {one.outputs[0].text[:80]!r}", flush=True)
