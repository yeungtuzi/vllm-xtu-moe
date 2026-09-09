#!/usr/bin/env python
"""Long-prefill GPU-vs-CPU comparison on real DS-V4-Flash weights.

One engine construction, then generate from two different long prompts: first
with VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=0 (CPU prefill), then with a high
threshold (GPU prefill). Reports TTFT for each.

Env:
  GLM_MAXLEN default 4096, MAX_NBT default 4096, GPU_UTIL,
  LPROMPT_LEN default 60 units (~600 tokens). The GPU path only makes sense for
  qlen >= the threshold; to keep the run quick the env is applied directly here.
"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
MODEL = os.environ.get("TEST_MODEL", "")
MAXLEN = int(os.environ.get("GLM_MAXLEN", "4096"))
MAX_NBT = int(os.environ.get("MAX_NBT", "4096"))
UNITS = int(os.environ.get("LPROMPT_LEN", "300"))
GPU_MIN = os.environ.get("VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS", 0)
from vllm import LLM, SamplingParams

UNITA = ("Marie Curie was a physicist and chemist who did pioneering research on "
         "radioactivity and discovered polonium and radium. ")
UNITB = ("The Amazon rainforest produces a large share of the world's oxygen. "
         "It is home to an enormous diversity of species. ")

def main():
    print(f"[longprefill] GPU threshold env = {GPU_MIN} (0 = all-CPU path this run)"
          if str(GPU_MIN) == "0" else
          f"[longprefill] GPU threshold env = {GPU_MIN}")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=int(os.environ.get("TP", "1")),
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
        max_model_len=MAXLEN,
        max_num_seqs=16,
        max_num_batched_tokens=MAX_NBT,
        enforce_eager=True,
        load_format="auto",
        trust_remote_code=False,
        kernel_config={"enable_jit_warmup": False},
        enable_prefix_caching=False,
    )
    t0 = time.time()
    a = llm.generate([UNITA * UNITS], SamplingParams(max_tokens=1, temperature=0.0))[0]
    t_cpu = time.time() - t0
    os.environ["VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS"] = str(GPU_MIN)
    t1 = time.time()
    b = llm.generate([UNITB * UNITS], SamplingParams(max_tokens=1, temperature=0.0))[0]
    t_gpu = time.time() - t1
    print(f"[longprefill] CPU  TTFT={t_cpu:.2f}s text={a.outputs[0].text[:40]!r}", flush=True)
    print(f"[longprefill] GPU  TTFT={t_gpu:.2f}s text={b.outputs[0].text[:40]!r}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
