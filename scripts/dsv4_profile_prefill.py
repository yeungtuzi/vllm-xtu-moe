#!/usr/bin/env python
"""One long-prefill pass under nsys, to decompose MoE vs attention/indexer.

Loads the model exactly like dsv4_prefill_curve.py, warms the pinned cache, then
brackets a single long generate with cudaProfilerStart/Stop so
`nsys profile --capture-range=cudaProfilerApi` only records that pass.

Env: TEST_MODEL, TP, GLM_MAXLEN, MAX_NBT, GPU_UTIL, MINTOK, LEN (default 16000).
Output: report/prof_<LEN>.nsys-rep (see report/profile_prefill.sh).
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get("TEST_MODEL", "")
TP = int(os.environ.get("TP", "1"))
MAXLEN = int(os.environ.get("GLM_MAXLEN", "16384"))
MAX_NBT = int(os.environ.get("MAX_NBT", str(MAXLEN)))
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.85"))
MINTOK = int(os.environ.get("MINTOK", "512"))
LEN = int(os.environ.get("LEN", "16000"))

from vllm import LLM, SamplingParams  # noqa: E402

FILLER = ("Marie Curie was a physicist and chemist who did pioneering research "
          "on radioactivity and discovered polonium and radium. ")


def make_prompt(n_tokens):
    units = max(1, int(n_tokens / 22) + 2)
    return (FILLER * units)[: int(n_tokens * 5.9)]


def main():
    llm = LLM(
        model=MODEL, tensor_parallel_size=TP, gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAXLEN, max_num_seqs=16, max_num_batched_tokens=MAX_NBT,
        enforce_eager=True, load_format="auto", trust_remote_code=False,
        kernel_config={"enable_jit_warmup": False}, enable_prefix_caching=False,
    )
    tok = llm.get_tokenizer()
    prompt = make_prompt(LEN)
    n = len(tok.encode(prompt))
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    t0 = time.time()
    llm.generate([prompt], sp)
    print(f"[prof] warmup {time.time()-t0:.1f}s tokens={n}", flush=True)

    rt = __import__("torch").cuda.cudart()
    rt.cudaProfilerStart()
    t0 = time.time()
    llm.generate([prompt], sp)
    dt = time.time() - t0
    rt.cudaProfilerStop()
    print(f"[prof] measured {dt:.2f}s tokens={n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
