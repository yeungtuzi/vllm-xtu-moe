#!/usr/bin/env python
"""GLM-5.3-Flash + GPU/CPU Mixed Mode E2E test.

Env:
  GLM_LOAD_FORMAT = dummy | auto   (default auto)
  GLM_MAXLEN      = max_model_len  (default 4096)
  VLLM_EXPERTS_LOAD_DEVICE = cpu | gpu
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_PRECOMPILED", "1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = "/home/user/.cache/modelscope/models/zai-org/GLM-5.3-Flash"
LOAD_FORMAT = os.environ.get("GLM_LOAD_FORMAT", "auto")
MAXLEN = int(os.environ.get("GLM_MAXLEN", "4096"))
EXPERT_DEV = os.environ.get("VLLM_EXPERTS_LOAD_DEVICE", "gpu")

from vllm import LLM, SamplingParams  # noqa: E402


def main() -> int:
    t0 = time.time()
    print(f"[glm] construct: load_format={LOAD_FORMAT} experts_device={EXPERT_DEV} "
          f"max_len={MAXLEN}", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=MAXLEN,
        max_num_seqs=8,
        enforce_eager=True,
        load_format=LOAD_FORMAT,
        trust_remote_code=False,
        skip_tokenizer_init=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    print(f"[glm] CONSTRUCTED in {time.time()-t0:.1f}s", flush=True)

    sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    tt = time.time()
    # skip_tokenizer_init: pass token ids directly (no tokenizer files present).
    out = llm.generate([{"prompt_token_ids": [100, 200, 300, 400]}], sp)
    dt = time.time() - tt
    ntok = len(out[0].outputs[0].token_ids)
    print(f"[glm] GENERATED {ntok} tok in {dt:.2f}s -> {ntok/max(dt,1e-9):.2f} tok/s",
          flush=True)
    print(f"[glm] token_ids: {out[0].outputs[0].token_ids[:16]}", flush=True)
    print("[glm] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
