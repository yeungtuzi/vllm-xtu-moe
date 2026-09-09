#!/usr/bin/env python
"""DS-V4-Flash throughput sweep on one constructed engine (A100 + mixed-mode CPU MoE).

Constructs the model once, then decodes at several concurrency levels to map
tok/s vs batch size. Env: TEST_MODEL, GLM_MAXLEN (default 256), OUT_TOKENS (32),
SWEEP (comma list, default 1,32,64,128,256).
"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_PRECOMPILED", "1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get("TEST_MODEL", "")
MAXLEN = int(os.environ.get("GLM_MAXLEN", "256"))
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", "32"))
SWEEP = [int(x) for x in os.environ.get("SWEEP", "1,32,64,128,256").split(",")]
PROMPT = os.environ.get("PROMPT", "The capital of France is")
CUDAGRAPH = os.environ.get("CUDAGRAPH", "0") == "1"

from vllm import LLM, SamplingParams  # noqa: E402


def main() -> int:
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
        max_model_len=MAXLEN,
        max_num_batched_tokens=int(os.environ.get("MAX_NBT", "4096")),
        max_num_seqs=max(SWEEP) * 2,
        enforce_eager=not CUDAGRAPH,
        compilation_config=(
            {"cudagraph_mode": "FULL_DECODE_ONLY"} if CUDAGRAPH else None
        ),
        load_format="auto",
        trust_remote_code=False,
        kernel_config={"enable_jit_warmup": False},
    )
    print(
        f"[sweep] constructed {time.time()-t0:.1f}s maxlen={MAXLEN} "
        f"cudagraph={CUDAGRAPH}",
        flush=True,
    )

    sp1 = SamplingParams(max_tokens=OUT_TOKENS, temperature=0.0)
    first = llm.generate([PROMPT], sp1)[0]
    print(
        f"[sweep] B=1 sanity token_ids={first.outputs[0].token_ids[:16]} "
        f"text={first.outputs[0].text[:80]!r}",
        flush=True,
    )

    for b in SWEEP:
        sp = SamplingParams(max_tokens=OUT_TOKENS, temperature=0.0, ignore_eos=True)
        prompts = [PROMPT] * b
        tt = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - tt
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        print(
            f"[sweep] B={b:4d} {total:6d} tok in {dt:7.2f}s -> "
            f"{total/max(dt,1e-9):7.2f} tok/s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
