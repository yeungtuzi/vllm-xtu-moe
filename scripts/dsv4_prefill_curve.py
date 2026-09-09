#!/usr/bin/env python
"""Comprehensive prefill/latency/concurrency measurement for the report.

One engine construction, then a sequence of measurements; results are printed as
JSON lines (one object per measurement) so they can be turned into charts.

Env:
  TEST_MODEL   model path (required)
  TP           tensor-parallel size (default 1)
  GLM_MAXLEN   max_model_len (default 16384)
  MAX_NBT      max_num_batched_tokens (default = GLM_MAXLEN)
  GPU_UTIL     gpu_memory_utilization (default 0.85)
  MINTOK       GPU-prefill threshold for the "gpu" mode (default 2048)
  LENS         comma list of prompt token targets (default 1024,2048,4096,8192,16384)
  MODES        cpu,gpu  (default both; cpu skipped for TP>1 unless CPU=1)
  CONC         comma list of concurrency levels (default 1,2,4,8)
  CONC_LEN     prompt length used for the concurrency sweep (default 2048)
  DECODE_N     generated tokens for the decode-throughput test (default 64)
  OUT          jsonl output path (default prefill_curve.jsonl)
"""
import json
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
MINTOK = int(os.environ.get("MINTOK", "2048"))
LENS = [int(v) for v in os.environ.get(
    "LENS", "2048,4096,8192,16384").split(",")]
MODES = [m for m in os.environ.get("MODES", "cpu,gpu").split(",") if m]
if TP > 1 and os.environ.get("CPU") != "1":
    MODES = [m for m in MODES if m != "cpu"]
CONC = [int(v) for v in os.environ.get("CONC", "1,2,4,8").split(",")]
CONC_LEN = int(os.environ.get("CONC_LEN", "2048"))
DECODE_N = int(os.environ.get("DECODE_N", "64"))
# The CPU engine's per-expert scratch grows with the batch and can exhaust host
# RAM at very long prompts; skip the CPU measurement above this length.
SKIP_CPU_ABOVE = int(os.environ.get("SKIP_CPU_ABOVE", "0") or 0)
OUT = os.environ.get("OUT", "prefill_curve.jsonl")
# Runtime threshold switch: the model runs in worker processes (TP>=1) that do
# not see this process's os.environ, so we write the threshold to a file that
# gpu_prefill.gpu_prefill_min_tokens() re-reads on every layer.
THR_FILE = os.environ.get(
    "XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(OUT)) or ".", "thr.txt"))
os.environ["XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE"] = THR_FILE


def set_threshold(v: int) -> None:
    with open(THR_FILE, "w") as f:
        f.write(str(int(v)))

from vllm import LLM, SamplingParams  # noqa: E402

# ~4 chars/token English filler; words chosen so the tokenizer is stable.
FILLER = ("Marie Curie was a physicist and chemist who did pioneering research "
          "on radioactivity and discovered polonium and radium. ")


def make_prompt(n_tokens: int) -> str:
    # FILLER is ~127 chars / ~23 tokens for this text (~5.6 chars/token).
    units = max(1, int(n_tokens / 22) + 2)
    return (FILLER * units)[: int(n_tokens * 5.9)]


def emit(**kw):
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = json.dumps(kw, ensure_ascii=False)
    print("JSON " + line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


def main() -> int:
    print(f"[curve] model={MODEL} tp={TP} maxlen={MAXLEN} mnbt={MAX_NBT} "
          f"modes={MODES} lens={LENS} conc={CONC}", flush=True)
    t_load = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAXLEN,
        max_num_seqs=max(16, max(CONC)),
        max_num_batched_tokens=MAX_NBT,
        enforce_eager=True,
        load_format="auto",
        trust_remote_code=False,
        kernel_config={"enable_jit_warmup": False},
        enable_prefix_caching=False,
    )
    emit(kind="load", tp=TP, maxlen=MAXLEN, mnbt=MAX_NBT,
         load_s=round(time.time() - t_load, 1))

    tok = llm.get_tokenizer()

    def real_len(s):
        return len(tok.encode(s))

    # Warm-up: the first GPU-prefill call builds the pinned K-major host cache
    # (~147 GB / TP) and JIT-compiles the Triton kernels. Absorb it here so the
    # measured points are steady-state.
    if os.environ.get("WARMUP", "1") == "1" and "gpu" in MODES:
        set_threshold(MINTOK)
        _p = make_prompt(min(2048, MAXLEN))
        t0 = time.time()
        llm.generate([_p], SamplingParams(max_tokens=1, temperature=0.0))
        print(f"[curve] warmup (pinned cache + JIT) = {time.time()-t0:.1f}s",
              flush=True)
        emit(kind="warmup", tp=TP, wall_s=round(time.time() - t0, 1))

    # ---------------- TTFT vs prompt length ----------------
    for L in LENS:
        prompt = make_prompt(L)
        n = real_len(prompt)
        for mode in MODES:
            if mode == "cpu" and SKIP_CPU_ABOVE and L > SKIP_CPU_ABOVE:
                continue
            set_threshold(0 if mode == "cpu" else MINTOK)
            sp = SamplingParams(max_tokens=1, temperature=0.0)
            t0 = time.time()
            out = llm.generate([prompt], sp)[0]
            dt = time.time() - t0
            emit(kind="ttft", tp=TP, mode=mode, target_len=L, tokens=n,
                 ttft_s=round(dt, 3), tok_per_s=round(n / dt, 1),
                 text=out.outputs[0].text[:24])
            print(f"[curve] L={L}({n} tok) {mode}: {dt:.2f}s "
                  f"({n/dt:.0f} tok/s)", flush=True)

    # ---------------- concurrency / aggregate prefill throughput ----------
    set_threshold(MINTOK)
    prompt = make_prompt(CONC_LEN)
    n = real_len(prompt)
    for c in CONC:
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        t0 = time.time()
        outs = llm.generate([prompt] * c, sp)
        dt = time.time() - t0
        emit(kind="concurrency", tp=TP, conc=c, tokens=n, total_tokens=c * n,
             wall_s=round(dt, 3), tok_per_s=round(c * n / dt, 1),
             per_req_ttft_s=round(dt, 3),
             n_out=len(outs))
        print(f"[curve] conc={c} x {n} tok: {dt:.2f}s "
              f"({c*n/dt:.0f} tok/s aggregate)", flush=True)

    # ---------------- decode throughput (short prompt, N tokens) ---------
    set_threshold(MINTOK)
    short = make_prompt(256)
    sp = SamplingParams(max_tokens=DECODE_N, temperature=0.0, ignore_eos=True)
    t0 = time.time()
    out = llm.generate([short], sp)[0]
    dt = time.time() - t0
    nout = len(out.outputs[0].token_ids)
    emit(kind="decode", tp=TP, out_tokens=nout, wall_s=round(dt, 3),
         decode_tok_per_s=round(nout / dt, 2))
    print(f"[curve] decode {nout} tok in {dt:.2f}s ({nout/dt:.1f} tok/s)",
          flush=True)
    print("[curve] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
