#!/usr/bin/env python
"""fp8 MoE 模型的通用 CPU experts 后端端到端冒烟(首个非 DS-V4 真实模型)。

默认模型: `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`
  - 128 专家 / top-8 / 48 层 / hidden 2048 / moe_inter 768
  - 专家权重 fp8 e4m3 block 128x128,scale_inv 是 **bf16**(GLM-5.3 是 fp32)
  - 在 A100 上可以跑(没有 GLM-5.3 那种 MLA 维度问题)

测:能否加载 → oracle 是否选中 xiaotu CPU FP8 后端 → 输出是否连贯 →
短 prompt 时延 + 2000 token 长 prefill 的 TTFT 与 decode 速率。

用法:
  CUDA_VISIBLE_DEVICES=2 VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_MOE_SINGLECOPY=1 \
    python scripts/fp8_moe_smoke.py
"""
import os
import sys
import time

os.environ.setdefault("VLLM_EXPERTS_LOAD_DEVICE", "cpu")
os.environ.setdefault("XIAOTU_MOE_SINGLECOPY", "1")
os.environ.setdefault("XIAOTU_GPU_PREFILL_MIN_TOKENS", "0")
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "7200")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get(
    "SMOKE_MODEL",
    "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8"
    "/snapshots/5a5a776300a41aaa681dd7ff0106608ef2bc90db",
)
MAXLEN = int(os.environ.get("SMOKE_MAXLEN", "4096"))
LONG_TOKENS = int(os.environ.get("SMOKE_LONG_TOKENS", "2000"))


def main() -> int:
    from vllm import LLM, SamplingParams

    print(
        f"[fp8smoke] model={MODEL}\n"
        f"[fp8smoke] experts_device={os.environ['VLLM_EXPERTS_LOAD_DEVICE']} "
        f"singlecopy={os.environ.get('XIAOTU_MOE_SINGLECOPY')} maxlen={MAXLEN}",
        flush=True,
    )
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=MAXLEN,
        max_num_seqs=2,
        enforce_eager=True,
        trust_remote_code=False,
        dtype="bfloat16",
        enable_prefix_caching=False,  # 否则第二次同 prompt 会命中前缀缓存
        kernel_config={"enable_jit_warmup": False},
    )
    print(f"[fp8smoke] LOAD_OK in {time.time() - t0:.1f}s", flush=True)

    tok = llm.get_tokenizer()

    def chat_ids(q: str):
        msgs = [{"role": "user", "content": q}]
        try:
            out = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=False, enable_thinking=False,
            )
        except TypeError:
            out = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_dict=False
            )
        if hasattr(out, "input_ids"):      # BatchEncoding
            out = out["input_ids"]
        if isinstance(out, list) and out and hasattr(out[0], "ids"):
            out = out[0].ids              # list[Encoding]
        return list(out)

    # ---- 1) 连贯性 ----
    qs = [
        "中国的首都是哪座城市?只回答城市名。",
        "1+1 等于几?只回答数字。",
        "用一句话解释什么是混合专家(MoE)模型。",
    ]
    sp = SamplingParams(max_tokens=32, temperature=0.0, ignore_eos=False)
    t = time.time()
    outs = llm.generate(
        [{"prompt_token_ids": chat_ids(q)} for q in qs], sp
    )
    dt = time.time() - t
    print(f"[fp8smoke] 3 short prompts in {dt:.2f}s", flush=True)
    for q, o in zip(qs, outs):
        txt = o.outputs[0].text.strip().replace("\n", " ")
        print(f"  Q: {q}\n  A: {txt[:200]}\n", flush=True)

    # ---- 2) 长 prefill(2K)TTFT + decode ----
    seed = "在一座遥远的城市里,有一位工程师每天研究如何让大模型在普通的服务器上跑得更快。"
    filler = seed * (LONG_TOKENS // max(1, len(tok.encode(seed))) + 1)
    long_ids = tok.encode(filler)[:LONG_TOKENS]
    long_ids = long_ids + chat_ids("\n请用一句话总结上面这段文字。")
    print(f"[fp8smoke] long prompt = {len(long_ids)} tokens", flush=True)

    t = time.time()
    llm.generate(
        [{"prompt_token_ids": long_ids}],
        SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
    )
    ttft = time.time() - t

    n_dec = 32
    t = time.time()
    out = llm.generate(
        [{"prompt_token_ids": long_ids}],
        SamplingParams(max_tokens=n_dec, temperature=0.0, ignore_eos=True),
    )
    total = time.time() - t
    dec_rate = (n_dec - 1) / max(total - ttft, 1e-9)
    print(
        f"[fp8smoke] long prefill TTFT={ttft:.2f}s "
        f"({len(long_ids) / ttft:.0f} tok/s prefill), "
        f"decode≈{dec_rate:.1f} tok/s "
        f"(total {total:.2f}s for {n_dec} tokens)",
        flush=True,
    )
    print(f"  A: {out[0].outputs[0].text.strip()[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
