#!/usr/bin/env python
"""GLM-5.3-Flash 端到端冒烟测试(通用 CPU experts 后端,无模型级 OOT 覆盖)。

目的:验证 `mixed_experts.py` 的通用路径对 fp8 模型成立:
  1. oracle 选中 xiaotu CPU FP8 后端(而非 GPU MARLIN/TRITON);
  2. 306 GB 权重能加载、KV 能分配;
  3. 路由(sigmoid + noaux_tc + e_score_correction_bias)正确 -> 输出连贯;
  4. 记录 prefill / decode 时延(诚实记录,不美化)。

用法:
  CUDA_VISIBLE_DEVICES=2 VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_MOE_SINGLECOPY=1 \
    python scripts/glm53_smoke.py
"""
import os
import sys
import time

os.environ.setdefault("VLLM_EXPERTS_LOAD_DEVICE", "cpu")
os.environ.setdefault("XIAOTU_MOE_SINGLECOPY", "1")
os.environ.setdefault("XIAOTU_GPU_PREFILL_MIN_TOKENS", "0")  # 先纯 CPU prefill
os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "7200")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get(
    "GLM_MODEL", "/home/user/.cache/modelscope/models/zai-org/GLM-5.3-Flash"
)
MAXLEN = int(os.environ.get("GLM_MAXLEN", "4096"))
LONG_TOKENS = int(os.environ.get("GLM_LONG_TOKENS", "2000"))


def main() -> int:
    from vllm import LLM, SamplingParams

    print(
        f"[glm53] model={MODEL}\n"
        f"[glm53] experts_device={os.environ['VLLM_EXPERTS_LOAD_DEVICE']} "
        f"singlecopy={os.environ.get('XIAOTU_MOE_SINGLECOPY')} "
        f"gpu_prefill_min={os.environ.get('XIAOTU_GPU_PREFILL_MIN_TOKENS')} "
        f"maxlen={MAXLEN}",
        flush=True,
    )

    t0 = time.time()
    hf_overrides = None
    raw = os.environ.get("GLM_HF_OVERRIDES")
    if raw:
        import json

        hf_overrides = json.loads(raw)
        print(f"[glm53] hf_overrides={hf_overrides}", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=MAXLEN,
        max_num_seqs=2,
        enforce_eager=True,
        trust_remote_code=False,
        dtype="bfloat16",
        hf_overrides=hf_overrides,
        kernel_config={"enable_jit_warmup": False},
    )
    load_s = time.time() - t0
    print(f"[glm53] LOAD_OK in {load_s:.1f}s", flush=True)

    tok = llm.get_tokenizer()
    sp_short = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=False)

    # ---- 1) 短 prompt(应走 CPU prefill) ----
    prompts = [
        "中国的首都是哪座城市?请用一句话回答。",
        "1+1等于几?只回答数字。",
        "用一句话解释什么是混合专家(MoE)模型。",
    ]
    t = time.time()
    outs = llm.generate(prompts, sp_short)
    dt = time.time() - t
    print(f"[glm53] short batch: {len(prompts)} prompts in {dt:.2f}s", flush=True)
    for p, o in zip(prompts, outs):
        txt = o.outputs[0].text.strip().replace("\n", " ")
        print(f"  Q: {p}\n  A: {txt[:300]}\n", flush=True)

    # ---- 2) 长 prompt(2K)测 prefill 时延 ----
    seed = "在一座遥远的城市里,有一位工程师每天研究如何让大模型在普通的服务器上跑得更快。"
    filler = (seed * (LONG_TOKENS // max(1, len(tok.encode(seed))) + 1))
    ids = tok.encode(filler)[:LONG_TOKENS]
    prompt_ids = ids + tok.encode("\n请用一句话总结上面这段文字。")
    print(f"[glm53] long prompt: {len(prompt_ids)} tokens", flush=True)
    sp_long = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    t = time.time()
    out = llm.generate([{"prompt_token_ids": prompt_ids}], sp_long)
    dt = time.time() - t
    n_out = len(out[0].outputs[0].token_ids)
    print(
        f"[glm53] long prefill+decode: {dt:.2f}s "
        f"(~{dt - n_out * 0.01:.2f}s prefill est, {n_out} out tokens)",
        flush=True,
    )
    print(f"  A: {out[0].outputs[0].text.strip()[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
