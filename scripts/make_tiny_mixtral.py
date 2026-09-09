#!/usr/bin/env python
"""生成一个 vLLM 主线命名兼容的微型 Mixtral(bfloat16)检查点,用于通用后端
CPU/GPU 端到端等价性测试。

背景:`hf-internal-testing/tiny-random-MixtralForCausalLM` 用的是新版 transformers
的融合命名(`mlp.experts.gate_up_proj`),与主线 vLLM 的 `block_sparse_moe.experts.*`
不兼容(加载报 "no module named layers.0.mlp")。这里按主线期望的命名重新生成一份,
随机权重,尺寸极小(约 10 MB)。

用法: python scripts/make_tiny_mixtral.py [输出目录]
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

# Source of the tokenizer/config template (a public tiny Mixtral checkpoint).
SRC = os.environ.get("TINY_SRC", "")
if not SRC:
    raise SystemExit(
        "set TINY_SRC to a local snapshot of "
        "hf-internal-testing/tiny-random-MixtralForCausalLM (tokenizer/config only)"
    )
OUT = sys.argv[1] if len(sys.argv) > 1 else "./tiny-mixtral-vllm"

H, I, V = 64, 128, 32000
E, TOPK, L = 4, 2, 2
NKV, NH = 2, 4
HD = H // NH


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    g = torch.Generator().manual_seed(1234)
    std0 = float(os.environ.get("TINY_STD", "0.02"))

    def rnd(*shape, std=None):
        std = std0 if std is None else std
        return (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)

    t = {
        "model.embed_tokens.weight": rnd(V, H),
        "model.norm.weight": torch.ones(H, dtype=torch.bfloat16),
        "lm_head.weight": rnd(V, H),
    }
    for i in range(L):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        t[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        t[p + "self_attn.q_proj.weight"] = rnd(NH * HD, H)
        t[p + "self_attn.k_proj.weight"] = rnd(NKV * HD, H)
        t[p + "self_attn.v_proj.weight"] = rnd(NKV * HD, H)
        t[p + "self_attn.o_proj.weight"] = rnd(H, NH * HD)
        t[p + "block_sparse_moe.gate.weight"] = rnd(E, H)
        for e in range(E):
            t[p + f"block_sparse_moe.experts.{e}.w1.weight"] = rnd(I, H)
            t[p + f"block_sparse_moe.experts.{e}.w2.weight"] = rnd(H, I)
            t[p + f"block_sparse_moe.experts.{e}.w3.weight"] = rnd(I, H)

    save_file(t, os.path.join(OUT, "model.safetensors"))

    cfg = json.load(open(os.path.join(SRC, "config.json")))
    cfg.update(
        {
            "architectures": ["MixtralForCausalLM"],
            "torch_dtype": "bfloat16",
            "hidden_size": H,
            "intermediate_size": I,
            "num_hidden_layers": L,
            "num_local_experts": E,
            "num_experts_per_tok": TOPK,
            "num_attention_heads": NH,
            "num_key_value_heads": NKV,
            "vocab_size": V,
            "max_position_embeddings": 512,
            "sliding_window": None,
            "output_router_logits": False,
            "tie_word_embeddings": False,
        }
    )
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
              "special_tokens_map.json", "generation_config.json"):
        src = os.path.join(SRC, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, f))
    print(f"[tiny] wrote {OUT}: {len(t)} tensors, "
          f"{sum(x.numel() for x in t.values()) / 1e6:.2f} M params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
