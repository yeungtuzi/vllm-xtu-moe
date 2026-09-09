#!/usr/bin/env python
"""Bounded binding validation: simulate mainline DS-V4 expert-loading (mega
mapping + CpuMegaExpertsParams.weight_loader) against the real checkpoint layer 3,
on CPU. Confirms raw fp4/raw-e8m0 binds into our CPU params matching xiaotu shape."""
import glob
import json
import os
import sys

import numpy as np
import torch  # noqa: F401
from safetensors import safe_open

REPO = "/home/user/lvllm/vllm-xiaotu-moe"
sys.path.insert(0, REPO)
import vllm_xiaotu_moe.hybrid_model as hm
from vllm_xiaotu_moe.hybrid_model import CpuMegaExpertsParams

MODEL = ("/home/user/.cache/modelscope/models/deepseek-ai--"
         "DeepSeek-V4-Flash-0731/snapshots/master")
H, I, LAYER, E, K = 4096, 2048, 3, 256, 6

# replicate mainline mega mapping exactly
from vllm.models.deepseek_v4.nvidia.model import make_deepseek_v4_expert_params_mapping
mapping = make_deepseek_v4_expert_params_mapping(E)

params = CpuMegaExpertsParams(num_experts=E, hidden_size=H,
                              intermediate_size=I, device="cpu")
print("params:", {n: tuple(tuple(p.shape) for p in [v])[0] for n, v in
                  [("w13", params.w13_weight), ("w2", params.w2_weight),
                   ("s13", params.w13_weight_scale), ("s2", params.w2_weight_scale)]})

# find shard
shard = None
for p in sorted(glob.glob(os.path.join(MODEL, "model-*.safetensors"))):
    with open(p, "rb") as f:
        hl = int.from_bytes(f.read(8), "little")
        json.loads(f.read(hl))
    with safe_open(p, "pt") as f:
        if f"layers.{LAYER}.ffn.experts.0.w1.weight" in f.keys():
            shard = p
            break

pd = {f"layers.{LAYER}.ffn.experts.{n}": p for n, p in params.named_parameters()}
bound, failed = 0, []
with safe_open(shard, "pt") as f:
    for k in sorted(f.keys(), key=lambda s: (s.split("experts.")[1].split(".")[0] if "experts." in s else "zz")):
        if f"layers.{LAYER}.ffn.experts." not in k:
            continue
        loaded = f.get_tensor(k)
        # apply hf->vllm mapper (fp4):  experts.\d+.w[123].scale -> .weight_scale
        import re
        k = re.sub(r"(\.experts\.\d+\.w\d)\.scale$", r"\1.weight_scale", k)
        if "weight_scale" in k and loaded.dtype == torch.float8_e8m0fnu:
            loaded = loaded.view(torch.uint8)
        hit = False
        for (param_name, weight_name, expert_id, shard_id) in mapping:
            if weight_name not in k:
                continue
            name_mapped = k.replace(weight_name, param_name)
            if name_mapped not in pd:
                continue
            param = pd[name_mapped]
            ok = param.weight_loader(param, loaded, name_mapped,
                                     shard_id=shard_id, expert_id=expert_id,
                                     return_success=True)
            if ok:
                bound += 1
                hit = True
                break
        if not hit:
            failed.append(k.replace(f"layers.{LAYER}.ffn.", ""))

print(f"bound {bound} expert tensors; failed {len(failed)}")
for f_ in failed[:10]:
    print("  FAILED:", f_)

# spot-verify a bound value equals raw checkpoint (w13 expert 0 w1 rows)
print("\nverify w13_weight[0, 0:I] == checkpoint experts.0.w1.weight:")
with safe_open(shard, "pt") as f:
    ck_w1 = f.get_tensor(f"layers.{LAYER}.ffn.experts.0.w1.weight").cpu()
got = params.w13_weight.data[0, 0:I]
same = bool(torch.equal(got, ck_w1.view(torch.uint8)))
print("  w13 w1 rows identical (byte view):", same, "| shape", tuple(got.shape))
print("  w13 nonzero:", int((params.w13_weight.data != 0).sum()),
      "/", params.w13_weight.data.numel())
print("  s13 nonzero:", int((params.w13_weight_scale.data != 0).sum()))
print("  s13[0,0:I]:", params.w13_weight_scale.data[0, 0:I].flatten()[:8].tolist())
print("RESULT:", "PASS" if same and bound >= E * 6 else "FAIL")
