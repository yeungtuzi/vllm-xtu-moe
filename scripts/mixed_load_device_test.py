#!/usr/bin/env python
"""Verify GPU/CPU Mixed: VLLM_EXPERTS_LOAD_DEVICE=cpu puts routed-expert params
on CPU (and leaves the rest of the layer on the compute device), without OOM.

Lightweight: builds a tiny FusedMoEFactory via a bare VllmConfig (pattern from
tests/kernels/moe/test_zero_expert_moe.py). Run once with =gpu and once with =cpu.
"""
import os
import sys
import tempfile

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory


def main() -> int:
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        # Simulate a real GPU vLLM run: default allocation device is cuda. The
        # mixed-mode code inside RoutedExperts.__init__ must then override it
        # for the expert weights only (VLLM_EXPERTS_LOAD_DEVICE=cpu).
        with torch.device("cuda"):
            layer = FusedMoEFactory(
            num_experts=8,
            top_k=2,
            hidden_size=256,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="mixed_load_device_test",
            renormalize=True,
            routed_scaling_factor=1.0,
            scoring_func="softmax",
        )
    re_ = layer.routed_experts
    print(f"[test] routed_experts type = {type(re_).__name__}")
    params = dict(re_.named_parameters())
    print(f"[test] expert params: {sorted(params)}")
    for name in ("w13_weight", "w2_weight"):
        if name in params:
            print(f"[test]   {name}.device = {params[name].device}")
    # Non-expert parts of the layer should stay on the compute device.
    gate_dev = None
    for n, p in layer.named_parameters():
        if n.endswith("gate.weight") or ".gate." in n:
            gate_dev = p.device
            break
    print(f"[test] gate param device = {gate_dev}")
    # Under =cpu, expert params must be CPU; under =gpu they must be cuda.
    want = os.environ.get("VLLM_EXPERTS_LOAD_DEVICE", "gpu")
    exp_dev = params["w13_weight"].device.type if "w13_weight" in params else "?"
    ok = (exp_dev == "cpu") if want == "cpu" else (exp_dev == "cuda")
    print(f"[test] VLLM_EXPERTS_LOAD_DEVICE={want} -> expert on {exp_dev}: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
