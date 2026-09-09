## Goal

Make DeepSeek-V4's `o_proj` produce **numerically correct** output on SM 8.x (A100/A800).
Without this, DeepSeek-V4 on Ampere runs but emits garbage tokens.

## Problem (two independent bugs, same op)

`o_proj` does not go through `apply_weights()`. It consumes `wo_a.weight` **directly** through a
fused per-group fp8 einsum, so the weight must keep its on-disk `[N, K]` layout together with its
block scales.

1. **Marlin repacking.** On SM 8.0 the FP8 quant method selects Marlin, whose
   `process_weights_after_loading` repacks `wo_a.weight` into an opaque int32 layout and renames
   the block scales to `weight_scale_inv`. The einsum then reads the packed integers as fp8.
2. **No DeepGEMM kernel.** `vllm.utils.deep_gemm.fp8_einsum` has no SM8.x kernel, so even with the
   correct layout the op cannot run on A100.

## Design

1. **`quantization/fp8.py`** — when the layer is a direct-weight consumer (`layer.is_bmm`) and
   Marlin was selected: pre-dequantize the block-FP8 weight to bf16 **in place**, keeping `[N, K]`
   and the original scale layout, then set `use_marlin = False`.

   ```python
   if self.use_marlin and getattr(layer, "is_bmm", False):
       weight, weight_scale_inv = process_fp8_weight_block_strategy(
           layer.weight, layer.weight_scale_inv)
       scale = weight_scale_inv.to(torch.float32)
       scale = torch.repeat_interleave(scale, block_m, dim=-2)
       scale = torch.repeat_interleave(scale, block_k, dim=-1)
       weight = (weight.to(torch.float32) * scale).to(torch.bfloat16)
       replace_parameter(layer, "weight", weight.data)
       layer.input_scale = None
       self.use_marlin = False
   ```

   This is a correctness path, not a performance path: it trades a little memory for a weight
   layout that the fused einsum can actually read.

2. **`quantization/utils/fp8_utils.py`** — add `_e4m3_uint8_to_f32` and `_f32_to_e4m3_uint8`:
   uint8-typed e4m3 encode/decode, because the `float8e4nv` Triton dtype does not exist on Ampere.

3. **`models/deepseek_v4/nvidia/ops/fp8_einsum.py` (new)** — a portable Triton fp8 einsum for
   SM8.x / SM12x. `o_proj.py` dispatches to it only when the arch has no DeepGEMM fp8_einsum
   (`cap.major in (8, 12)`); the existing SM90 / SM100 / SM110 recipes are untouched.

   The kernel is a straightforward per-group einsum: `o_fp8 @ wo_a.weight` with block scales
   applied on the K axis, fp32 accumulate, bf16 output. On Ampere it decodes fp8 from uint8
   (`DECODE_E4M3`) and takes the weight as bf16 (`B_BF16`).

## Algorithm summary

```
out[b, h, d] = sum_r  o_fp8[b, h, r] * wo_a_weight[r, d] * scale_o[b, h, r/128] * scale_w[r/128, d/128]
```
with `o_fp8` quantized per 128-wide block along the head dim and `wo_a` block-FP8 with 128×128
blocks; both scale tensors are expanded to element-wise factors and folded into the accumulator.
SM90+ uses DeepGEMM's TMA/TCGEN05 recipe; SM8.x uses the Triton kernel added here.

## Files taken from a fork

`fp8_einsum.py` and the `o_proj.py` dispatch come from the vLLM fork
[Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x) (Apache-2.0, author guqiong96). SPDX headers
are retained; only mainline API adaptations were made.

## Testing

- A100-PCIE-40GB + DeepSeek-V4-Flash: `o_proj` output is numerically correct and generation is
  coherent (previously the model produced garbage tokens). See the measurement table in #56120.
- SM90 / SM100 / SM110 code paths are not modified (dispatch is capability-gated).
- +422/−12 across 4 files.

## Related

- #56118 (CPU-offloaded experts) and #56120 (SM80 Triton fallbacks) are the rest of the same
  effort; this PR is the numerical-correctness prerequisite for #56120.
