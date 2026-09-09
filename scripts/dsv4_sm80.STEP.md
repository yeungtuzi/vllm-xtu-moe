# DS-V4-Flash on A100 (SM80) — mainline minimal fix STEP log

Date: 2025-09-09 (session)
Checkout: `/home/user/lvllm/process_data/ref/repos/vllm-mainline` (vLLM 0.1.dev1+g6c73b08de)
Env: `vllm-xiaotu-moe` (conda), test GPU2 only (CUDA_VISIBLE_DEVICES=2).

## Problem
DS-V4-Flash at runtime calls `deep_gemm.get_paged_mqa_logits_metadata(...)`
(→ `_get_paged_mqa_logits_metadata_impl`) and the DeepGEMM MQA/HC GEMM wrappers
`fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`, `tf32_hc_prenorm_gemm`.
DeepGEMM is only built for Hopper/Blackwell-datacenter, so on A100 (SM80) the
kernel launch asserts:
`Assertion error (attention.hpp:270): Unsupported architecture`.

Note: on this host `has_deep_gemm()` is True (SM80) but
`support_deep_gemm()` is False, so `is_deep_gemm_supported()` is False — the
regular grouped/fp8 GEMM paths already route away from DeepGEMM. The ONLY
DeepGEMM calls that would still reach a kernel on SM80 are the paged-MQA
metadata builder + the three MQA/HC wrappers above. That is why the assertion
surfaces at `get_paged_mqa_logits_metadata`.

## Root cause reference (fork `Lvllmds4-x`)
The fork simply hardcodes `has_deep_gemm()` → `return False`
(vllm/utils/import_utils.py), globally disabling DeepGEMM, so EVERYTHING
routes to portable Triton/torch fallbacks. We must NOT do that: mainline must
keep SM90/SM100 behavior. So we apply targeted, additive SM80 routing instead.

## Changes made (mainline only)

### 1. `vllm/utils/deep_gemm.py` — route 3 DeepGEMM-only wrappers when SM80/SM120
The 6 helper functions (`_use_sm12x_mqa_fallback`, `_fp8_mqa_logits_sm12x`,
`_fp8_paged_mqa_logits_sm12x`, `_tf32_hc_prenorm_gemm_sm12x`, ...) were already
appended at file bottom. The wrapper bodies did NOT route to them. Added (at top
of each wrapper body, before `_lazy_init()`):
- `fp8_fp4_mqa_logits`: `if _use_sm12x_mqa_fallback() and q[1] is None: return _fp8_mqa_logits_sm12x(...)`
- `fp8_fp4_paged_mqa_logits`: same guard → `_fp8_paged_mqa_logits_sm12x(...)`
- `tf32_hc_prenorm_gemm`: `if _use_sm12x_mqa_fallback(): return _tf32_hc_prenorm_gemm_sm12x(...)`

These fallbacks live in `vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py`
(+ triton kernels in `sm12x_mqa.py`), both already present in mainline.
`_use_sm12x_mqa_fallback()` returns True on SM120 or (is_cuda and SM80).
On SM90/SM100 it is False → original DeepGEMM path is unchanged.
`q[1] is None` guard preserves the FP4 Q path (mxfp4 requires SM100/indexer
rejects non-100 → FP4 never occurs on SM80).

### 2. `vllm/v1/attention/backends/mla/indexer.py` — skip DeepGEMM scheduler metadata on SM80/SM120
Added module-level helper:
```
def _uses_deep_gemm_scheduler_metadata() -> bool:
    return (current_platform.is_cuda() and has_deep_gemm()
            and not is_device_capability_family(80)
            and not is_device_capability_family(120))
```
and changed the `build()` decode metadata block (was
`if current_platform.is_cuda() and has_deep_gemm():`) to use it.

Important semantics: `get_paged_mqa_logits_metadata` has NO SM12x fallback
counterpart (there is no `_fp8_*_metadata_sm12x`). The SM80/SM120 paged-MQA
fallback `_fp8_paged_mqa_logits_sm12x(...)` does NOT take/use `schedule_metadata`
— it iterates the paged KV in Triton/torch with its own internal chunking. So on
SM80 we simply skip building metadata; `schedule_metadata=` stays
`self.scheduler_metadata_buffer` (unused). This mirrors the fork, which also
skips DeepGEMM metadata on the portable archs (fork's `_uses_deep_gemm_scheduler_metadata()`
excludes SM120; we additionally exclude SM80 because fork globally returns
has_deep_gemm()=False while mainline has it True on SM80).
On SM90/SM100 the helper returns True and the original DeepGEMM metadata build
(including the `indices=` kwarg / `_paged_mqa_logits_schedule_slots` sizing)
runs unchanged.

## Verification
Import/syntax check:
`rm -rf vllm/utils/__pycache__ && python -c "from vllm.utils import deep_gemm"` → OK
`python -c "from vllm.v1.attention.backends.mla import indexer"` → OK
`python -c "from ...sm12x_deep_gemm_fallbacks, sm12x_mqa import"` → OK
Runtime flags on this SM80 host:
`deep_gemm._use_sm12x_mqa_fallback()` → True
`indexer._uses_deep_gemm_scheduler_metadata()` → False

Full mixed test on GPU2 (this machine is all-SM80):
```
CUDA_VISIBLE_DEVICES=2 VLLM_EXPERTS_LOAD_DEVICE=cpu \
TEST_MODEL=/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master \
GLM_LOAD_FORMAT=dummy GLM_MAXLEN=1024 SKIP_TOK=1 OMP_NUM_THREADS=48 \
python /home/user/lvllm/vllm-xiaotu-moe/scripts/model_mixed_test.py
```
Output: `/home/user/lvllm/vllm-xiaotu-moe/scripts/dsv4_sm80.out`
Result: `[glm] CONSTRUCTED in 384.3s` → `[glm] GENERATED 16 tok in 4.62s` →
`[glm] DONE`; exit 0. No DeepGEMM assertion / no new error.
(The GENERATED tokens are all 0 / empty text only because SKIP_TOK=1 disables
the tokenizer — expected dummy behavior.)

## Next
The DeepGEMM-on-A100 blocker is cleared: CONSTRUCTED + GENERATED achieved.
Remaining work (outside this sub-task): real tokenizer/accuracy validation,
and SM90/100 regression confirmation on a Hopper/Blackwell host.
