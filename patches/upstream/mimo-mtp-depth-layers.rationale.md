# MiMo-V2 MTP depth-layer selection — rationale

Worktree: `/home/user/lvllm/vllm-mtp2` (detached at `43aef91579`, the commit
`vllm-up-133b71e0b` is on).
Upstream reference: `origin/main` = `674b6d95d6`.

## Patches

* `mimo-mtp-depth-layers.patch` — `git diff 43aef91579` in the worktree. This is
  the corrected delta **relative to our local commit** (which already contained
  the hardcoded-3 change `eddc6d0eb7`). Use this to see exactly what was wrong
  with the local patch.
* `mimo-mtp-depth-layers-upstream-mainline.patch` — `git diff origin/main` for
  the four touched files. **This is the self-contained, submittable patch**; it
  applies to current upstream `main` and contains code + tests.

## What was wrong with `eddc6d0eb7` (our local commit)

1. `vllm/config/speculative.py` set `n_predict = _MIMO_V2_*_NUM_MTP_LAYERS`
   (a constant) and wrote it into the draft config's
   `num_nextn_predict_layers`; `mimo_v2_mtp.py` then read that same field back.
   The checkpoint's real `num_nextn_predict_layers` was therefore never
   consulted, so the `min(checkpoint_layers, k)` comment was not what the code
   did.
2. Raising the constants from upstream's `1` to `3` meant a checkpoint that
   ships fewer depths would still instantiate 3 `model.mtp.layers.*` modules;
   `load_weights` would silently skip the depths the checkpoint does not have
   and leave them at their `torch.empty` initialization.

## Fix

* `speculative.py` now reads the checkpoint's own
  `num_nextn_predict_layers` (top-level, with a `text_config` fallback for the
  Omni wrapper) and raises when it is missing or `< 1`. The hardcoded constants
  are deleted, so the two files can no longer drift.
* `mimo_v2_mtp.py::_resolve_num_mtp_layers` builds
  `min(checkpoint_depths, num_speculative_tokens)`: a `k=1` run builds one layer
  (same cost as upstream) and `k=3` builds the full trained chain.
* `mimo_v2_mtp.py::_verify_mtp_depths_loaded` fails closed after loading when a
  built depth layer received no weights, reusing the existing
  `is_mtp_completeness_check_enabled()` hook in
  `vllm/model_executor/model_loader/mtp_validation.py` — the same pattern
  already used by the Inkling and MiniMax-M3 MTP loaders.

## Resulting behaviour

| checkpoint `num_nextn_predict_layers` | `num_speculative_tokens` | layers built | notes |
|---|---|---|---|
| 3 | 1 | 1 | k=1 keeps the old single-layer cost |
| 3 | 3 | 3 | full trained chain (the point of the PR) |
| 3 | 5 | 3 | clamped to the checkpoint; `forward` reuses the last depth via `spec_step_idx % num_mtp_layers` (unchanged upstream behaviour) |
| 1 | 4 | 1 | layer-0 reuse path preserved for single-depth checkpoints |
| 0 / missing | any | — | `ValueError` at config-build time |
| 3 declared, weights for depth 2 absent | 3 | 3 | `ValueError` at load time (fail closed) |

`k > checkpoint depth` is deliberately **not** an error: upstream already
supports depth reuse for `k > num_mtp_layers` (PR #41905 removed the `k != 1`
guard and kept the `spec_step_idx % num_mtp_layers` dispatch). The fail-closed
check is about *weights*, not about `k`: it refuses to run a built depth whose
parameters were never loaded.
