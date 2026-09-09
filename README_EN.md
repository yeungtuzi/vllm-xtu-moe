# vllm-xtu-moe

> **XTU = X Transformers Unity** (pronounced *"xiao tu"*). This repository is the
> **hybrid inference plugin for upstream vLLM**: MoE expert weights live on the
> CPU, attention and long prefill stay on the GPU.
>
> Internal identifiers (`xiaotu_moe` / `vllm_xiaotu_moe` package names,
> `XIAOTU_*` env vars, directory names) are deliberately unchanged so existing
> configs and callers keep working.

[**中文版**](README_CN.md) · English (default)

---

## ⚠️ Project boundaries (read first)

| Project | What it is | Status |
|---|---|---|
| **`vllm-xtu-moe`** (this repository) | **Upstream-vLLM plugin**: hybrid inference (experts on CPU, attention / long prefill on GPU); registers CPU compute backends into vLLM's `FusedMoEFactory` per quantization format | the published project |
| **`xiaotu-moe`** (separate repository) | Standalone **CPU MoE engine project**: an open reimplementation of the closed-source `lk_moe`, originally built as a drop-in replacement inside the `Lvllm` / `Lvllmds4-x` vLLM forks | **now private, reference only**; its engine source is vendored here as the built-in compute kernel (`vllm-xiaotu-moe/xiaotu_moe/` + `csrc/`) |
| **`Lvllmds4-x`** (third-party fork) | Someone else's vLLM fork | code provenance for PR2/PR3 only (Apache-2.0, attribution kept); **not a runtime dependency** |

In one line: **the engine comes from `xiaotu-moe`, the project itself is
`vllm-xtu-moe`, and it runs on upstream vLLM (not a fork).**

> This repository therefore no longer contains the standalone `xiaotu-moe`
> project directory. Engine-era material still kept here (fork integration notes,
> the engine reports, thread-geometry / perf comparisons) is marked as
> historical reference and does **not** represent this project's roadmap.
> See [`vllm-xiaotu-moe/docs/BACKLOG.md`](vllm-xiaotu-moe/docs/BACKLOG.md).

---

## What this repository contains

```
vllm-xtu-moe/
├── vllm-xiaotu-moe/     # ★ the plugin project (see its README for details)
│   ├── vllm_xiaotu_moe/ #   mainline integration layer
│   ├── xiaotu_moe/      #   bundled CPU engine (Python bindings + prebuilt .so)
│   ├── csrc/            #   bundled CPU engine (C++ kernels)
│   ├── patches/         #   upstream PR snapshots / mainline patches / RFC
│   ├── scripts/         #   benchmarks, numeric tests, end-to-end smoke tests
│   └── docs/            #   experiment report, backlog, designs
├── report/              # measurement artifacts referenced by the report
├── results.txt          # append-only measurement log
├── LICENSE, NOTICE, THIRD_PARTY_NOTICES.md
```

Full documentation, quick start, current status and the TODO/decision ledger live
in **[`vllm-xiaotu-moe/README.md`](vllm-xiaotu-moe/README.md)** and
**[`vllm-xiaotu-moe/docs/BACKLOG.md`](vllm-xiaotu-moe/docs/BACKLOG.md)**.

## Why it exists

Upstream vLLM ships quantized CPU MoE kernels (MXFP4 / FP8 / INT4 / INT8) that
**all require AMX** (Intel-only). On an AMD EPYC host there is therefore no
usable CPU kernel for MXFP4/FP8, while expert weights are far too large for VRAM
(DeepSeek-V4 ~137 GiB, Qwen3.8-Flash-Next 185 GB). This plugin supplies no-AMX
AVX512 CPU kernels for those formats and wires them into upstream vLLM, so
`VLLM_EXPERTS_LOAD_DEVICE=cpu` just works.

## License

Apache-2.0 for this project and the bundled engine. Third-party code (the SM80
port taken from `Lvllmds4-x`) keeps its Apache-2.0 SPDX attribution; no code
from the closed-source `lk_moe` is copied. See `THIRD_PARTY_NOTICES.md`.

---

## Revision history

- **2026-09-09 (rev 2)** — Repo front page rewritten: the project is now described
  as the **upstream-vLLM plugin** (`vllm-xtu-moe`); the standalone `xiaotu-moe`
  project directory was removed from this repository (kept locally, gitignored,
  project now private) and the "two components, one project" framing was dropped.
- **2026-09-08 (rev 1)** — Initial front page (engine-project framing).
