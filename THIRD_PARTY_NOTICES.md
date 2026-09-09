# Third-party notices / 第三方声明

本项目自身:**Apache-2.0**,Copyright 2026 大河马 (BigHippo) `<dahema@me.com>`
(见 `LICENSE`、`NOTICE`)。

下面是本项目使用或派生的第三方代码/数据,以及我们对相应许可义务的履行方式。
**凡逐字节采用的部分,原始文件的 `SPDX-License-Identifier` / 版权头一律保留。**

---

## 1. vLLM — Apache-2.0

- 来源:<https://github.com/vllm-project/vllm>
- 用法:本项目是 vLLM **主线**(非 fork)的 OOT 插件,通过官方
  `vllm.general_plugins` 入口加载;补丁基线为
  `6c73b08dec2af5052288169663549687ba61f330`。
- 派生:`patches/mainline_sm80_mixed_mode.patch` 等对 vLLM 源码的改动,
  以及 PR3 中取自 vLLM fork 的文件(见下)。
- 义务:保留 Apache-2.0 `LICENSE`;对修改过的文件在提交信息/PR 描述中说明改动。

## 2. Lvllmds4-x (guqiong96) — Apache-2.0

- 来源:<https://github.com/guqiong96/Lvllmds4-x>(vLLM fork),作者
  **guqiong96 `<g_qiong@hotmail.com>`**。
- 用法:**SM80(A100/Ampere)可移植 Triton 路径**逐字节取自该 fork,包括:
  `vllm/v1/attention/backends/mla/sparse_mla_env.py`、`sparse_mla_kernels.py`、
  `vllm/models/deepseek_v4/nvidia/ops/{sm12x_deep_gemm_fallbacks,sm12x_mqa,fp8_einsum}.py`,
  以及 `flashmla.py` / `o_proj.py` 的 SM80 分支。
- 这些文件原本就带 `SPDX-License-Identifier: Apache-2.0` /
  `SPDX-FileCopyrightText: Copyright contributors to the vLLM project`,**保持不变**。
- 义务:在 PR 描述与提交 trailer 中署名原作者(`Co-authored-by:`);说明我们做的改动
  (主线新版 API 适配,例如 `combine_topk_swa_indices` 的 `out=(idx,lens)` 签名)。

## 3. ktransformers (KVCache.AI) — Apache-2.0

- 来源:<https://github.com/kvcache-ai/ktransformers>
- 用法:本项目内置的 CPU 引擎(`xiaotu_moe/`)在 NUMA 调度与 MoE 编排的**设计**上
  与 ktransformers 同源,并以 Apache-2.0 构建块重新实现。
- 义务:保留 Apache-2.0 声明,并在本文中说明血缘关系。

## 4. lk_moe — 专有许可(未复制代码)

- 来源:PyPI `lk-moe`(闭源,专有许可)。
- 用法:仅作为 **ABI/接口契约**的对照物(`MOEConfigV2` 字段、`cpu_decode` /
  `cpu_prefill` 签名)。**没有复制其任何代码**;内置引擎
  (`xiaotu_moe/`)是独立重实现。
- 义务:无(未使用其代码);文中提及仅用于说明兼容性目标。

## 5. 模型与数据

- 测试所用的模型权重(例如 DeepSeek-V4-Flash、GLM-5.3-Flash、Qwen3.8-Flash-Next)
  遵循各自的模型许可,**不随本仓库分发**。
- ShareGPT 等公开数据集仅用于吞吐测试,**不随本仓库分发**(见 `.gitignore`)。

---

## 我们自己的署名格式

- 代码/文档:`Copyright 2026 大河马 (BigHippo) <dahema@me.com>`
- 新文件建议加 SPDX 头:

  ```
  # SPDX-License-Identifier: Apache-2.0
  # SPDX-FileCopyrightText: Copyright 2026 大河马 (BigHippo) <dahema@me.com>
  ```
