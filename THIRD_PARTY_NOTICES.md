# Third-party notices / 第三方声明

`vllm-xtu-moe` 自身: **Apache-2.0**,Copyright 2026 大河马 (BigHippo) `<dahema@me.com>`
(见 `LICENSE`、`NOTICE`)。

下面是本仓库使用或派生的第三方代码/数据,以及我们对其许可义务的履行方式。
**凡逐字节采用的部分,原始文件的 `SPDX-License-Identifier` / 版权头一律保留。**

---

## 1. vLLM — Apache-2.0

- 来源: <https://github.com/vllm-project/vllm>
- 用法: `vllm-xiaotu-moe/` 是 vLLM **主线**(非 fork)的 OOT 插件,通过官方
  `vllm.general_plugins` entry point 加载;补丁基线为
  `6c73b08dec2af5052288169663549687ba61f330`。
- 派生: `vllm-xiaotu-moe/patches/mainline_sm80_mixed_mode.patch` 里对 vLLM 源码的改动,
  以及 PR3 中取自 vLLM fork 的文件(见下)。
- 义务: 保留 Apache-2.0 `LICENSE`;对修改过的文件在提交信息/PR 描述中说明改动。

## 2. Lvllmds4-x (guqiong96) — Apache-2.0

- 来源: <https://github.com/guqiong96/Lvllmds4-x>(vLLM fork),作者
  **guqiong96 `<g_qiong@hotmail.com>`**。
- 用法: **SM80(A100/Ampere)可移植 Triton 路径**逐字节取自该 fork,包括:
  `vllm/v1/attention/backends/mla/sparse_mla_env.py`、`sparse_mla_kernels.py`、
  `vllm/models/deepseek_v4/nvidia/ops/{sm12x_deep_gemm_fallbacks,sm12x_mqa,fp8_einsum}.py`,
  以及 `flashmla.py` / `o_proj.py` 的 SM80 分支。
- 这些文件原本就带 `SPDX-License-Identifier: Apache-2.0` /
  `SPDX-FileCopyrightText: Copyright contributors to the vLLM project`,**保持不变**。
- 义务: 在 PR 描述与提交 trailer 中署名原作者(`Co-authored-by:`);说明我们做的改动
  (主线新版 API 适配:如 `combine_topk_swa_indices` 的 `out=(idx,lens)` 签名)。

## 3. ktransformers (KVCache.AI) — Apache-2.0

- 来源: <https://github.com/kvcache-ai/ktransformers>
- 用法: `lk_moe` 官方随包发布的 `THIRD_PARTY_LICENSES` 指出其 NUMA 调度器
  (`backend_numa.cpp`)与 MoE 骨架(`moe.cpp/h`)衍生自 ktransformers;**本仓库内置的
  CPU 引擎**(`vllm-xiaotu-moe/xiaotu_moe/` + `vllm-xiaotu-moe/csrc/`,源码来自独立项目
  `xiaotu-moe`,该项目已转私有)是对该契约的开源重实现,其中 NUMA 调度与 MoE 编排的
  **设计**同源。
- 义务: 保留 Apache-2.0 声明,并在本文与各 README 中说明血缘关系。

## 4. lk_moe — 专有许可(未复制代码)

- 来源: PyPI `lk-moe`(闭源,专有许可)。
- 用法: 仅作为 **ABI/接口契约**的对照物(`MOEConfigV2` 字段、`cpu_decode` /
  `cpu_prefill` 签名)。**没有复制其任何代码**;内置引擎
  (`vllm-xiaotu-moe/xiaotu_moe/` + `csrc/`)是独立重实现。
- 义务: 无(未使用其代码);文中提及仅用于说明兼容性目标。

## 5. 模型与数据

- **DeepSeek-V4-Flash-0731**: 仅作为测试权重使用,遵循其自身模型许可
  (见模型目录内的 `LICENSE`)。
- **ShareGPT_V3_unfiltered_cleaned_split.json**: 公开数据集,仅用于吞吐测试,
  **不随本仓库分发**(见 `.gitignore`)。

---

## 我们自己的署名格式

- 代码/文档: `Copyright 2026 大河马 (BigHippo) <dahema@me.com>`
- 新文件建议加 SPDX 头:

  ```
  # SPDX-License-Identifier: Apache-2.0
  # SPDX-FileCopyrightText: Copyright 2026 大河马 (BigHippo) <dahema@me.com>
  ```

---
## 修订记录

- **2026-09-09(第 2 版)** — 项目边界澄清:把对 `xiaotu-moe/`(独立项目,已转私有)
  的引用改为**本仓库内置引擎**路径(`vllm-xiaotu-moe/xiaotu_moe/` + `csrc/`);
  明确 `vllm-xtu-moe` 是 vLLM 主线插件。依据:用户 2026-09-09 澄清。
- **2026-09-09(第 1 版)** — 新建。按用户 D8 决策:我方 Apache-2.0、署名
  "大河马(BigHippo) dahema@me.com";第三方按各自 license 要求署名/声明。
  依据:各仓库 `LICENSE`/SPDX 头 + `lk_moe` 的 `THIRD_PARTY_LICENSES`。
