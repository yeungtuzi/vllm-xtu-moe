# vllm-xtu-moe v0.2.3 — 跟进上游 + GLM/MiMo 的 MTP

**发布日期**:2026-09-20
**主题**:把补丁栈**跟进到上游 wheel-backed 提交**;补齐 **GLM-5.3-Flash** 与
**MiMo-V2.5** 的 MTP(投机解码)支持;**重标定显存契约**。

---

## 1. 一句话

补丁栈从上游 `dabc4362b4` **rebase 到 `133b71e0b`**(+11 个补丁 / 40 文件);
**GLM-5.3-Flash 的 MTP 落地并默认开 `SPEC_K=1`**(TPOT 小赚、代价是 KV −27%);
**MiMo-V2.5(310B/15B)在单张 A100-40GB 上端到端跑通**,**MTP k=1 实测 TPOT −9.4%**;
所有正确性门禁全绿。

---

## 2. 上游跟进(rebase 到 `133b71e0b`)

* 目标取**最新有预编译 wheel 的提交**(本机不能源码编译:`nvcc 12.1` vs `torch cu130`,
  且无 `cargo`)。写 handoff 时的 `fbe8a157f` 已被 wheel 索引越过 4 个提交。
* 10 个本地补丁 + 新增 1 个 MiMo 补丁 = **11 个补丁 / 40 文件**;手工解冲突 5 个文件
  (`oracle/fp8.py`、`deepseek_v4/nvidia/model.py`、`deepseek_v41/.../{cache_utils,fused_compress_quant_cache}.py`、
  `deepseek_v41/nvidia/flashmla.py`);`glm5next/nvidia/attention.py → common/attention.py` 由 git rename 自动重指。
* 门禁(SM8x 稀疏 MLA、GPU 预填充三方对拍、FP8 assembly/kernel/vs-CPU、GLM53 层内数值、
  引擎确定性 11/11、fp8 一致性)**全部通过**。
* ⚠️ **升级必读**:上游改了 KV 定容/激活估计 ⇒ **GLM 生产 `GPU_UTIL` 必须从 `0.85` 降到 `0.82`**
  (同一 util 下 KV 池 971,949 → 1,018,328,激活余量被吃掉,两路并发会 OOM;0.82 下 KV 915,487 通过)。

## 3. GLM-5.3-Flash 的 MTP:默认开 `SPEC_K=1`

插件补上了 GLM 的 draft 判定:第 45 层在模型里只构造一次,DSpark 的"同 prefix 二次出现"
认不出;现按"层号 ≥ 目标 `num_hidden_layers`"识别,命中即 **GPU 常驻(3.38 GiB/rank)**。
serve 脚本加 `SPEC_K`(默认 **1**)。

| k | accept(random) | accept(ShareGPT) | out tok/s(256/128 N=8×2) | TPOT | KV 池 |
|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 16.14 / 16.13 | 45.76 / 45.72 ms | **915,487** |
| **1(默认)** | **1.469** | **1.459** | 16.65 / 16.12 | **43.33 / 45.08 ms** | 666,366 |
| 2 † | — | 1.539 | — | — | — |
| 3 † | — | 1.595 | — | — | — |
| 4 † | 1.619 | 1.626 | (单请求)10.19 | 80.82 ms | 508,519 |

† k=2/3/4 由一次独立 k=4 运行的逐位接受率反推(p0≈0.40、p1≈0.14、p2≈0.06、p3≈0.03),
与直接实测的 k=1 有 run-to-run 差异,不要混着比。

**怎么读这张表**:k=1 的 **TPOT 小赚(−3%,在噪声内)**,真正付出的是 **KV 池 −27%**
(256K 并发 3.49× → 2.54×,仍够"2 路 256K")和 **ITL 脉冲化(46 → 64 ms)**;
**k>1 不要开** —— accept 只涨到 1.63,但每步要验证 T=1+k 个 token,CPU 专家路径的
不同专家数近乎翻倍,吞吐反而掉(11.2 tok/s)。想关:`SPEC_K=0`。


## 4. MiMo-V2.5:单卡端到端已支持;MTP 用 **k=1**

310B / 15B active、48 层、256 专家 top-8、Hybrid SWA-128 + DiffKV、官方 FP8 block-128。

* **P0 后端自检通过**:`TRITON_ATTN_DIFFKV`(SM80 必需回退)、
  `CPU Fp8 MoE backend`(插件换后端)、`V2 Model Runner`;
  每层引擎 `E=256 H=4096 I=2048 topk=8 group=128x128 routing=sigmoid/grouped1x1/bias`;
* **P1**:短问答连贯;单卡 A100-40GB(TP=1,专家在 CPU)加载 ~25-30 min;
* **MTP**:检查点带 3 层 dense MTP,本插件已解锁 `min(checkpoint_layers, k)` 建层:
  * **k=1 可用**:accept **1.74~1.87**,TPOT **64.28 → 58.25 ms(平均 −9.4%,N=8×2)**,
    贪心输出与不开 MTP **逐字节相同**;
  * **k=3 坏**:accept 1.016(p0 从 0.83 崩到 0.016)⇒ 反而慢 2.4×,与上游 PR
    [#31180](https://github.com/vllm-project/vllm/pull/31180) 的 *"acceptance rate of 0"* 一致;
  * ⇒ **推荐 `num_speculative_tokens=1`,不要开 k>1**。

启动(TP=1,单卡):`--language-model-only --trust-remote-code --kv-cache-dtype bfloat16`,
`VLLM_EXPERTS_LOAD_DEVICE=cpu`。详见 `docs/MODEL_GUIDES.md` §3。

## 5. 显存契约重标定(升级必读)

rebase 后上游改了 KV 定容/激活估计,**同一组参数下 KV 池变大、激活余量变少**:

| 配置(GLM-5.3,TP=2) | KV 池 | 32k | 两路 14k+15k |
|---|---|---|---|
| v0.2.2(util 0.85) | 971,949 | OK | OK |
| rebase(util **0.85**) | 1,018,328 | OK | **OOM,引擎死亡** |
| rebase(util **0.82**) | 915,487 | OK | OK |

⇒ **GLM 生产必须把 `GPU_UTIL` 降到 `0.82`**。开 MTP 还有额外代价(profile 会把激活峰
从 2.9 低估到 0.84 GiB,叠加 draft 常驻),必须配 `GPU_UTIL=0.82` 或 `GP_PREFILL=0`。

## 6. 已知问题

* **GPU 流式预填充偶发 `illegal memory access`**(报错点 `byte_transpose._ktranspose_bytes_kernel`,
  非确定)。**本轮用满模型复跑 20+ 次未能复现**(6×4096/64、8×(256/128+4096/64)、32k、两路并发、
  完整 A/B 序列全部零错误);缩小范围见 `docs/KNOWN_LIMITATIONS.md` §8.3。
  如需彻底规避:`GP_PREFILL=0`(长 prompt 退 CPU,慢但稳)。
* MiMo 的 **MTP k>1 不可用**(见 §4)。
* DeepSeek-V4-Flash 的基准**不再复测**(过时),文档中一律引用历史数据。

## 7. 升级 / 复现

```bash
# 插件
pip install vllm-xtu-moe-0.2.3-*.whl        # 或源码 pip install -e .

# GLM-5.3-Flash(2×A100-40GB,TP=2)
GPU_UTIL=0.82 bash scripts/serve_glm53_mainline.sh
# 默认已开 MTP k=1;要关:SPEC_K=0 bash scripts/serve_glm53_mainline.sh

# MiMo-V2.5(单卡,TP=1)
VLLM_EXPERTS_LOAD_DEVICE=cpu CUDA_VISIBLE_DEVICES=2 \
python -m vllm.entrypoints.openai.api_server --model <MiMo-V2.5>/snapshots/master \
  --tensor-parallel-size 1 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --language-model-only --trust-remote-code
```

**License**:Apache-2.0。完整参数表与门禁见 `docs/RUNBOOK.md`、`docs/MODEL_GUIDES.md`、
`docs/BENCHMARKS.md`、`docs/KNOWN_LIMITATIONS.md`。
