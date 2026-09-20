# vllm-xtu-moe

> **XTU = X Transformers Unity**(读音:汉语「小兔」)—— 一个**高性能 MoE 推理加速层**。
>
> 这是一个面向**超大 MoE 模型**的 **vLLM 插件**,核心思路是把计算拆开:
>
> * **GPU** 负责注意力、KV cache、embedding 等**非专家部分**;
> * **CPU** 负责**专家权重**与**专家计算**;
> * 对**长 prompt**,还会把**一部分专家计算流式搬到 GPU**。
>
> 也就是说:在**"显存放不下专家权重"**的情况下,仍然让大 MoE 模型跑起来,
> 而且**尽量不改主线 vLLM**。
>
> **项目重点**
>
> * **DeepSeek / GLM / MiMo** 这类**超大 MoE**
> * **CPU + GPU 混合推理** · **低显存需求**
> * **高性能长上下文推理**
> * **兼容主线 vLLM,不需要 fork**
> * 主要面向**大模型部署与工程优化**

[**English**](README_EN.md) · 中文(默认)

> **📌 当前版本:v0.2.3**(2026-09-20)—— **跟进上游 + GLM/MiMo 的 MTP**:
> 补丁栈 rebase 到上游 `133b71e0b`;**GLM-5.3-Flash 的 MTP 落地并默认开**;
> **MiMo-V2.5(310B/15B)单卡端到端跑通**;GLM 显存契约重标定(`GPU_UTIL` → **0.82**)。
> 发行说明:[`RELEASE_NOTES_v0.2.3.md`](RELEASE_NOTES_v0.2.3.md)

---

## 目标与愿景

1. **任意 MoE 模型**:不绑死某一代架构。已跑通 DeepSeek-V4 / V4.1 系列、
   **GLM-5.3-Flash**(含 A100 / SM80 的注意力后端)与 **MiMo-V2.5**。
2. **任意 x86 指令集**:`scalar → AVX2 → AVX-512(base/VNNI/BF16/VBMI)`,
   运行时按 `/proc/cpuinfo` 自动选最高可用变体。
3. **显存优先级固定不变**:`KV 池 → GPU 预填充 staging → 投机解码 draft → 激活工作区(∝ MBT)`;
   任何新功能都不得把这条顺序往后挤。
   > **2026-09-20 修订**:①**去掉"专家层常驻"** —— 实测效果很差(每层 3.36 GiB,换来的收益
   > 抵不上它对 32K 预填充的挤压,用户裁定不再列为优先级);
   > ②**补上"激活工作区"** —— 它正比于 chunk(=MBT),是之前连续两次长 prompt OOM 的真凶,
   > 却一直没被列进优先级。
4. **内存只用一份**:专家权重在内存里只有 **1 份**(不是"参考源张量 + 引擎副本"两份)。
5. **引擎效率对标业界最强**:`xiaotu_moe` 与 `lk_moe` 在**同一台机器、同一份权重、
   同一线程数**下对比,目标是不低于其 90%。

---

## 实测平台

下文所有性能数字都在这台机器上取得:

| 项 | 配置 |
|---|---|
| CPU | 2× AMD EPYC 9654(192 物理核 / 384 线程,8 NUMA node) |
| 内存 | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| 系统 | Ubuntu 22.04 · conda env `lvllm` |

---

## 性能(最新版本 **v0.2.3**)

> **v0.2.3 变更**:补丁栈 rebase 到上游 `133b71e0b`;**GLM 生产 `GPU_UTIL` 从 `0.85` 降到 `0.82`**
> (上游改了 KV 定容 ⇒ 0.85 下两路并发会 OOM);GLM 的 MTP **默认开 `SPEC_K=1`**
> (decode 小赚 +3%,代价是 **KV 池 −27%**、单 token 间隔脉冲化);
> **MiMo-V2.5 新增支持**,MTP k=1 实测 **decode 15.6 → 17.2 tok/s(+10%)**。
> 详见 [`RELEASE_NOTES_v0.2.3.md`](RELEASE_NOTES_v0.2.3.md)。

> **口径(全 README 统一)**:`prefill (tok/s) = prompt_tokens / TTFT`(首 token 之前的预填充速率)、
> `decode (tok/s) = 1000 / TPOT`(首 token 之后的解码速率,**不含 prefill**)。
> 两者都由同一次 `vllm bench serve` 的 TTFT / TPOT 换算而来。
> **不再公布 `output tok/s`(含 TTFT 的混合值)、TPOT、ITL。**
> ⚠️ **短 prompt 的 prefill 会被固定开销压低**,必须按长 prompt 读(见下表最后一段说明)。
> C=1 与 C=2 的速率都是**单流**值(每路各自),不是聚合吞吐。

| 模型(最优配置) | prompt | 实际 token | 并发 | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>TP=2 · util 0.82 · GPU 预填充<br>KV 池封顶 2 GiB · MBT 4096 | 短 | 151 | 1 | **114.8** | **21.7** |
| | 短 | 126 | 4 | **36.5** | 8.2 |
| | 长 | 4,538 | 1 | **178.3** | 21.3 |
| | 长 | 4,518 | 4 | **92.9** | 2.2 |
| **MiMo-V2.5**<br>单卡 · maxlen 16K · KV 封顶 4 GiB<br>MTP k=1 · GPU 预填充 | 短 | 153 | 1 | 70.9 | **18.0** |
| | 短 | 153 | 4 | 59.9 | 6.0 |
| | 长 | 4,148 | 1 | 179.1 | 17.4 |
| | 长 | 4,148 | 4 | **1,844.8** | 6.4 |
| **DeepSeek-V4.1-Flash**<br>TP=2 · dspark k=5 · GPU 预填充<br>KV 封顶 0.5 GiB · MBT 8192 | 短 | 84 | 1 | **217.7** | **21.3** |
| | 短 | 84 | 4 | 83.7 | 10.7 |
| | **长** | 4,796 | 1 | **330.3** | **24.7** |
| | **长** | 4,796 | 4 | **2,595.2** | 10.6 |

> ### ⚠️ 2026-09-20 更正:C=4 那一档曾有**前缀缓存假象**
> 早期版本的 C=4 数据(C=1 与 C=4 **用同一批 prompt**、服务端开前缀缓存)出现
> **MiMo 1,845 / DeepSeek-V4.1 2,595 tok/s** 的虚高值 —— 那是 **C=1 跑完后留下的缓存命中**,
> 根本没有做预填充。冷/热对照实测:**同一批 prompt 冷 31,896 ms vs 热 838 ms(38×)**。
> **已按「两侧使用互不重叠的 prompt 切片」重测**(上表 GLM 两行已替换;MiMo/V4.1 待补)。
> **硬规则:任何 C=1 vs C>1 的对照,两侧必须用不重叠的 prompt 切片。**

> **数据集 = ShareGPT 真实对话**(经长度筛选成"短"~150 tok 与"长"~4.5K tok 两档,
> 用 `--dataset-name custom` 喂入以绕开 vLLM ShareGPT 加载器的 **1024-token 硬上限** ——
> 该上限会让所有"长 prompt"静默退化成短 prompt),**前缀缓存开**(产品行为),
> 每格先跑一遍不同切片 warmup 丢弃,再测。`--backend openai-chat`(V4.1 无 chat_template,用 `openai`)。
>
> ⚠️ **两个反直觉的读数**:
> ① **短 prompt 的 prefill 被固定开销压低**(GLM 126 tok 只有 110,4,918 tok 才到 207);
> ② **合批能把 prefill 抬一个数量级** —— MiMo 同一 prompt,C=1→C=4 让 prefill 从 179 涨到
> **1,845 tok/s**(4 路把 16,592 token 合成大批,每 chunk 的固定搬运成本摊到 4 倍 token 上)。
> **这是本项目唯一实测到 >1000 tok/s 的服务级数据**(另一处是引擎侧 1725 tok/s @13.8K chunk,
> 见下面的 GPU 预填充小节)。

**每行的配置与样本量**

| 模型 | 服务配置 | 样本 / 备注 |
|---|---|---|
| GLM-5.3-Flash | `GPU_UTIL=0.82`,**投机默认关**(2026-09-20 起,理由见 [`RUNBOOK.md`](docs/RUNBOOK.md) §3.6b),GPU 预填充 ON(阈值 1500),**KV 池封顶 2 GiB + MBT 4096**(长格必需;不封顶会 CUDA OOM) | 短格 N=16、长格 N=8,均先 warmup。KV 池:关 MTP **915,487** / 开 MTP **666,366**(−27%);交付配置 **256K × 2 路** |
| MiMo-V2.5 | 单卡 TP=1,Hybrid SWA-128 + DiffKV(`TRITON_ATTN_DIFFKV`),MTP k=1,GPU 预填充 ON,`maxlen 16384` + KV 封顶 4 GiB | 短格 N=16、长格 N=8。**MTP k=3 不可用**(accept 1.016 ⇒ 反而慢 2.4×);贪心输出与不开 MTP **逐字节相同**;加载 25–30 min |
| DeepSeek-V4.1-Flash | TP=2 · MBT=8192 · dspark k=5 · GPU 预填充 ON · KV 封顶 0.5 GiB | ⚠️ 该快照**没有 `chat_template`** ⇒ bench 必须 `--backend openai --skip-chat-template`(否则客户端直接抛异常,**见 [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) §9.1**) |

**投机的收益随并发反转(C=1 赚、C≥4 亏)** —— DeepSeek-V4.1 的 dspark,ShareGPT 口径:

| 并发 | 投机关 (输出 tok/s) | 投机开 (输出 tok/s) |
|---|---|---|
| C=1 | 13.79 | **18.38(+33%)** |
| C=4 | **41.62** | 31.03(**−25%**) |
| C=8 | **48.56** | 41.26(−15%) |

⇒ 原因是 **draft 与 target 抢同一份 CPU 专家算力**。GLM 的 MTP 连 C=1 都只有 **+2~3%**
(单层回收、接受率仅 1.46),所以**默认关**。
**关键读法**:投机在 **C=1 赚、C≥4 亏**(draft 与 target 抢同一份 CPU 专家算力);
**random 口径下这个 +33% 完全看不到** —— 这就是为什么不能用 random 做产品摘要。

**怎么读 prefill** —— 两条规律,都不直观:

1. **prompt 越长越高**(固定成本被摊薄):GLM **110 → 207**(126 → 4,918 tok)、
   MiMo **71 → 179**(153 → 4,148 tok)、DeepSeek-V4.1 **218 → 330**(84 → 4,796 tok);
2. ⚠️ **并发越高、每请求 prefill 越慢**(预填充是**共享的串行资源**):
   DeepSeek-V4.1 长 prompt **C=1 330 → C=4 131 tok/s**、GLM **178 → 93**。
   **~~"合批能抬一个数量级"~~ 是错的,已作废**(原因见下面的更正说明)。
   **短 prompt 看着"慢"**是因为 84-153 tok 的活儿里固定开销占大头。
3. **decode 反过来**:并发越高越低(GLM 长格 C=1 21.6 → C=4 **2.8**),这是预填充与解码
   争同一份 CPU 专家算力的必然结果。

**其它要点**

* **GLM 交付配置 = 256K 上下文 × 2 路并发**;0.2.3 起 `GPU_UTIL` **必须 `0.82`**
  (上游改了 KV 定容,0.85 下两路并发会 OOM):KV 池 915,487(关 MTP)/ 666,366(开 MTP),
  即 **MTP 的真代价是 KV −27%**(256K 并发 3.49× → 2.54×),且**单 token 间隔脉冲化**
  (一步吐 1–2 个 token,流式体验变差)。1M 不在目标内(需 fp8 KV,已决定不做)。
* **正确性**:引擎确定性门禁 11/11;层门禁 rms_rel 4.4e-3;29,746-token 长文密钥检索完全命中;
  3 路 ~8K 并发(限两路)三个密钥全部正确;0 OOM。
* **MiMo 开 GPU 流式预填充**要用更低的 `GPU_UTIL`(实测 0.65),因为 staging **12.75 GiB/rank**
  (比 GLM 的 7.59 大,`E=256×I=2048` 更宽);8K 上下文下 5.8× 并发仍够用。
* FP8 引擎内层另有一个**默认关闭**的加速开关 `XIAOTU_MOE_FP8_BF16_MMA=1`
  (AVX512-BF16 `vdpbf16ps`,M≥6 快 1.17–1.20×,代价是权重舍入 bf16:两路 rms_rel 3.6e-3)。

### DeepSeek-V4.1-Flash 服务级:与参考实现 `lk_moe` 的等参数 A/B

两个 arm 跑在**同一个 conda env**、唯一变量是 CPU MoE 引擎;prompt 逐字节相同、线程数都是 60。
比值为 **我们 / `lk_moe`**,**> 1.0 表示我们更快**。

| prompt / output | `lk_moe` prefill (tok/s) | 我们 prefill (tok/s) | 比值 | `lk_moe` decode (tok/s) | 我们 decode (tok/s) | 比值 |
|---|---|---|---|---|---|---|
| 256 / 32 | 118.9 | 107.1 | **0.900×** | 44.7 | 31.1 | **0.695×** |
| 256 / 1024 | 118.6 | 105.1 | **0.886×** | 44.9 | 33.5 | **0.746×** |
| 8192 / 32 | 127.5 | 115.0 | **0.902×** | 45.4 | 34.3 | **0.756×** |
| 8192 / 1024 | 127.6 | 120.9 | **0.948×** | 44.8 | 34.7 | **0.775×** |

⇒ 服务级我们**仍慢**:prefill 慢 **5-11%**、**decode 慢 29-44%**,但比上一版已收窄 **+8%~+15%**。

### GPU 预填充(把长 prefill 交给 GPU)

把长 prefill 的专家计算逐层流式搬上 GPU,支持两种专家格式:**MXFP4**(DeepSeek-V4.1-Flash 等)与
**FP8 e4m3 block-128**(GLM-5.3-Flash)。

**⚠️ 收益完全由 chunk 大小决定** —— 每个 chunk 都要把**该步所有层的专家权重整份 H2D 搬一遍**
(与这个 chunk 里有多少 token **无关**)⇒ **单 chunk 成本近似固定**,chunk 越大摊得越薄。
所以"收益是 1.3× 还是 4.4×"取决于**这个模型能不能用大 chunk**:

| GLM-5.3-Flash(同一 prompt,切不同 chunk) | 纯 CPU | GPU 流式 | 收益 |
|---|---|---|---|
| **2176**(GLM 的**实际上限**,被 KDA `block_size` 钉死) | 11.3 s | 9.7 s | 1.2× |
| 4096(反事实:去掉 KDA 约束) | 21.3 s | 9.7 s | **2.2×** |
| 8192(反事实) | 42.6 s | 9.7 s | **4.4×** |

⇒ **GPU 那一列三种 chunk 下都是 9.7 s**,这就是"固定成本"的直接证据;CPU 那列随 chunk 线性涨。
**GLM 的 1.2× 不是这条路没用,是被 2176 封顶。**

> **⚠️ 已定论(2026-09-20 实测)**:GLM 的 chunk **不会**随 `MBT` 增长 —— 同一 prompt 把 `MBT` 从
> 2048 提到 8192(4×),TTFT 只从 24.4 s 降到 22.9 s(**仅 6%**,而真涨 chunk 应降到 ~7 s)⇒
> **chunk 就是钉在 ~2176,天花板 ~380 tok/s,是 KDA state 分页粒度的物理上限**,不是实现缺陷;
> 要突破得改 KDA 的 block 粒度(引擎级改动)。**DSV4.1 / MiMo 不受此限** —— 它们能吃满 MBT,
> 长 prompt + C=4 时分别到 **2,595 / 1,845 tok/s**(见上面的性能总表)。

**DeepSeek-V4.1-Flash 不受此限,能用大 chunk,所以是另一个量级**(同一个 **13.8K** prompt):

| 版本 | TTFT | prefill |
|---|---|---|
| 修复前 | 76.5 s | 181 tok/s |
| **修复后**(环形槽复用 + 设备级判定 + 快速转置) | **8.008 s** | **1725 tok/s** |

⇒ **9.6×**;对照 CPU 预填充(13.8K × 3.9 ms ≈ 54 s)是 **6.7×**。

**每层成本分解(GLM 实测)**:`装配 207 ms`(权重 H2D 3.62 GB/rank)+ `GPU MoE 23 ms` —— **装配占 90%**,
所以优化点是**把装配与 attention 重叠**(默认开启的 side stream),不是"搬得更快"。
隔离环境实测装配 **134.9 ms = 26.86 GB/s**,即本机 H2D 天花板(1-D 与 pitched 2-D 同速,
锁页/NUMA 交错/两 rank 并发都不改变);服务里的 207 ms 是与 vLLM 自身 PCIe 流量
(TP=2 的 all-reduce,GPU0/1 之间无 NVLink)争用的结果。

**GLM 专用注意**:chunk 被 `block_size=2176` 钉死,插件默认阈值 4096 **永远不会触发**
⇒ `scripts/serve_glm53_mainline.sh` 已把 GLM 默认改为 `GPU_PREFILL_MIN=1500`(实测盈亏平衡 ~1300);
MXFP4 侧默认阈值 ≥4096 才划算。

CPU 侧另有一个可选开关 `XIAOTU_MOE_FP8_BF16_MMA=1`:FP8 内层改走 AVX512-BF16
`vdpbf16ps`,**M≥6 快 1.17-1.20×**(全 CPU 预填充端到端 TTFT 1.12×),代价是权重被舍入到
bf16(两路 rms_rel 3.6e-3),**因此默认关闭**;`M=1` 的单流解码仍走精确 fp32 路。

开启方式与显存配方见 [`docs/RUNBOOK.md`](docs/RUNBOOK.md);

**完整的服务级对照、口径说明与复现命令**:[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)、
[`RELEASE_NOTES_v0.2.1.md`](RELEASE_NOTES_v0.2.1.md)。

---

## 快速开始

```bash
# 1) 主线 vLLM(本插件是插件,不需要 fork)
pip install vllm==2.5.0

# 2) 本插件(发行名 vllm-xtu-moe,**不发 PyPI**;二选一)
# (a) 从 GitHub Release 附件装:wheel 里已含 6 个 ISA 变体,无需本地编译器
pip install ./vllm_xtu_moe-0.2.2-cp312-cp312-manylinux_2_34_x86_64.whl
# (b) 源码安装(需要本地编译器):
# CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh && pip install -e .

# 3) 跑一个专家权重放不进显存的 MoE 模型
VLLM_EXPERTS_LOAD_DEVICE=cpu \
vllm serve <MODEL_DIR> --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 65536 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95
```

逐模型配方、显存核算、自检与排错 → **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**、
**[`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md)**。

---

## 版本变更(简)

| 版本 | 主题 |
|---|---|
| **v0.2.3** | **跟进上游 + GLM/MiMo 的 MTP** —— 补丁栈 rebase 到上游 `133b71e0b`(11 补丁/40 文件);**GLM-5.3-Flash 的 MTP 落地**(draft 层识别 + GPU 常驻,`SPEC_K=1..4`);**MiMo-V2.5(310B/15B)单卡端到端支持**,MTP k=1 decode +10%;显存契约重标定(GLM `GPU_UTIL` 0.85 → **0.82**) |
| **v0.2.2** | **支持 GLM-5.3-Flash** —— FP8 GPU 预填充接线(4K prompt TTFT 29.3 → 22.8 s)、256K × 2 路并发交付配置;修掉 e4m3 次正规数解码缺陷 + 新增全码字门禁 |
| **v0.2.1** | **针对引擎的显著性能优化** —— CPU MoE 引擎在全部真实形状上**反超 `lk_moe`**;DeepSeek-V4-Flash 同步受益 |
| v0.2 | DeepSeek-V4.1-Flash 全链路可用(1M 上下文 + GPU 预填充 + 投机解码)+ CPU 预填充路径优化 |
| v0.1.0 | 首个公开版:混合模式(CPU 专家 + GPU 其余)、AVX2 / AVX-512 多 ISA、DeepSeek-V4 系列 |

改动清单、性能对照与运行参数变更:
[**v0.2.3**](RELEASE_NOTES_v0.2.3.md) · [**v0.2.2**](RELEASE_NOTES_v0.2.2.md) · [**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
(归档:[v0.2pre](RELEASE_NOTES_v0.2pre.md))

---

## 文档

**面向使用者 —— [`docs/`](docs/)**

| 文档 | 内容 |
|---|---|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | **运行手册**:安装、启动参数、显存核算、GPU 预填充配方、JIT 缓存、发布流程 |
| [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) | 逐模型的资源需求、启动命令与性能 |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | 实测数据与复现方式 |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | 已知限制与不支持的组合 |
| [`docs/INSTALL_MAINLINE.md`](docs/INSTALL_MAINLINE.md) | 主线 vLLM 环境准备 |

**面向开发者** —— 架构与主线集成、GPU 预填充实现、上游漂移、调优记录与全部内部报告,
**属于内部开发文档,不随本仓库发布**(只保留在本地工作副本里)。

---

## 致谢

本项目受到 **KTransformers** 和 **Lvllm** 项目启发,尤其是计算引擎 **`xiaotu-moe`** 深度借鉴了 **lk-moe** 思路,
**`xiaotu-moe` 全部代码均为独立编写**,特此致谢。

* KTransformers —— <https://github.com/kvcache-ai/ktransformers>
* Lvllm(及其 CPU MoE 引擎 **lk-moe**;本项目对照的 `lk_moe` 2.4.2 即来自此仓库)—— <https://github.com/guqiong96/Lvllmds4-x>
* 同时感谢 **vLLM**(<https://github.com/vllm-project/vllm>)提供的插件式扩展点,使本项目无需 fork 即可接入。

---

## 许可

Apache-2.0。第三方组件与致谢清单见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`NOTICE`](NOTICE)。
